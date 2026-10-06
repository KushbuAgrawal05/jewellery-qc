"""Folder pipeline for Jewellery Casting QC.

Watches an inbox folder (default: ~/Desktop/qc_inbox), runs every new image through the
same models as the Streamlit app (via engine.py), and writes:

    <outbox>/results.csv        one row per image (verdict, probability, top-3 types)
    <outbox>/reports/*.png      Grad-CAM explanation image for images flagged as defective
    <outbox>/processed/         originals after a successful run
    <outbox>/failed/            files that could not be read / scored
    <outbox>/duplicates/        images identical to one already tested (skipped)

Put this file next to engine.py and model_info.json, then:

    python batch_qc.py                      # keep watching the inbox
    python batch_qc.py --once               # process what's there now and exit
    python batch_qc.py --inbox "D:/photos" --threshold 0.4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import engine  # same module the Streamlit app uses

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def wait_until_stable(path: Path, tries: int = 10, delay: float = 0.5) -> bool:
    """True once the file size stops changing (i.e. the copy to the folder has finished)."""
    last = -1
    for _ in range(tries):
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == last and size > 0:
            return True
        last = size
        time.sleep(delay)
    return False


def unique_dest(folder: Path, name: str) -> Path:
    dest = folder / name
    if dest.exists():
        dest = folder / f"{dest.stem}_{datetime.now():%Y%m%d_%H%M%S}{dest.suffix}"
    return dest


def process_one(models, path: Path, outbox: Path, threshold: float, reports: bool, sha1: str) -> dict:
    data = path.read_bytes()
    pil = engine.load_image(data)
    x = engine.to_model_input(pil)
    res = engine.analyse(models, x)

    p_def = res["p_defect"]
    probs = res["type_probs"]
    top = np.argsort(probs)[::-1][:3]
    is_defect = p_def >= threshold
    verdict = "DEFECT" if is_defect else "OK"
    names = [models.classes[int(i)]["name"] for i in top]

    if reports and is_defect and models.multi.xai_ready:
        display = engine.fit_for_display(pil)
        cam = models.multi.gradcam(x, int(top[0]))
        overlay = engine.heatmap_overlay(display, cam)
        boxed, _ = engine.derive_boxes(display, cam)
        header = (f"{verdict} | defect prob {p_def:.1%} | top type: {names[0]} "
                  f"({probs[top[0]]:.0%}) | {path.name}")
        png = engine.make_report_png(
            header, [("Original", display), ("Grad-CAM heatmap", overlay), ("Regions of interest", boxed)]
        )
        (outbox / "reports" / f"{path.stem}_report.png").write_bytes(png)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": path.name,
        "sha1": sha1,
        "verdict": verdict,
        "defect_probability": round(p_def, 4),
        "threshold": threshold,
        "type_1": names[0], "type_1_prob": round(float(probs[top[0]]), 4),
        "type_2": names[1], "type_2_prob": round(float(probs[top[1]]), 4),
        "type_3": names[2], "type_3_prob": round(float(probs[top[2]]), 4),
    }


def load_seen(csv_path: Path) -> set[str]:
    """Fingerprints of images already tested, read from results.csv (the 'register')."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["sha1"] for r in csv.DictReader(f) if r.get("sha1")}


def append_csv(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def sweep(models, inbox: Path, outbox: Path, threshold: float, reports: bool, seen: set[str]) -> int:
    files = sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    done = 0
    for path in files:
        if not wait_until_stable(path):
            continue  # still being copied; pick it up next sweep
        try:
            sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
            if sha1 in seen:  # exact same image was already tested
                print(f"[SKIPPED] {path.name}: already in results.csv")
                shutil.move(str(path), unique_dest(outbox / "duplicates", path.name))
                continue
            row = process_one(models, path, outbox, threshold, reports, sha1)
            append_csv(outbox / "results.csv", row)
            seen.add(sha1)
            shutil.move(str(path), unique_dest(outbox / "processed", path.name))
            print(f"[{row['verdict']:6}] {path.name}  p={row['defect_probability']:.1%}  "
                  f"top={row['type_1']}")
            done += 1
        except Exception as e:  # unreadable image, etc.
            print(f"[FAILED] {path.name}: {e}")
            shutil.move(str(path), unique_dest(outbox / "failed", path.name))
    return done


def main() -> None:
    desktop = Path.home() / "Desktop"
    ap = argparse.ArgumentParser(description="Jewellery casting QC folder pipeline")
    ap.add_argument("--inbox", type=Path, default=desktop / "qc_inbox")
    ap.add_argument("--outbox", type=Path, default=desktop / "qc_results")
    ap.add_argument("--threshold", type=float, default=0.50, help="defect probability cut-off")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between folder checks")
    ap.add_argument("--once", action="store_true", help="process current files and exit")
    ap.add_argument("--no-reports", action="store_true", help="skip Grad-CAM report images (faster)")
    args = ap.parse_args()

    for d in (args.inbox, args.outbox / "reports", args.outbox / "processed", args.outbox / "failed", args.outbox / "duplicates"):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    # Looks in models/ first; otherwise downloads from MODEL_BASE_URL (if set) and caches the files.
    models = engine.load_models(os.environ.get("MODEL_BASE_URL"))
    seen = load_seen(args.outbox / "results.csv")
    print(f"Watching {args.inbox}  ->  results in {args.outbox}   (Ctrl+C to stop)")

    try:
        while True:
            sweep(models, args.inbox, args.outbox, args.threshold, not args.no_reports, seen)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
