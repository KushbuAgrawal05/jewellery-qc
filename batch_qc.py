"""Folder pipeline for Jewellery Casting QC.

Watches an inbox folder (default: ~/Desktop/qc_inbox), runs every new image through the
same models as the Streamlit app (via engine.py), and writes:

    <outbox>/results.csv        one row per image (verdict, probability, top-3 defect types)
    <outbox>/reports/*.png      a FULL report per image: verdict, defect %, top-3 types with
                                descriptions, then either the detailed-scan map + close-ups (default)
                                or Grad-CAM / regions / occlusion views (--mode whole)
    <outbox>/processed/         originals after a successful run
    <outbox>/failed/            files that could not be read / scored
    <outbox>/duplicates/        images identical to one already tested (skipped)

Put this file next to engine.py and model_info.json, then:

    python batch_qc.py                      # keep watching the inbox
    python batch_qc.py --once               # process what's there now and exit
    python batch_qc.py --no-occlusion       # faster: skip the occlusion check
    python batch_qc.py --no-isolate         # don't remove the background (analyse the whole photo)
    python batch_qc.py --mode whole         # old behaviour: shrink the piece to 224 px (fast, loses small defects)
    python batch_qc.py --scan-side 2688     # finer detailed scan (slower); default 1792
    python batch_qc.py --inbox "/path/to/photos" --threshold 0.4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import engine  # same module the Streamlit app uses

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BORDERLINE_BAND = 0.10  # same "borderline" zone as the app
HEAT_ALPHA = 0.45       # heatmap opacity (app default)
BOX_THR = 0.50          # box sensitivity (app default)

RUBY, EMERALD, AMBER = (178, 58, 72), (47, 125, 91), (193, 134, 47)
GOLD, INK, SOFT, TRACK = (184, 137, 43), (43, 35, 23), (110, 102, 92), (236, 230, 218)
PANEL_BG = (250, 247, 240)


# --------------------------------------------------------------------------- #
# Full report image
# --------------------------------------------------------------------------- #
def _wrap(d: ImageDraw.ImageDraw, text: str, font, width: int, max_lines: int = 3) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if d.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;") + "…"
    return lines


def _tile(arr: np.ndarray, size: int) -> Image.Image:
    im = Image.fromarray(arr).convert("RGB")
    f = size / max(im.size)                                  # fit the frame: shrink big images, enlarge small ones
    im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))),
                   Image.LANCZOS if f < 1 else Image.BICUBIC)
    tile = Image.new("RGB", (size, size), PANEL_BG)
    tile.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return tile


def make_full_report(original: np.ndarray, verdict: str, color: tuple, p_def: float,
                     threshold: float, top: list[tuple[str, float, str]],
                     rows: list[tuple[str, list[tuple[str, np.ndarray]]]],
                     filename: str, model_note: str = "", warning: str = "",
                     howto: str | None = None) -> bytes:
    """top: [(name, prob, description)] ; rows: [(row_title, [(panel_title, image), ...])]."""
    M, P, G = 40, 400, 20                      # margin, panel size, gap
    W = 2 * M + 3 * P + 2 * G
    title_h, head_h, row_h, foot_h = 64, 490, 60 + P + 34, 215
    H = title_h + 10 + head_h + len(rows) * row_h + foot_h

    f_title, f_big, f_mid, f_small, f_tiny = (ImageFont.load_default(size=s) for s in (26, 34, 21, 17, 15))
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)

    # title band
    d.text((M, 18), "Jewellery Casting QC  |  Inspection report", fill=INK, font=f_title)
    stamp = f"{filename}   |   {datetime.now():%Y-%m-%d %H:%M}"
    d.text((W - M - d.textlength(stamp, font=f_small), 24), stamp, fill=SOFT, font=f_small)
    d.line([(M, title_h), (W - M, title_h)], fill=GOLD, width=2)

    # header: original photo (left) + verdict and top-3 (right)
    y0 = title_h + 24
    canvas.paste(_tile(original, P), (M, y0))
    d.text((M, y0 + P + 8), "Original", fill=SOFT, font=f_small)

    x0, rw = M + P + 40, W - M - (M + P + 40)
    d.rectangle([x0, y0, x0 + 6, y0 + 70], fill=color)
    d.text((x0 + 20, y0 - 4), verdict, fill=INK, font=f_big)
    d.text((x0 + 20, y0 + 42), f"Defect probability {p_def:.1%}   |   threshold {threshold:.0%}",
           fill=SOFT, font=f_mid)

    if warning:
        d.text((x0 + 20, y0 + 72), warning, fill=AMBER, font=f_small)
    d.text((x0, y0 + 100), "Top 3 suspected defect types", fill=INK, font=f_mid)
    if p_def < threshold:
        d.text((x0 + 330, y0 + 104), "(below threshold: for reference only)", fill=SOFT, font=f_small)
    y = y0 + 142
    for rank, (name, prob, desc) in enumerate(top, 1):
        d.text((x0, y), f"{rank}. {name}", fill=INK, font=f_mid)
        pct = f"{prob:.1%}"
        d.text((x0 + rw - d.textlength(pct, font=f_mid), y), pct, fill=INK, font=f_mid)
        d.rounded_rectangle([x0, y + 31, x0 + rw, y + 43], radius=6, fill=TRACK)
        d.rounded_rectangle([x0, y + 31, x0 + max(10, int(rw * prob)), y + 43], radius=6, fill=GOLD)
        for i, line in enumerate(_wrap(d, desc, f_tiny, rw, max_lines=2)):
            d.text((x0, y + 52 + i * 19), line, fill=SOFT, font=f_tiny)
        y += 110

    # explanation rows
    y = title_h + 10 + head_h
    d.line([(M, y - 8), (W - M, y - 8)], fill=TRACK, width=2)
    for row_title, panels in rows:
        d.text((M, y + 4), row_title, fill=INK, font=f_mid)
        if not panels:
            d.text((M, y + 60), "A detailed heatmap isn't available for this selection.",
                   fill=SOFT, font=f_small)
        for c, (ptitle, arr) in enumerate(panels):
            px = M + c * (P + G)
            d.text((px, y + 36), ptitle, fill=SOFT, font=f_small)
            canvas.paste(_tile(arr, P), (px, y + 60))
        y += row_h

    # footer
    d.line([(M, y), (W - M, y)], fill=TRACK, width=2)
    notes = [howto] if howto else [
        "How to read these:  Grad-CAM: warm colours mark the regions that drove the selected answer. "
        "For the defect vs. no-defect row it shows evidence for 'defect', so a clean verdict shows little heat.",
        "Regions of interest: outlines around areas where the heatmap is above half of its peak.  "
        "Occlusion: a patch is slid across the image; regions where covering it changes the score most are "
        "highlighted. It works differently from Grad-CAM, so when both agree the finding is better supported.",
    ]
    if model_note:
        notes.append(model_note)
    ty = y + 14
    for note in notes:
        for line in _wrap(d, note, f_tiny, W - 2 * M, max_lines=3):
            d.text((M, ty), line, fill=SOFT, font=f_tiny)
            ty += 19
        ty += 6

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
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


def explain_row(models, clf, kind: str, idx: int, x, display, title: str, occlusion: bool, mask=None):
    """One report row: Grad-CAM, regions of interest and (optionally) occlusion."""
    if not clf.xai_ready:
        return title, []
    cam = clf.gradcam(x, idx)
    boxed, boxes = engine.derive_boxes(display, cam, BOX_THR, mask=mask)
    panels = [("Grad-CAM heatmap", engine.heatmap_overlay(display, cam, HEAT_ALPHA, mask=mask)),
              (f"Regions of interest ({len(boxes)} outlined)", boxed)]
    if occlusion:
        occ = clf.occlusion(x, idx)
        panels.append(("Occlusion sensitivity", engine.heatmap_overlay(display, occ, HEAT_ALPHA, mask=mask)))
    return title, panels


def process_one(models, path: Path, outbox: Path, threshold: float, reports: bool,
                occlusion: bool, isolate: bool, mode: str, scan_side: int, sha1: str) -> dict:
    data = path.read_bytes()
    prep = engine.prepare_image(data, isolate)       # removes the backdrop when isolate=True
    use_mask = isolate and prep.found
    detailed = mode == "detailed"
    if detailed:                                     # slide 224-px windows over the full-resolution piece
        sc = engine.scan_piece(models, prep.image, prep.mask if use_mask else None, max_side=scan_side)
        p_def, probs = sc.p_defect, sc.type_probs
    else:                                            # shrink the whole piece to 224 x 224
        x = engine.to_model_input(prep.image)
        res = engine.analyse(models, x)
        p_def, probs = res["p_defect"], res["type_probs"]
    order = [int(i) for i in np.argsort(probs)[::-1][:3]]
    top = [(models.classes[i]["name"], float(probs[i]), models.classes[i]["description"]) for i in order]
    is_defect = p_def >= threshold

    if p_def >= threshold + BORDERLINE_BAND:
        verdict, color = "Defect suspected", RUBY
    elif p_def >= threshold - BORDERLINE_BAND:
        verdict = "Borderline, leans defect" if is_defect else "Borderline, leans clean"
        color = AMBER
    else:
        verdict, color = "No defect detected", EMERALD

    if reports:
        orig_disp = engine.fit_for_display(prep.original)
        display = engine.fit_for_display(prep.image)
        rows, howto = [], None
        if use_mask:
            rows.append(("Step 1: background removed (the models only see the jewellery piece)",
                         [("Original photo", orig_disp),
                          ("Detected outline", engine.outline_overlay(orig_disp, prep.full_mask)),
                          ("Isolated piece", display)]))
        if detailed:
            panels, _ = engine.scan_panels(sc, threshold, HEAT_ALPHA)
            rows.append((f"Detailed scan: {len(sc.tile_p)} windows checked at high detail", panels))
            ups = engine.closeups(models, sc, n=6)
            for s in range(0, len(ups), 3):
                title = ("Most suspicious close-ups (scan detail, with the model's attention for its top guess)"
                         if s == 0 else "More close-ups")
                rows.append((title, [(f"#{s + j + 1}  defect {u['p_defect']:.0%}  |  {u['type_name']} {u['type_prob']:.0%}",
                                      u["overlay"]) for j, u in enumerate(ups[s:s + 3])]))
            howto = ("How to read these:  Defect-probability map: the models' defect probability for each area, "
                     "averaged over the overlapping windows that cover it (warmer = more suspicious). Flagged areas: "
                     "outlines at or above the threshold. Close-ups: the most suspicious windows at scan detail, with "
                     "Grad-CAM inside each. Judge the real surface there: is there a pore or a scratch? "
                     "The piece score is the mean of the 3 most suspicious windows.")
        else:
            mask = engine.fit_mask(prep.mask, display.shape[:2]) if use_mask else None
            rows.append(explain_row(models, models.binary, "bin", 0, x, display,
                                    "Why: defect vs. no-defect verdict", occlusion, mask))
            for rank, i in enumerate(order, 1):
                rows.append(explain_row(models, models.multi, "multi", i, x, display,
                                        f"Why: defect type #{rank}, {top[rank - 1][0]} ({top[rank - 1][1]:.0%})",
                                        occlusion, mask))
        m = models.info.get("model_card", {}).get("multi", {})
        note = ""
        if "accuracy" in m:
            note = (f"Model note: the defect-type model scored {m['accuracy']:.0%} accuracy "
                    f"(macro-F1 {m.get('macro_f1', 0):.0%}) on {m.get('test_images', '?')} test images "
                    "of synthetic data, so treat the type ranking as indicative only.")
        if use_mask:
            note = ("Background removed: heatmaps and outlines are limited to the jewellery piece. " + note).strip()
        warning = ("Couldn't separate the piece from its backdrop; whole photo analysed."
                   if isolate and not prep.found else "")
        png = make_full_report(orig_disp, verdict, color, p_def, threshold, top, rows, path.name, note, warning, howto)
        (outbox / "reports" / f"{path.stem}_{sha1[:6]}_{mode}_report.png").write_bytes(png)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": path.name,
        "sha1": sha1,
        "verdict": "DEFECT" if is_defect else "OK",
        "defect_probability": round(p_def, 4),
        "threshold": threshold,
        "type_1": top[0][0], "type_1_prob": round(top[0][1], 4),
        "type_2": top[1][0], "type_2_prob": round(top[1][1], 4),
        "type_3": top[2][0], "type_3_prob": round(top[2][1], 4),
        "mode": mode,
        "windows": len(sc.tile_p) if detailed else 1,
    }


def load_seen(csv_path: Path) -> set[str]:
    """Fingerprints of images already tested, read from results.csv (the 'register')."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {f"{r['sha1']}:{r.get('mode') or 'whole'}" for r in csv.DictReader(f) if r.get("sha1")}


def append_csv(csv_path: Path, row: dict) -> None:
    if csv_path.exists():                                   # older results.csv with fewer columns: widen it
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header, old = reader.fieldnames or [], list(reader)
        if header != list(row):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row), restval="")
                w.writeheader()
                w.writerows(old)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def sweep(models, inbox: Path, outbox: Path, threshold: float, reports: bool,
          occlusion: bool, isolate: bool, mode: str, scan_side: int, seen: set[str]) -> int:
    files = sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    done = 0
    for path in files:
        if not wait_until_stable(path):
            continue  # still being copied; pick it up next sweep
        try:
            sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
            key = f"{sha1}:{mode}"
            if key in seen:  # exact same image was already tested in this mode
                print(f"[SKIPPED] {path.name}: already in results.csv")
                shutil.move(str(path), unique_dest(outbox / "duplicates", path.name))
                continue
            row = process_one(models, path, outbox, threshold, reports, occlusion, isolate, mode, scan_side, sha1)
            append_csv(outbox / "results.csv", row)
            seen.add(key)
            shutil.move(str(path), unique_dest(outbox / "processed", path.name))
            print(f"[{row['verdict']:6}] {path.name}  -> report saved")
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
    ap.add_argument("--mode", choices=["detailed", "whole"], default="detailed",
                    help="detailed: scan the piece in overlapping windows (finds small defects). "
                         "whole: shrink the piece to 224 px (fast)")
    ap.add_argument("--scan-side", type=int, default=1792,
                    help="detailed mode: longest side (px) the piece is scanned at; 2688 = finer, slower")
    ap.add_argument("--no-occlusion", action="store_true", help="whole mode only: skip the (slower) occlusion check")
    ap.add_argument("--no-isolate", action="store_true",
                    help="analyse the whole photo instead of cutting the piece out of its backdrop")
    ap.add_argument("--no-reports", action="store_true", help="skip the report images (CSV only)")
    args = ap.parse_args()

    for d in (args.inbox, args.outbox / "reports", args.outbox / "processed",
              args.outbox / "failed", args.outbox / "duplicates"):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    # Looks in models/ first; otherwise downloads from MODEL_BASE_URL (if set) and caches the files.
    models = engine.load_models(os.environ.get("MODEL_BASE_URL"))
    seen = load_seen(args.outbox / "results.csv")
    print(f"Watching {args.inbox}  ->  results in {args.outbox}   (Ctrl+C to stop)")

    try:
        while True:
            sweep(models, args.inbox, args.outbox, args.threshold,
                  not args.no_reports, not args.no_occlusion, not args.no_isolate, args.mode, args.scan_side, seen)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
