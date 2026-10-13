"""Model loading, inference and explainability (Grad-CAM + occlusion) for the
jewellery casting QC app. No Streamlit imports here so it can be tested on its own.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

# Big QC photos (e.g. 5939 x 3837 = 23 MP) are normal here. Allow them, and tolerate files whose
# last bytes were cut off (common with interrupted copies/uploads) instead of refusing them.
Image.MAX_IMAGE_PIXELS = 500_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = (224, 224)          # must match the notebook (IMG = (224, 224))
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
CACHE_DIR = APP_DIR / ".model_cache"
BIN_FILE = "bin_defect_classifier.keras"
MULTI_FILE = "best_defect_classifier.keras"


# --------------------------------------------------------------------------- #
# Model files
# --------------------------------------------------------------------------- #
def _writable_cache_dir() -> Path:
    for d in (CACHE_DIR, Path(tempfile.gettempdir()) / "jewellery_qc_models"):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".probe"
            probe.write_text("ok")
            probe.unlink()
            return d
        except OSError:
            continue
    raise OSError("No writable directory available for model download.")


def ensure_model_file(filename: str, base_url: str | None = None) -> Path:
    """Return a local path to `filename`.

    Looks in models/ (files committed to the repo), then in the download cache.
    If neither has it and `base_url` is given, downloads it (e.g. a GitHub Release).
    """
    # 1) files committed to the repo: next to app.py, or in models/
    for committed in (APP_DIR / filename, MODEL_DIR / filename):
        if committed.exists():
            return committed

    for d in (CACHE_DIR, Path(tempfile.gettempdir()) / "jewellery_qc_models"):
        if (d / filename).exists():
            return d / filename

    if not base_url:
        raise FileNotFoundError(filename)

    dest_dir = _writable_cache_dir()
    dest, tmp = dest_dir / filename, dest_dir / (filename + ".part")
    url = base_url.rstrip("/") + "/" + filename
    try:
        urllib.request.urlretrieve(url, tmp)   # follows GitHub's release redirects
        if not zipfile.is_zipfile(tmp):        # a .keras file is a zip archive
            raise ValueError(f"Downloaded file from {url} is not a valid .keras archive.")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def load_model_info() -> dict:
    with open(APP_DIR / "model_info.json", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Classifier wrapper: predictions + Grad-CAM + occlusion
# --------------------------------------------------------------------------- #
class Classifier:
    """Wraps a saved Keras model of the form  input -> [aug] -> preprocess ->
    backbone -> GlobalAveragePooling2D -> [Dropout] -> Dense.

    IMPORTANT: in Keras 3 the `preprocess_input` step is part of the graph but is
    NOT listed in `model.layers`. So the feature extractor is built from the graph
    (input -> GAP's input tensor) instead of walking layers, which keeps the
    preprocessing in place. `xai_ready` says whether the extractor reproduces the
    model's own output (checked once at load time).
    """

    def __init__(self, model: tf.keras.Model, name: str):
        self.model = model
        self.name = name
        self.xai_ready = False
        self.feat_model = None
        self.head = None
        # Compiled graph call: TF reuses buffers, so memory stays flat with batch size.
        # (Eager Keras keeps every intermediate activation, ~100 MB/image for ResNet50V2.)
        self._predict_fn = tf.function(
            lambda t: self.model(t, training=False),
            input_signature=[tf.TensorSpec([None, *IMG_SIZE, 3], tf.float32)],
        )

        gap = next((l for l in model.layers
                    if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)), None)
        head = model.layers[-1]
        if gap is not None and isinstance(head, tf.keras.layers.Dense):
            try:
                self.feat_model = tf.keras.Model(model.input, gap.input)
                self.head = head
                self._logit_fn = tf.function(
                    lambda t: self._logits_from_features(self.feat_model(t, training=False)),
                    input_signature=[tf.TensorSpec([None, *IMG_SIZE, 3], tf.float32)],
                )
                self.xai_ready = self._verify()
            except Exception:
                self.xai_ready = False

    # -- predictions --------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (N, 224, 224, 3) float32 in 0-255. Returns model outputs (N, C)."""
        return self._predict_fn(tf.convert_to_tensor(x, tf.float32)).numpy()

    def _logits_from_features(self, feats):
        pooled = tf.reduce_mean(feats, axis=(1, 2))
        kernel = tf.convert_to_tensor(self.head.kernel)
        bias = tf.convert_to_tensor(self.head.bias)
        return tf.matmul(pooled, kernel) + bias

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Pre-activation scores (N, C). Unlike probabilities these don't saturate at 0/1,
        so they still change when a confident prediction loses evidence."""
        return self._logit_fn(tf.convert_to_tensor(x, tf.float32)).numpy()

    def _verify(self) -> bool:
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 255, size=(1,) + IMG_SIZE + (3,)).astype("float32")
        feats = self.feat_model(x, training=False)
        rebuilt = self.head.activation(self._logits_from_features(feats)).numpy()
        return bool(np.allclose(rebuilt, self.predict(x), atol=1e-4))

    # -- Grad-CAM -----------------------------------------------------------
    def gradcam(self, x: np.ndarray, target: int) -> np.ndarray:
        """Grad-CAM for output unit `target`, computed on the pre-activation score
        (logit), which avoids softmax saturation. Returns a small (h, w) map in [0, 1].
        """
        if not self.xai_ready:
            raise RuntimeError(f"Grad-CAM unavailable for the {self.name} model.")
        feats = self.feat_model(x, training=False)
        with tf.GradientTape() as tape:
            tape.watch(feats)
            score = self._logits_from_features(feats)[:, target]
        grads = tape.gradient(score, feats)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(feats * weights[:, None, None, :], axis=-1))[0].numpy()
        peak = cam.max()
        return cam / peak if peak > 1e-8 else cam

    # -- Occlusion sensitivity ----------------------------------------------
    def occlusion(self, x: np.ndarray, target: int, patch: int = 48,
                  n_pos: int = 9, batch: int = 9) -> np.ndarray:
        """Slide a mean-colour patch over the image and record how much the score
        (logit) for `target` drops. Model-agnostic cross-check for Grad-CAM.
        Returns a (224, 224) map in [0, 1].
        """
        if not self.xai_ready:
            raise RuntimeError(f"Occlusion unavailable for the {self.name} model.")
        h, w = IMG_SIZE
        base = float(self.logits(x)[0, target])
        fill = x[0].mean(axis=(0, 1))
        ys = np.linspace(0, h - patch, n_pos).astype(int)
        xs = np.linspace(0, w - patch, n_pos).astype(int)
        spots = [(y, xx) for y in ys for xx in xs]

        acc = np.zeros((h, w), np.float32)
        cnt = np.zeros((h, w), np.float32)
        for i in range(0, len(spots), batch):
            chunk = spots[i:i + batch]
            stack = np.repeat(x, len(chunk), axis=0)
            for k, (y0, x0) in enumerate(chunk):
                stack[k, y0:y0 + patch, x0:x0 + patch, :] = fill
            drops = base - self.logits(stack)[:, target]
            for k, (y0, x0) in enumerate(chunk):
                acc[y0:y0 + patch, x0:x0 + patch] += drops[k]
                cnt[y0:y0 + patch, x0:x0 + patch] += 1
        heat = np.clip(acc / np.maximum(cnt, 1), 0, None)
        peak = heat.max()
        return heat / peak if peak > 1e-8 else heat


@dataclass
class Models:
    binary: Classifier
    multi: Classifier
    info: dict

    @property
    def classes(self) -> list[dict]:
        return self.info["classes"]


def load_models(base_url: str | None = None) -> Models:
    bin_path = ensure_model_file(BIN_FILE, base_url)
    multi_path = ensure_model_file(MULTI_FILE, base_url)
    binary = Classifier(tf.keras.models.load_model(bin_path, compile=False), "binary")
    multi = Classifier(tf.keras.models.load_model(multi_path, compile=False), "multi-class")
    info = load_model_info()

    n_out = multi.model.output_shape[-1]
    if len(info["classes"]) != n_out:
        raise ValueError(
            f"model_info.json lists {len(info['classes'])} classes but the multi-class "
            f"model has {n_out} outputs. The class order/count must match training."
        )
    if binary.model.output_shape[-1] != 1:
        raise ValueError("Binary model should have a single sigmoid output.")
    return Models(binary=binary, multi=multi, info=info)


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _to_rgb(img: Image.Image) -> Image.Image:
    """Any PIL mode -> 8-bit RGB. Transparent pixels become the white backdrop (a plain
    convert('RGB') would turn them black, which the background removal then treats as a piece)."""
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):          # 16-bit / float scans
        a = np.asarray(img).astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(a)
        return Image.fromarray(a.astype(np.uint8)).convert("RGB")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, BG_FILL + (255,))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def load_image(data: bytes) -> Image.Image:
    """Decode an upload to RGB. Tries Pillow first, then OpenCV; if both fail the error says why."""
    if not data:
        raise ValueError("The uploaded file is empty (0 bytes).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()                                   # decode now so problems surface here
        return _to_rgb(ImageOps.exif_transpose(img))
    except Exception as pil_err:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # BGR, 8-bit
        if arr is None:
            raise ValueError(f"Could not decode the file with Pillow ({pil_err!r}) or OpenCV.") from pil_err
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def to_model_input(img: Image.Image) -> np.ndarray:
    """Same resize as training (tf.image.resize, bilinear). Returns (1,224,224,3) float32 0-255.
    Preprocessing (scaling) lives inside the saved models, so no scaling here."""
    arr = np.asarray(img)
    if max(arr.shape[:2]) > 4 * IMG_SIZE[0]:       # huge photo: area-average down first (memory + less aliasing)
        s = 4 * IMG_SIZE[0] / max(arr.shape[:2])
        arr = cv2.resize(arr, (round(arr.shape[1] * s), round(arr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return tf.image.resize(arr.astype(np.float32), IMG_SIZE).numpy()[None]


def fit_for_display(img: Image.Image, max_side: int = 720) -> np.ndarray:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# Background removal: isolate the jewellery piece from a plain backdrop
# --------------------------------------------------------------------------- #
BG_FILL = (255, 255, 255)   # what the removed background is painted with (a plain white backdrop)


@dataclass
class Prepared:
    original: Image.Image      # the photo as uploaded
    image: Image.Image         # what the models see (isolated piece on a plain backdrop, or the original)
    mask: np.ndarray           # bool, same size as `image`; True = jewellery piece
    full_mask: np.ndarray      # bool, same size as `original`
    found: bool                # False -> the cut-out wasn't reliable, so `image` is just the original
    isolated: bool             # whether isolation was requested


def _foreground_mask(rgb: np.ndarray, work: int = 2048) -> tuple[np.ndarray, bool]:
    """Boolean mask (full resolution) of everything that differs from the backdrop colour.

    The backdrop colour is estimated from the photo's border, so it works for white, off-white
    or light-grey backdrops. Small interior highlights are filled; large openings (the inside
    of a ring) stay background. Returns (mask, reliable).
    """
    H, W = rgb.shape[:2]
    s = min(1.0, work / max(H, W))
    small = cv2.resize(rgb, (max(1, round(W * s)), max(1, round(H * s))),
                       interpolation=cv2.INTER_AREA) if s < 1 else rgb
    h, w = small.shape[:2]
    k_blur = max(3, int(round(0.01 * max(h, w))) | 1)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (k_blur, k_blur), 0).astype("float32") / 255.0, cv2.COLOR_RGB2LAB)

    b = max(2, round(0.02 * max(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    plain = bool((np.linalg.norm(border - bg, axis=1) < 12).mean() >= 0.6)   # is the backdrop plain?

    dist = np.linalg.norm(lab - bg, axis=2)                                   # colour distance to backdrop
    d8 = np.clip(dist * 2.5, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu / 2.5, 12.0, 40.0))
    mask = (dist > thr).astype(np.uint8)
    k_close = max(3, 2 * int(round(0.003 * max(h, w))) + 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)), iterations=2)

    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros((H, W), bool), False
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= max(0.005 * areas.max(), 0.0002 * h * w)]  # drop dust, keep small stones
    mask = np.isin(cc, keep)

    inv = (~mask).astype(np.uint8)                                            # fill small interior holes
    n2, cc2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    on_edge = set(np.unique(np.concatenate([cc2[0], cc2[-1], cc2[:, 0], cc2[:, -1]])).tolist())
    holes = [i for i in range(1, n2) if i not in on_edge and stats2[i, cv2.CC_STAT_AREA] < 0.004 * h * w]
    if holes:
        mask = mask | np.isin(cc2, holes)

    reliable = plain and 0.01 <= mask.mean() <= 0.97
    full = cv2.resize(mask.astype("float32"), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
    return full, reliable


def segment_piece(img: Image.Image, pad_frac: float = 0.06):
    """Cut the jewellery piece out of its backdrop.

    Returns (isolated_image, mask_on_isolated, mask_on_original, found). The isolated image is
    cropped tightly (plus a small margin), has the backdrop painted with BG_FILL, and is padded
    to a square so resizing to 224x224 doesn't stretch the piece.
    """
    rgb = np.asarray(img)
    H, W = rgb.shape[:2]
    full, ok = _foreground_mask(rgb)
    if not ok:
        ones = np.ones((H, W), bool)
        return img, ones, ones, False

    ys, xs = np.where(full)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(round(pad_frac * max(y1 - y0, x1 - x0)))
    y0, x0, y1, x1 = max(0, y0 - pad), max(0, x0 - pad), min(H, y1 + pad), min(W, x1 + pad)

    crop, m = rgb[y0:y1, x0:x1], full[y0:y1, x0:x1]
    crop = np.where(m[..., None], crop, np.array(BG_FILL, np.uint8))
    h, w = m.shape
    side = max(h, w)
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = BG_FILL
    cmask = np.zeros((side, side), bool)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    cmask[oy:oy + h, ox:ox + w] = m
    return Image.fromarray(canvas), cmask, full, True


def prepare_image(data: bytes, isolate: bool = True) -> Prepared:
    """Load an upload and (optionally) isolate the jewellery piece. Feed `.image` to the models."""
    orig = load_image(data)
    if not isolate:
        ones = np.ones((orig.height, orig.width), bool)
        return Prepared(orig, orig, ones, ones, True, False)
    img, mask, full, found = segment_piece(orig)
    return Prepared(orig, img, mask, full, found, True)


def fit_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Resize a boolean piece-mask to (H, W) so it lines up with a display image."""
    h, w = shape_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def outline_overlay(img_rgb: np.ndarray, mask: np.ndarray, color=(0, 200, 110)) -> np.ndarray:
    """Draw the detected piece outline (including openings) on the photo, to check the cut-out."""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    contours, _ = cv2.findContours(fit_mask(mask, (H, W)).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, max(2, round(0.004 * max(H, W))))
    return out


# --------------------------------------------------------------------------- #
# Detailed scan: look at the piece in overlapping 224-px windows instead of shrinking it
# --------------------------------------------------------------------------- #
@dataclass
class ScanResult:
    image: np.ndarray        # RGB uint8: the isolated piece, at the scan resolution
    mask: np.ndarray         # bool, same size: True = jewellery piece
    tile_xy: np.ndarray      # (n, 2) top-left corner (x, y) of every scored window
    tile_p: np.ndarray       # (n,)   defect probability of every window
    tile_types: np.ndarray   # (n, C) defect-type probabilities of every window
    heat: np.ndarray         # float32 map in [0, 1]: mean defect probability of the windows covering each pixel
    p_defect: float          # piece-level score: mean of the k most suspicious windows
    type_probs: np.ndarray   # piece-level defect-type ranking (from the most suspicious windows)
    k: int
    window: int
    scale: float             # scan resolution relative to the isolated crop (<= 1)


def scan_piece(models: Models, image: Image.Image, mask: np.ndarray | None = None,
               max_side: int | None = 1792, window: int = 224, stride: int | None = None,
               min_cover: float = 0.25, k: int = 3, batch: int = 16,
               max_tiles: int = 700) -> ScanResult:
    """Score the piece window by window.

    The piece is shrunk only if its longest side exceeds `max_side`. Pass max_side=0 (or None)
    for NATIVE resolution: no resampling at all, every window is real, untouched pixels.
    `stride` defaults to half a window (50% overlap); if that would need more than `max_tiles`
    windows (typical at native resolution) it falls back to non-overlapping windows so the scan
    stays quick on a CPU. Windows that
    contain less than `min_cover` jewellery are skipped. The piece-level score is the mean of
    the `k` most suspicious windows, so one real flaw isn't averaged away by clean metal.
    """
    arr = np.asarray(image.convert("RGB"))
    m = np.ones(arr.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    h, w = arr.shape[:2]
    scale = 1.0
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        h, w = nh, nw
    if h < window or w < window:                       # tiny piece: pad so one window fits
        H2, W2 = max(h, window), max(w, window)
        canvas = np.empty((H2, W2, 3), np.uint8)
        canvas[:] = BG_FILL
        canvas[:h, :w] = arr
        mm = np.zeros((H2, W2), bool)
        mm[:h, :w] = m
        arr, m, h, w = canvas, mm, H2, W2

    def starts(n: int, step: int) -> list[int]:
        s = list(range(0, n - window + 1, step))
        if s[-1] != n - window:
            s.append(n - window)
        return s

    if stride is None:
        stride = window // 2
        if len(starts(h, stride)) * len(starts(w, stride)) > max_tiles:
            stride = window

    cand = [(x, y) for y in starts(h, stride) for x in starts(w, stride)]
    cover = np.array([m[y:y + window, x:x + window].mean() for x, y in cand])
    keep = np.where(cover >= min_cover)[0]
    if len(keep) == 0:                                  # nothing reaches min_cover: use the best window
        keep = np.array([int(cover.argmax())])
    xy = np.array([cand[i] for i in keep], dtype=int)

    pb, pm = [], []
    for i in range(0, len(xy), batch):
        stack = np.stack([arr[y:y + window, x:x + window] for x, y in xy[i:i + batch]]).astype("float32")
        if window != IMG_SIZE[0]:
            stack = np.stack([cv2.resize(t, IMG_SIZE, interpolation=cv2.INTER_AREA) for t in stack])
        pb.append(models.binary.predict(stack)[:, 0])
        pm.append(models.multi.predict(stack))
    pb, pm = np.concatenate(pb).astype("float32"), np.concatenate(pm).astype("float32")

    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for (x, y), p in zip(xy, pb):
        acc[y:y + window, x:x + window] += p
        cnt[y:y + window, x:x + window] += 1
    heat = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype("float32") * m

    order = np.argsort(pb)[::-1]
    kk = min(k, len(order))
    top = order[:max(kk, int(np.ceil(0.1 * len(order))))]
    wts = pb[top] + 1e-6
    type_probs = ((pm[top] * wts[:, None]).sum(0) / wts.sum()).astype("float32")
    return ScanResult(arr, m, xy, pb, pm, heat, float(pb[order[:kk]].mean()), type_probs, kk, window, scale)


def scan_panels(sc: ScanResult, threshold: float, alpha: float = 0.45, display_side: int = 720):
    """Overview panels for a scan: the piece, the defect-probability map, and the flagged areas."""
    disp = fit_for_display(Image.fromarray(sc.image), display_side)
    m = fit_mask(sc.mask, disp.shape[:2])
    heat = heatmap_overlay(disp, sc.heat, alpha, mask=m, normalize=False)
    boxed, boxes = derive_boxes(disp, sc.heat, threshold, mask=m, normalize=False)
    return [("Scanned piece", disp), ("Defect-probability map", heat), ("Flagged areas", boxed)], boxes


def closeups(models: Models, sc: ScanResult, n: int = 6, alpha: float = 0.45) -> list[dict]:
    """The n most suspicious windows (non-overlapping-ish), at native scan detail, each with a
    Grad-CAM for its top defect-type guess."""
    order = np.argsort(sc.tile_p)[::-1]
    chosen: list[int] = []
    for i in order:
        x, y = sc.tile_xy[i]
        if all(max(abs(x - sc.tile_xy[j][0]), abs(y - sc.tile_xy[j][1])) >= 0.75 * sc.window for j in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    out = []
    for i in chosen:
        x, y = (int(v) for v in sc.tile_xy[i])
        win = sc.window
        crop = sc.image[y:y + win, x:x + win]
        tmask = sc.mask[y:y + win, x:x + win]
        idx = int(sc.tile_types[i].argmax())
        overlay = crop
        if models.multi.xai_ready:
            inp = crop.astype("float32")
            if win != IMG_SIZE[0]:
                inp = cv2.resize(inp, IMG_SIZE, interpolation=cv2.INTER_AREA)
            cam = models.multi.gradcam(inp[None], idx)
            overlay = heatmap_overlay(crop, cam, alpha, mask=tmask if tmask.any() else None)
        out.append({"crop": crop, "overlay": overlay, "xy": (x, y), "p_defect": float(sc.tile_p[i]),
                    "type_name": models.classes[idx]["name"], "type_prob": float(sc.tile_types[i][idx])})
    return out


def zoom_crop(sc: ScanResult, cx: float, cy: float, size: int = 900, alpha: float = 0.45,
              show_heat: bool = True) -> np.ndarray:
    """A `size`-px square of the scanned piece centred at fractional position (cx, cy) in [0, 1].
    Shown at 1:1 (no resampling), so this is the real surface, optionally with the defect-probability
    map blended in. Use it to judge whether a flagged spot is a genuine scratch or pore."""
    H, W = sc.image.shape[:2]
    size = int(min(size, H, W))
    x0 = int(np.clip(round(cx * W - size / 2), 0, W - size))
    y0 = int(np.clip(round(cy * H - size / 2), 0, H - size))
    crop = sc.image[y0:y0 + size, x0:x0 + size]
    if not show_heat:
        return crop
    m = sc.mask[y0:y0 + size, x0:x0 + size]
    return heatmap_overlay(crop, sc.heat[y0:y0 + size, x0:x0 + size], alpha,
                           mask=m if m.any() else None, normalize=False)


def analyse(models: Models, x: np.ndarray) -> dict:
    return {
        "p_defect": float(models.binary.predict(x)[0, 0]),
        "type_probs": models.multi.predict(x)[0].astype("float32"),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _upsample(cam: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return np.clip(cv2.resize(cam.astype("float32"), (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)


def _masked_cam(cam: np.ndarray, shape_hw: tuple[int, int], mask: np.ndarray | None,
                normalize: bool = True) -> np.ndarray:
    """Upsample a map and (optionally) keep only the jewellery piece. With normalize=True the
    strongest spot *on the piece* is re-scaled to 1; with False the values are kept as they are
    (use that for real probabilities)."""
    full = _upsample(cam, shape_hw)
    if mask is not None:
        full = full * mask
        if normalize:
            peak = full.max()
            if peak > 1e-8:
                full = full / peak
    return full


def heatmap_overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4,
                    mask: np.ndarray | None = None, normalize: bool = True) -> np.ndarray:
    full = _masked_cam(cam, img_rgb.shape[:2], mask, normalize)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * full), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)
    if mask is not None:
        out[~mask] = img_rgb[~mask]          # leave the background untouched
    return out


def derive_boxes(img_rgb: np.ndarray, cam: np.ndarray, thr: float = 0.5,
                 min_area_frac: float = 0.005, max_boxes: int = 5, mask: np.ndarray | None = None,
                 normalize: bool = True, outline: bool = True):
    """Outline the connected regions where the map is >= thr (a fraction of the peak when
    normalize=True, an absolute probability otherwise).
    outline=True traces the hot area itself (a crescent stays a crescent); outline=False draws
    plain rectangles. Returns (image with outlines drawn, list of (x, y, w, h))."""
    H, W = img_rgb.shape[:2]
    full = _masked_cam(cam, (H, W), mask, normalize)
    hot = (full >= thr).astype("uint8")
    contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = [(cv2.boundingRect(c), c) for c in contours]
    items = [it for it in items if it[0][2] * it[0][3] >= min_area_frac * H * W]
    items = sorted(items, key=lambda it: it[0][2] * it[0][3], reverse=True)[:max_boxes]
    out = img_rgb.copy()
    thick = max(2, round(0.004 * max(H, W)))
    for (x, y, w, h), c in items:
        if outline:
            cv2.drawContours(out, [c], -1, (255, 60, 60), thick)
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 60, 60), thick)
    return out, [b for b, _ in items]


def make_report_png(header: str, panels: list[tuple[str, np.ndarray]], height: int = 420) -> bytes:
    font = ImageFont.load_default(size=16)
    band, gap = 34, 8
    tiles = []
    for title, arr in panels:
        im = Image.fromarray(arr)
        im = im.resize((max(1, round(im.width * height / im.height)), height))
        tile = Image.new("RGB", (im.width, height + band), "white")
        ImageDraw.Draw(tile).text((8, 8), title, fill=(30, 30, 30), font=font)
        tile.paste(im, (0, band))
        tiles.append(tile)
    width = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    top = 44
    canvas = Image.new("RGB", (max(width, 600), height + band + top), "white")
    ImageDraw.Draw(canvas).text((8, 12), header, fill=(20, 20, 20), font=font)
    x = 0
    for t in tiles:
        canvas.paste(t, (x, top))
        x += t.width + gap
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
