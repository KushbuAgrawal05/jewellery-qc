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
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def to_model_input(img: Image.Image) -> np.ndarray:
    """Same resize as training (tf.image.resize, bilinear). Returns (1,224,224,3) float32 0-255.
    Preprocessing (scaling) lives inside the saved models, so no scaling here."""
    arr = np.asarray(img, dtype=np.float32)
    return tf.image.resize(arr, IMG_SIZE).numpy()[None]


def fit_for_display(img: Image.Image, max_side: int = 720) -> np.ndarray:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(img)


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


def heatmap_overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    full = _upsample(cam, img_rgb.shape[:2])
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * full), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)


def derive_boxes(img_rgb: np.ndarray, cam: np.ndarray, thr: float = 0.5,
                 min_area_frac: float = 0.005, max_boxes: int = 5):
    """Boxes around connected regions where the map is >= thr * peak.
    Returns (image with boxes drawn, list of (x, y, w, h))."""
    H, W = img_rgb.shape[:2]
    full = _upsample(cam, (H, W))
    mask = (full >= thr).astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] >= min_area_frac * H * W]
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_boxes]
    out = img_rgb.copy()
    thick = max(2, round(0.004 * max(H, W)))
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 60, 60), thick)
    return out, boxes


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
