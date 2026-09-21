"""Jewellery Casting QC: upload a photo, get defect / no-defect, the suspected
defect type, and explanations (Grad-CAM + occlusion)."""
import hashlib
import os

import numpy as np
import streamlit as st

st.set_page_config(page_title="Jewellery Casting QC", page_icon="💎", layout="wide")

import engine  # noqa: E402  (imported after set_page_config so TF start-up noise stays out of the UI)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      .hero h1 {margin-bottom: .1rem;}
      .hero p {opacity: .75; margin-top: 0;}
      .verdict {padding: 1rem 1.25rem; border-radius: 14px; border-left: 8px solid var(--c);
                background: var(--bg); margin-bottom: .75rem;}
      .verdict .label {font-size: 1.5rem; font-weight: 700; line-height: 1.2;}
      .verdict .sub {opacity: .8; font-size: .92rem; margin-top: .2rem;}
      .verdict.defect     {--c: #dc3545; --bg: rgba(220,53,69,.12);}
      .verdict.clean      {--c: #2e9e5b; --bg: rgba(46,158,91,.12);}
      .verdict.borderline {--c: #e0a100; --bg: rgba(224,161,0,.15);}
      .pill {display:inline-block; padding:.1rem .6rem; border-radius:999px; font-size:.78rem;
             background: rgba(224,161,0,.2); border:1px solid rgba(224,161,0,.6); margin-left:.4rem;}
      .step {padding: 1rem 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.25); height: 100%;}
      .step b {font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources / computations
# --------------------------------------------------------------------------- #
def _model_base_url() -> str | None:
    """Optional URL prefix to download the .keras files from (e.g. a GitHub Release)."""
    try:
        url = st.secrets.get("MODEL_BASE_URL")
    except Exception:  # no secrets file
        url = None
    return url or os.environ.get("MODEL_BASE_URL")


@st.cache_resource(show_spinner="Loading models (first start takes a minute)…")
def get_models(base_url: str | None):
    return engine.load_models(base_url)


@st.cache_data(show_spinner="Analysing image…", max_entries=16)
def analyse(_models, data: bytes) -> dict:
    x = engine.to_model_input(engine.load_image(data))
    return engine.analyse(_models, x)


@st.cache_data(show_spinner="Computing Grad-CAM…", max_entries=32)
def gradcam(_models, data: bytes, kind: str, idx: int) -> np.ndarray:
    x = engine.to_model_input(engine.load_image(data))
    clf = _models.multi if kind == "multi" else _models.binary
    return clf.gradcam(x, idx)


@st.cache_data(show_spinner="Running occlusion sensitivity (~10 s)…", max_entries=16)
def occlusion(_models, data: bytes, kind: str, idx: int) -> np.ndarray:
    x = engine.to_model_input(engine.load_image(data))
    clf = _models.multi if kind == "multi" else _models.binary
    return clf.occlusion(x, idx)


# --------------------------------------------------------------------------- #
# Load models (with friendly errors)
# --------------------------------------------------------------------------- #
try:
    models = get_models(_model_base_url())
except FileNotFoundError as e:
    st.title("💎 Jewellery Casting QC")
    st.error(f"Model file not found: `{e}`")
    st.markdown(
        f"""
        Put these two files in the repo's **`models/`** folder (or set the `MODEL_BASE_URL`
        secret to a URL that hosts them, such as a GitHub Release):

        - `{engine.BIN_FILE}`
        - `{engine.MULTI_FILE}`

        See the README for the Colab export step.
        """
    )
    st.stop()
except Exception as e:  # corrupt file, class-count mismatch, Keras version problem...
    st.title("💎 Jewellery Casting QC")
    st.error("The models could not be loaded.")
    st.exception(e)
    st.stop()

info = models.info
classes = models.classes
card = info.get("model_card", {})

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Defect threshold", 0.05, 0.95, 0.50, 0.05,
        help="Images with a defect probability at or above this are treated as defective. "
             "Lower it to catch more (more false alarms); raise it to be stricter.",
    )
    st.divider()
    st.subheader("About this model")
    multi_card = card.get("multi", {})
    bin_card = card.get("binary", {})
    st.markdown(
        f"""
        - **Defect vs. no-defect:** {bin_card.get("backbone", "?")}
        - **Defect type:** {multi_card.get("backbone", "?")}, {len(classes)} classes
        """
    )
    if multi_card.get("accuracy") is not None:
        st.caption(
            f"Type model, held-out test set: **{multi_card['accuracy']:.1%}** accuracy, "
            f"macro-F1 **{multi_card.get('macro_f1', float('nan')):.2f}** "
            f"(random guessing ≈ {1 / len(classes):.1%})."
        )
    with st.expander("Limitations, please read"):
        st.markdown(
            f"""
            - Trained on **synthetic** images ({card.get("data", "synthetic casting defects")}).
              Real photos can look quite different.
            - {bin_card.get("note", "")}
            - The defect-**type** model is only moderately accurate, so read the top-3 list as
              *suggestions*, not a diagnosis.
            - Heatmaps show **where the model looked**, not proof that a defect is there.
            - Use as decision support, not as the final QC decision.
            - Uploaded images are processed in memory and not saved by this app.
            """
        )

# --------------------------------------------------------------------------- #
# Header + upload
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="hero"><h1>💎 Jewellery Casting QC</h1>'
    "<p>Upload a photo of a cast piece to check for defects, see the suspected defect type, "
    "and see <i>why</i> the model thinks so.</p></div>",
    unsafe_allow_html=True,
)

upload = st.file_uploader(
    "Drop an image here", type=["png", "jpg", "jpeg", "webp", "bmp"],
    help="Max 20 MB. A close, well-lit shot of the surface works best.",
)

if upload is None:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><b>1 · Upload</b><br>A photo of the piece or the area you are worried about.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="step"><b>2 · Verdict</b><br>Defect / no defect, plus the top-3 suspected defect types.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="step"><b>3 · Explain</b><br>Grad-CAM heatmap and boxes show which regions drove the result.</div>', unsafe_allow_html=True)
    st.stop()

data = upload.getvalue()
try:
    pil = engine.load_image(data)
except Exception:
    st.error("That file could not be read as an image. Try a PNG or JPEG.")
    st.stop()

img_id = hashlib.sha1(data).hexdigest()[:12]
display_img = engine.fit_for_display(pil)
result = analyse(models, data)

p_def = result["p_defect"]
type_probs = result["type_probs"]
top = np.argsort(type_probs)[::-1][:3]

# borderline band around the threshold
band = 0.10
if p_def >= threshold + band:
    kind, label = "defect", "Defect suspected"
elif p_def >= threshold - band:
    kind = "borderline"
    label = "Borderline, leans defect" if p_def >= threshold else "Borderline, leans clean"
else:
    kind, label = "clean", "No defect detected"
is_defect_call = p_def >= threshold

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.image(display_img, caption=upload.name)

with right:
    st.markdown(
        f'<div class="verdict {kind}"><div class="label">{label}</div>'
        f'<div class="sub">Defect probability <b>{p_def:.1%}</b> · threshold {threshold:.0%}</div></div>',
        unsafe_allow_html=True,
    )

    def type_block():
        st.markdown("##### Suspected defect type")
        for rank, i in enumerate(top):
            pct = float(type_probs[i])
            st.progress(min(max(pct, 0.0), 1.0), text=f"{classes[i]['name']}, {pct:.1%}")
        best = int(top[0])
        if type_probs[best] < 0.35:
            st.markdown(
                '<span class="pill">low confidence</span> The model is not sure; '
                "treat the type as a guess.", unsafe_allow_html=True,
            )
        st.markdown(f"**{classes[best]['name']}:** {classes[best]['description']}")
        st.caption("Reference description of the typical defect, not a finding about this image.")

    if is_defect_call:
        type_block()
    else:
        st.write("No defect type is shown because the defect probability is below the threshold.")
        with st.expander("Show the type ranking anyway"):
            type_block()

    st.caption(
        "⚠️ Trained on synthetic data: confirm on the physical piece before acting. "
        "See *About this model* in the sidebar."
    )

# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Why did the model say that?")

options = {f"Defect type: {classes[i]['name']} ({type_probs[i]:.0%})": ("multi", int(i)) for i in top}
options["Defect vs. no-defect verdict"] = ("bin", 0)
labels = list(options)

c_sel, c_alpha, c_thr = st.columns([2.2, 1, 1])
choice = c_sel.selectbox(
    "Explain", labels, index=0 if is_defect_call else len(labels) - 1, key=f"explain_{img_id}",
    help="Pick which prediction to explain. Each defect type is explained separately.",
)
alpha = c_alpha.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
box_thr = c_thr.slider("Box sensitivity", 0.2, 0.9, 0.5, 0.05,
                       help="Boxes outline areas where the heatmap is above this fraction of its peak. "
                            "Lower = larger/more boxes.")
use_occlusion = st.checkbox("Also run occlusion check (slower, independent second opinion)")

target_kind, target_idx = options[choice]
clf = models.binary if target_kind == "bin" else models.multi

if not clf.xai_ready:
    st.warning("Explanations are unavailable: this model's structure isn't compatible with the "
               "Grad-CAM routine (backbone → GlobalAveragePooling2D → Dense expected).")
    st.stop()

cam = gradcam(models, data, "bin" if target_kind == "bin" else "multi", target_idx)
overlay = engine.heatmap_overlay(display_img, cam, alpha)
boxed, boxes = engine.derive_boxes(display_img, cam, box_thr)

panels = [("Original", display_img), ("Grad-CAM heatmap", overlay), ("Regions of interest", boxed)]
if use_occlusion:
    occ = occlusion(models, data, "bin" if target_kind == "bin" else "multi", target_idx)
    panels.append(("Occlusion sensitivity", engine.heatmap_overlay(display_img, occ, alpha)))

cols = st.columns(len(panels))
for col, (title, arr) in zip(cols, panels):
    col.image(arr, caption=title)

if boxes:
    st.caption(f"{len(boxes)} region(s) outlined. Grad-CAM is a coarse 7×7 map: it points at "
               "*areas*, not exact defect outlines.")
else:
    st.caption("No region stands out at this sensitivity. Try lowering *Box sensitivity*.")

with st.expander("How to read these"):
    st.markdown(
        """
        - **Grad-CAM** – warm colours mark regions that pushed the model towards the selected
          answer. For *Defect vs. no-defect* it shows the evidence for "defect", so on a clean
          verdict expect weak or scattered heat.
        - **Occlusion** – grey patches are slid over the image; regions where hiding the patch
          lowers the model's score most are highlighted. It uses a different mechanism from
          Grad-CAM, so **agreement between the two** is a good sign, while disagreement means
          the model may be relying on something other than the defect (background, edges, lighting).
        - Heat that sits on the background rather than the piece is a warning sign.
        """
    )

header = f"{label} | defect prob {p_def:.1%} | top type: {classes[top[0]]['name']} ({type_probs[top[0]]:.0%}) | explaining: {choice}"
st.download_button(
    "⬇️ Download explanation (PNG)",
    data=engine.make_report_png(header, panels),
    file_name=f"qc_explanation_{img_id}.png",
    mime="image/png",
)
