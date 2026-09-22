"""Jewellery Casting QC: upload a photo, get defect / no-defect, the suspected
defect type, and explanations (Grad-CAM + occlusion)."""
import base64
import hashlib
import os
from pathlib import Path

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "logo.png"

st.set_page_config(page_title="Jewellery Casting QC", page_icon="💎", layout="wide")

import engine  # noqa: E402  (imported after set_page_config so TF start-up noise stays out of the UI)

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")


@st.cache_data(show_spinner=False)
def _logo_b64() -> str | None:
    if not LOGO_PATH.exists():
        return None
    return base64.b64encode(LOGO_PATH.read_bytes()).decode()


# --------------------------------------------------------------------------- #
# Styling — a jeweller's showcase: deep indigo velvet, warm gold hairlines,
# a serif display face for headings paired with a clean geometric sans.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Jost:wght@400;500;600&display=swap');

      :root {
        --midnight: #14122b;
        --midnight-soft: #1c1a3d;
        --midnight-line: rgba(212, 175, 116, .22);
        --ivory: #f6f1e6;
        --gold: #cda45e;
        --gold-bright: #e9c98a;
        --emerald: #4a9b78;
        --ruby: #c1596c;
        --amber: #c99a4d;
      }

      html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: radial-gradient(ellipse 120% 80% at 50% -10%, #211d47 0%, var(--midnight) 55%);
        color: var(--ivory);
        font-family: 'Jost', sans-serif;
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] {
        background: var(--midnight-soft);
        border-right: 1px solid var(--midnight-line);
      }
      [data-testid="stSidebar"] * { color: var(--ivory) !important; }

      .block-container {padding-top: 2.5rem; max-width: 1250px;}

      h1, h2, h3, h4, h5, .hero h1 {
        font-family: 'Cormorant Garamond', serif;
        color: var(--ivory);
        letter-spacing: .01em;
      }

      /* ---------- Hero ---------- */
      .hero-wrap {display: flex; align-items: center; gap: 1.75rem; margin-bottom: .25rem;}
      .hero-mark {
        background: var(--ivory); border-radius: 18px; padding: .55rem .7rem;
        border: 1px solid var(--midnight-line);
        box-shadow: 0 6px 24px rgba(0,0,0,.35);
        flex-shrink: 0;
      }
      .hero-mark img {display:block; height: 58px; width: auto;}
      .hero h1 {font-size: 2.5rem; font-weight: 600; margin: 0; line-height: 1.05;}
      .hero p {color: var(--gold-bright); opacity: .9; margin: .35rem 0 0; font-size: 1.05rem; font-weight: 400;}
      .gold-rule {
        height: 1px; margin: 1.5rem 0 2rem;
        background: linear-gradient(90deg, var(--gold) 0%, rgba(205,164,94,0) 70%);
      }

      /* ---------- Verdict ---------- */
      .verdict {
        padding: 1.4rem 1.6rem; border-left: 2px solid var(--c);
        background: var(--bg); margin-bottom: 1rem; border-radius: 2px;
      }
      .verdict .label {font-family: 'Cormorant Garamond', serif; font-size: 1.9rem; font-weight: 600; line-height: 1.15; color: var(--ivory);}
      .verdict .sub {opacity: .75; font-size: .92rem; margin-top: .3rem; letter-spacing: .01em;}
      .verdict.defect     {--c: var(--ruby);   --bg: rgba(193,89,108,.10);}
      .verdict.clean      {--c: var(--emerald); --bg: rgba(74,155,120,.10);}
      .verdict.borderline {--c: var(--amber);  --bg: rgba(201,154,77,.12);}

      /* ---------- Process steps (a true 3-step sequence) ---------- */
      .steps {display: flex; gap: 0; margin: 1.5rem 0 .5rem;}
      .step {flex: 1; padding: 0 1.5rem; position: relative;}
      .step:not(:first-child) {border-left: 1px solid var(--midnight-line);}
      .step .num {font-family: 'Cormorant Garamond', serif; color: var(--gold); font-size: 1.6rem; font-weight: 600; display:block; margin-bottom: .25rem;}
      .step b {font-family: 'Cormorant Garamond', serif; font-size: 1.15rem; font-weight: 600; color: var(--ivory); display:block; margin-bottom:.35rem;}
      .step span.body {opacity: .78; font-size: .92rem; line-height: 1.5;}

      /* ---------- Misc content controls ---------- */
      [data-testid="stMarkdownContainer"] p, .stCaption, small { color: var(--ivory); }
      .stCaption, [data-testid="stCaptionContainer"] { opacity: .7; }

      [data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, var(--gold), var(--gold-bright)); }
      [data-testid="stProgress"] > div > div > div { background: rgba(246,241,230,.12); }

      [data-testid="stFileUploaderDropzone"] {
        background: var(--midnight-soft); border: 1px dashed var(--midnight-line); border-radius: 10px;
      }

      button[kind], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
        border-radius: 999px !important; border: 1px solid var(--gold) !important;
        color: var(--ivory) !important; background: rgba(205,164,94,.12) !important;
      }
      [data-testid="stDownloadButton"] button:hover, [data-testid="stBaseButton-secondary"]:hover {
        background: rgba(205,164,94,.24) !important;
      }

      [data-testid="stExpander"] { border: 1px solid var(--midnight-line); border-radius: 10px; background: rgba(246,241,230,.03); }

      hr, [data-testid="stDivider"] { border-color: var(--midnight-line) !important; }
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


@st.cache_resource(show_spinner="Preparing the inspection models…")
def get_models(base_url: str | None):
    return engine.load_models(base_url)


@st.cache_data(show_spinner="Examining the piece…", max_entries=16)
def analyse(_models, data: bytes) -> dict:
    x = engine.to_model_input(engine.load_image(data))
    return engine.analyse(_models, x)


@st.cache_data(show_spinner="Rendering the heatmap…", max_entries=32)
def gradcam(_models, data: bytes, kind: str, idx: int) -> np.ndarray:
    x = engine.to_model_input(engine.load_image(data))
    clf = _models.multi if kind == "multi" else _models.binary
    return clf.gradcam(x, idx)


@st.cache_data(show_spinner="Cross-checking with occlusion sensitivity…", max_entries=16)
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
             "Lower it to catch more; raise it to be stricter.",
    )
    st.divider()
    st.subheader("About this inspection")
    st.markdown(
        """
        - **Defect vs. no-defect** — a first, fast pass over the piece
        - **Defect type** — narrows the finding to a likely category
        - **Explain** — a heatmap shows the region behind the call
        """
    )

# --------------------------------------------------------------------------- #
# Header + upload
# --------------------------------------------------------------------------- #
_logo = _logo_b64()
_mark = f'<div class="hero-mark"><img src="data:image/png;base64,{_logo}"/></div>' if _logo else ""
st.markdown(
    f'<div class="hero-wrap">{_mark}'
    '<div><h1>Jewellery Casting QC</h1>'
    "<p>Upload a photo of a cast piece to check for defects, see the suspected defect type, "
    "and see why.</p></div></div>"
    '<div class="gold-rule"></div>',
    unsafe_allow_html=True,
)

upload = st.file_uploader(
    "Drop an image here", type=["png", "jpg", "jpeg", "webp", "bmp"],
    help="Max 20 MB. A close, well-lit shot of the surface works best.",
)

if upload is None:
    st.markdown(
        """
        <div class="steps">
          <div class="step"><span class="num">I</span><b>Upload</b><span class="body">A photo of the piece, or the area you're checking.</span></div>
          <div class="step"><span class="num">II</span><b>Verdict</b><span class="body">Defect / no defect, plus the top suspected defect types.</span></div>
          <div class="step"><span class="num">III</span><b>Explain</b><span class="body">A heatmap and outline show which region drove the result.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        st.markdown(f"**{classes[best]['name']}:** {classes[best]['description']}")

    if is_defect_call:
        type_block()
    else:
        st.write("No defect type is shown because the defect probability is below the threshold.")
        with st.expander("Show the type ranking anyway"):
            type_block()

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
    st.info("A detailed heatmap isn't available for this particular selection.")
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
    st.caption(f"{len(boxes)} region(s) outlined.")
else:
    st.caption("No region stands out at this sensitivity. Try lowering *Box sensitivity*.")

with st.expander("How to read these"):
    st.markdown(
        """
        - **Grad-CAM** — warm colours mark the regions that drove the selected answer. For
          *Defect vs. no-defect* it shows the evidence for "defect", so on a clean verdict
          expect little heat.
        - **Occlusion** — a patch is slid across the image, and regions where covering it
          changes the score the most are highlighted. It works differently from Grad-CAM, so
          when both agree on the same region, that's a strong, corroborated finding.
        """
    )

header = f"{label} | defect prob {p_def:.1%} | top type: {classes[top[0]]['name']} ({type_probs[top[0]]:.0%}) | explaining: {choice}"
st.download_button(
    "⬇️ Download explanation (PNG)",
    data=engine.make_report_png(header, panels),
    file_name=f"qc_explanation_{img_id}.png",
    mime="image/png",
)
