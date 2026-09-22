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
# Styling — a jeweller's showroom: pearl-white and warm gold, a serif display
# face for headings paired with a clean geometric sans, set a size up from
# Streamlit's defaults so it reads well from across a room.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Jost:wght@400;500;600&display=swap');

      :root {
        --pearl: #fdfaf4;
        --pearl-panel: #ffffff;
        --hairline: rgba(184, 137, 43, .35);
        --ink: #2b2317;
        --ink-soft: rgba(43, 35, 23, .68);
        --gold: #b8892b;
        --gold-deep: #8c6a1b;
        --gold-light: #e7c77a;
        --emerald: #2f7d5b;
        --ruby: #b23a48;
        --amber: #c1862f;
      }

      html { font-size: 18px; }

      html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background-color: var(--pearl);
        background-image:
          radial-gradient(ellipse 120% 70% at 50% -10%, #ffffff 0%, rgba(255,255,255,0) 62%),
          url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='76' height='76'%3E%3Cpath d='M38 8 L68 38 L38 68 L8 38 Z' fill='none' stroke='%23b8892b' stroke-width='1' opacity='0.09'/%3E%3C/svg%3E");
        background-repeat: no-repeat, repeat;
        background-size: cover, 76px 76px;
        color: var(--ink);
        font-family: 'Jost', sans-serif;
        font-size: 1.05rem;
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] {
        background: var(--pearl-panel);
        border-right: 1px solid var(--hairline);
      }
      [data-testid="stSidebar"] * { color: var(--ink) !important; }
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 1.5rem; }

      .block-container {padding-top: 2.5rem; max-width: 1250px;}

      h1, h2, h3, h4, h5 {
        font-family: 'Cormorant Garamond', serif;
        color: var(--ink);
        letter-spacing: .01em;
      }
      h2 {font-size: 1.9rem;} h3 {font-size: 1.55rem;}
      p, li, label, span { font-size: 1.05rem; }

      /* ---------- Hero ---------- */
      .hero-wrap {display: flex; align-items: center; gap: 1.75rem; margin-bottom: .25rem;}
      .hero-mark {
        background: var(--pearl-panel); border-radius: 18px; padding: .6rem .8rem;
        border: 1px solid var(--hairline);
        box-shadow: 0 8px 22px rgba(184,137,43,.18);
        flex-shrink: 0;
      }
      .hero-mark img {display:block; height: 64px; width: auto;}
      .hero-title {position: relative; display: inline-block;}
      .hero h1 {
        font-size: 3rem; font-weight: 600; margin: 0; line-height: 1.05;
        background: linear-gradient(100deg, var(--gold-deep) 0%, var(--gold-light) 25%, var(--gold) 50%, var(--gold-light) 75%, var(--gold-deep) 100%);
        background-size: 250% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: shimmer 7s ease-in-out infinite;
      }
      .hero p {color: var(--ink-soft); margin: .5rem 0 0; font-size: 1.2rem; font-weight: 400;}
      .sparkle {
        position: absolute; color: var(--gold-light); font-size: 1rem; line-height: 1;
        opacity: 0; animation: twinkle 3.6s ease-in-out infinite;
      }
      @keyframes shimmer {
        0%, 100% {background-position: 0% center;}
        50% {background-position: 100% center;}
      }
      @keyframes twinkle {
        0%, 100% {opacity: 0; transform: scale(.6) rotate(0deg);}
        50% {opacity: .9; transform: scale(1) rotate(15deg);}
      }
      @media (prefers-reduced-motion: reduce) {
        .hero h1, .sparkle { animation: none; }
      }

      /* ---------- Ornate divider: a gem set between two hairlines ---------- */
      .ornate-rule {display: flex; align-items: center; gap: .85rem; margin: 1.9rem 0 2.3rem;}
      .ornate-rule .line {flex: 1; height: 1px; background: linear-gradient(90deg, rgba(184,137,43,0), var(--gold) 50%, rgba(184,137,43,0));}
      .ornate-rule .gem {color: var(--gold); font-size: 1.05rem; flex-shrink: 0;}
      .ornate-rule.left .line:first-child {flex: 0 0 0;}

      .footer-tag {text-align: center; color: var(--ink-soft); font-size: .95rem; margin-top: .5rem; letter-spacing: .02em;}
      .footer-tag img {height: 22px; vertical-align: middle; margin-right: .5rem; opacity: .85;}

      /* ---------- Verdict ---------- */
      .verdict {
        padding: 1.5rem 1.75rem; border-left: 3px solid var(--c);
        background: var(--bg); margin-bottom: 1.1rem; border-radius: 4px;
        box-shadow: 0 4px 18px rgba(43,35,23,.06);
      }
      .verdict .label {font-family: 'Cormorant Garamond', serif; font-size: 2.15rem; font-weight: 600; line-height: 1.15; color: var(--ink);}
      .verdict .sub {opacity: .75; font-size: 1rem; margin-top: .35rem; letter-spacing: .01em;}
      .verdict.defect     {--c: var(--ruby);   --bg: rgba(178,58,72,.07);}
      .verdict.clean      {--c: var(--emerald); --bg: rgba(47,125,91,.07);}
      .verdict.borderline {--c: var(--amber);  --bg: rgba(193,134,47,.09);}

      /* ---------- Process steps (a true 3-step sequence) ---------- */
      .steps {display: flex; gap: 0; margin: 1.5rem 0 .5rem;}
      .step {flex: 1; padding: 0 1.75rem; position: relative;}
      .step:not(:first-child) {border-left: 1px solid var(--hairline);}
      .step .icon {color: var(--gold); display:block; margin-bottom: .6rem;}
      .step .num {font-family: 'Cormorant Garamond', serif; color: var(--gold); font-size: 1.8rem; font-weight: 600; display:block; margin-bottom: .25rem;}
      .step b {font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; font-weight: 600; color: var(--ink); display:block; margin-bottom:.4rem;}
      .step span.body {color: var(--ink-soft); font-size: 1.02rem; line-height: 1.55;}

      /* ---------- Misc content controls ---------- */
      [data-testid="stMarkdownContainer"] p, .stCaption, small { color: var(--ink); }
      .stCaption, [data-testid="stCaptionContainer"] { opacity: .7; font-size: .98rem; }

      [data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, var(--gold-deep), var(--gold-light)); }
      [data-testid="stProgress"] > div > div > div { background: rgba(184,137,43,.14); }

      [data-testid="stFileUploaderDropzone"] {
        background: var(--pearl-panel); border: 1px dashed var(--hairline); border-radius: 10px;
      }

      button[kind], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
        border-radius: 999px !important; border: 1px solid var(--gold) !important;
        color: var(--gold-deep) !important; background: rgba(184,137,43,.08) !important;
        font-size: 1.02rem !important;
      }
      [data-testid="stDownloadButton"] button:hover, [data-testid="stBaseButton-secondary"]:hover {
        background: rgba(184,137,43,.18) !important;
      }

      [data-testid="stExpander"] { border: 1px solid var(--hairline); border-radius: 10px; background: rgba(184,137,43,.03); }

      hr, [data-testid="stDivider"] { border-color: var(--hairline) !important; }
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
    '<div><div class="hero-title"><h1>Jewellery Casting QC</h1>'
    '<span class="sparkle" style="top:-10px; left:-14px; animation-delay:.2s;">✦</span>'
    '<span class="sparkle" style="top:6px; right:-18px; animation-delay:1.4s; font-size:.7rem;">✦</span>'
    '<span class="sparkle" style="bottom:-14px; left:38%; animation-delay:2.4s; font-size:.8rem;">✦</span>'
    "</div>"
    "<p>Upload a photo of a cast piece to check for defects, see the suspected defect type, "
    "and see why.</p></div></div>"
    '<div class="ornate-rule"><span class="line"></span><span class="gem">◆</span><span class="line"></span></div>',
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
          <div class="step">
            <span class="icon"><svg viewBox="0 0 48 48" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M24 6v22M16 16l8-8 8 8"/><path d="M8 30v8a2 2 0 0 0 2 2h28a2 2 0 0 0 2-2v-8"/></svg></span>
            <span class="num">I</span><b>Upload</b><span class="body">A photo of the piece, or the area you're checking.</span>
          </div>
          <div class="step">
            <span class="icon"><svg viewBox="0 0 48 48" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M24 5l15 6v11c0 10-6.5 17-15 21C15.5 39 9 32 9 22V11z"/><path d="M17 24l5 5 10-11"/></svg></span>
            <span class="num">II</span><b>Verdict</b><span class="body">Defect / no defect, plus the top suspected defect types.</span>
          </div>
          <div class="step">
            <span class="icon"><svg viewBox="0 0 48 48" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="21" cy="21" r="13"/><path d="M30.5 30.5 41 41"/></svg></span>
            <span class="num">III</span><b>Explain</b><span class="body">A heatmap and outline show which region drove the result.</span>
          </div>
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
st.markdown('<div class="ornate-rule"><span class="line"></span><span class="gem">◆</span><span class="line"></span></div>', unsafe_allow_html=True)
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

st.markdown('<div class="ornate-rule"><span class="line"></span><span class="gem">◆</span><span class="line"></span></div>', unsafe_allow_html=True)
_footer_mark = f'<img src="data:image/png;base64,{_logo}"/>' if _logo else ""
st.markdown(f'<p class="footer-tag">{_footer_mark}Jewellery Casting QC</p>', unsafe_allow_html=True)
