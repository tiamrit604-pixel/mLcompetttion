"""
app.py
======
Main Streamlit entry point -- the landing page.
Run with:   streamlit run app.py
"""
import os
import sys
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.theme import (
    inject_theme, side_nav, app_bar, hero, kpi, section_card, step_mini,
    thin_divider,
)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Flange Detection · UH",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help'    : None,
        'Report a bug': None,
        'About'       : (
            "Bolted-Flange Looseness Detection — University of Houston, "
            "Spring 2026 ML Final Project. Author: Amrit Tiwari."
        ),
    },
)
inject_theme()
side_nav(active="home")

# -----------------------------------------------------------------------------
# Top app bar
# -----------------------------------------------------------------------------
app_bar(crumb="Overview", status="System ready")

# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
hero_l, hero_r = st.columns([5, 1.4], gap="medium")

with hero_l:
    hero(
        eyebrow="Spring 2026  •  University of Houston",
        title_html=(
            "Classifying <span class='accent'>Pipeline Flange "
            "Torque</span><br>from a Single Hammer Strike"
        ),
        subtitle=(
            "A non-destructive ML pipeline for bolted-flange looseness "
            "detection from percussion-induced acoustic signals."
        ),
        meta=(
            "<b>Amrit Tiwari</b> &nbsp;·&nbsp; Graduate Student, "
            "Mechanical Engineering<br>"
            "Cullen College of Engineering &nbsp;·&nbsp; "
            "University of Houston"
        ),
        chips=[
            ("Production-ready pipeline", "live"),
            ("9 ML models · 1 ensemble", None),
            ("Validated on 4 unseen flanges", None),
        ],
    )

with hero_r:
    logo_path = os.path.join(HERE, 'assets', 'uh_logo.png')
    if os.path.exists(logo_path):
        st.markdown("<div style='padding-top:1.6em'></div>",
                    unsafe_allow_html=True)
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown(
            "<div style='margin-top:1.6em;text-align:center;font-size:0.78em;"
            "color:var(--subtle);padding:1.6em 0.8em;border:1.5px dashed "
            "var(--border-strong);border-radius:14px;line-height:1.5;"
            "background:var(--surface-2);'>"
            "Drop <code>uh_logo.png</code><br>into <code>assets/</code><br>"
            "for the UH shield"
            "</div>",
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4, gap="small")
with m1:
    kpi(value="3", label="Torque classes<br>0 / 25 / 50 ft-lb",
        icon="⚙", trend="3-way", trend_kind="info",
        sparkline=[0.3, 0.5, 0.7, 0.6, 0.8, 0.9, 1.0, 1.0])
with m2:
    kpi(value="9", label="ML / DL models<br>compared head-to-head",
        icon="◆", trend="Ensemble", trend_kind="info",
        sparkline=[0.4, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 1.0])
with m3:
    kpi(value="88", unit="%",
        label="3-class accuracy<br>on an unseen flange",
        icon="◎", trend="LOFO", trend_kind="success",
        sparkline=[0.5, 0.6, 0.65, 0.72, 0.78, 0.82, 0.86, 0.88])
with m4:
    kpi(value="99.6", unit="%",
        label="Binary loose-vs-tight<br>field-relevant",
        icon="✓", trend="+11.5pp", trend_kind="success",
        sparkline=[0.6, 0.7, 0.8, 0.85, 0.92, 0.96, 0.98, 1.0])

thin_divider()

# -----------------------------------------------------------------------------
# Why this matters
# -----------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("<h2 style='margin-top:0.2em'>Why this matters</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "Bolted flange joints in oil-and-gas, water and process pipelines "
        "rely on bolt preload to maintain a leak-tight seal. A single loose "
        "bolt can initiate fluid leakage, environmental damage and "
        "catastrophic line failure."
    )
    st.markdown(
        "This application asks a simple question: "
        "**can a single hammer strike, recorded by a phone, classify a "
        "flange as loose, partially tight, or fully tight?** "
        "The full pipeline — feature extraction, within-flange-area "
        "normalisation, PCA, training, leave-one-flange-out validation "
        "and accuracy-weighted ensemble prediction — is delivered through "
        "this web application."
    )

with right:
    flange_image = os.path.join(HERE, 'assets', 'flange_setup.png')
    if os.path.exists(flange_image):
        st.image(flange_image, caption="Bolted-flange test specimen",
                 use_container_width=True)
    else:
        st.markdown(
            "<div style='padding:1.2em 1.4em;border-radius:14px;"
            "background:var(--surface-2);border:1px solid var(--border);"
            "font-size:0.9em;color:var(--muted);line-height:1.6;"
            "box-shadow:var(--shadow-xs);'>"
            "<div style='display:flex;align-items:center;gap:0.5em;"
            "margin-bottom:0.6em;'>"
            "<span style='display:inline-flex;align-items:center;"
            "justify-content:center;width:28px;height:28px;border-radius:8px;"
            "background:var(--brand-50);color:var(--brand);'>📷</span>"
            "<b style='color:var(--ink);font-size:1.0em;'>Optional images</b>"
            "</div>"
            "Drop these into the <code>assets/</code> folder to enrich this "
            "landing page:<br><br>"
            "&nbsp;•&nbsp; <code>uh_logo.png</code> &mdash; UH shield<br>"
            "&nbsp;•&nbsp; <code>flange_setup.png</code> &mdash; test rig<br>"
            "&nbsp;•&nbsp; <code>hammer_recording.png</code> &mdash; data acquisition"
            "</div>",
            unsafe_allow_html=True,
        )

thin_divider()

# -----------------------------------------------------------------------------
# Three sections
# -----------------------------------------------------------------------------
st.markdown("<h2>Get started</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:var(--muted);margin-top:-0.4em;margin-bottom:1.4em;"
    "font-size:1.02em;'>"
    "Three self-contained workflows. Open whichever fits your current goal."
    "</p>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    section_card(
        num="①",
        title="Train Your Own Model",
        body_html=(
            "<p>Upload labelled flange recordings and run the full pipeline: "
            "feature extraction, WFAN normalisation, PCA, dependent "
            "test (70/30) and the leave-one-flange-out independent test.</p>"
            "<p>Inspect every confusion matrix, every per-flange accuracy, "
            "then download the trained models as a single bundle.</p>"
        ),
        tags=[("Heavy · CPU-bound", "warn"), ("End-to-end", "info")],
    )
    if st.button("Open Section 1  →", key="go1",
                 use_container_width=True):
        st.switch_page("pages/1_Train_Your_Own_Model.py")

with c2:
    section_card(
        num="②",
        title="Competition Prediction",
        body_html=(
            "<p>Use the pre-trained leave-one-flange-out model bundles to "
            "predict torque level from a single recording. Choose the "
            "flange and area, upload an audio file or record live in the "
            "browser.</p>"
            "<p>See per-model votes plus the accuracy-weighted ensemble "
            "decision &mdash; the same pipeline that produced the F1&ndash;F4 "
            "predictions on the poster.</p>"
        ),
        tags=[("Demo-ready", "success"), ("&lt; 2 sec inference", "info")],
    )
    if st.button("Open Section 2  →", key="go2",
                 type="primary", use_container_width=True):
        st.switch_page("pages/2_Competition_Prediction.py")

with c3:
    section_card(
        num="③",
        title="Signal Processing & Analysis",
        body_html=(
            "<p>Upload labelled recordings of any torque level and see "
            "time-series, PSD, MFCC, delta-MFCC, log-Mel, STFT and Hilbert "
            "envelope side-by-side across the three classes.</p>"
            "<p>Useful for understanding <i>why</i> the classifier works: "
            "the spectral and ring-down differences between 0, 25 and 50 "
            "ft-lb are visible to the eye.</p>"
        ),
        tags=[("Educational", ""), ("Interactive", "info")],
    )
    if st.button("Open Section 3  →", key="go3",
                 use_container_width=True):
        st.switch_page("pages/3_Signal_Analysis.py")

thin_divider()

# -----------------------------------------------------------------------------
# Pipeline at a glance
# -----------------------------------------------------------------------------
st.markdown("<h2>How it works</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:var(--muted);margin-top:-0.4em;margin-bottom:1.4em;"
    "font-size:1.02em;'>The eight stages of the pipeline, end to end."
    "</p>",
    unsafe_allow_html=True,
)

pipeline_steps = [
    ("01", "Audio loading",
     "<code>audioread</code> decodes iPhone <code>.m4a</code> at native "
     "48&nbsp;kHz, avoiding silent resampling bugs in "
     "<code>librosa.load</code>."),
    ("02", "Peak detection &amp; segmentation",
     "Strikes are detected as peaks ≥ 30&nbsp;% of the file maximum, "
     "separated by ≥ 0.5&nbsp;s. Each strike is cut to a 0.42&nbsp;s "
     "window covering the full ring-down."),
    ("03", "Three parallel feature views",
     "(a) a 221-d flat vector (80 log-PSD bands + 128 MFCC stats + 13 "
     "physics features), (b) a 3-channel CNN image (MFCC | Δ-MFCC | "
     "log-Mel), (c) a 64×32 MFCC sequence for the BiLSTM."),
    ("04", "Within-Flange-Area Normalisation",
     "For each of the 16 (flange, area) combinations, the population mean "
     "is subtracted from every sample. Converts an inter-unit variability "
     "problem into a torque-classification problem."),
    ("05", "PCA",
     "Retains 99&nbsp;% of the variance &mdash; typically ≈ 30–60 components."),
    ("06", "Nine models in parallel",
     "KNN, Decision Tree, Logistic Regression, SVM, LDA, XGBoost, BPNN, "
     "CNN, BiLSTM &mdash; each fed its appropriate feature view."),
    ("07", "Leave-one-flange-out validation",
     "Tests generalisation to a flange the model has never seen. The "
     "honest version of the accuracy number."),
    ("08", "Accuracy-weighted ensemble",
     "At competition time, the top-5 models per flange (by independent-"
     "test accuracy) cast a soft-probability vote weighted by their "
     "independent accuracy."),
]

for i in range(0, len(pipeline_steps), 2):
    col_a, col_b = st.columns(2, gap="medium")
    cols = [col_a, col_b]
    for col, step in zip(cols, pipeline_steps[i:i+2]):
        with col:
            step_mini(*step)

thin_divider()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div style='text-align:center;padding:1.4em 0 0.4em 0;
            color:var(--muted);font-size:0.88em;line-height:1.7;'>
    Built for the <b style='color:var(--ink)'>Spring 2026 Machine-Learning
    Final Project Competition</b>.<br>
    Acknowledgements: Midstream Integrity Services (MIS),
    Smart Materials &amp; Structures Lab (SMSL),<br>
    Artificial Intelligence Lab for Monitoring &amp; Inspection (AILMI),
    University of Houston.
</div>
""", unsafe_allow_html=True)
