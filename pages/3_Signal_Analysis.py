"""
3_Signal_Analysis.py
====================
Section 3 of the app -- exploratory signal analysis.

The user provides one or more recordings AND specifies the torque
level for each (instead of inferring it from a filename, which
restricts where audio can come from). Each recording is segmented
around hammer strikes, then renders side-by-side feature plots
across the three torque classes.

Two input modes are supported:
    A. File upload  (any .m4a / .wav, any naming)
    B. Live browser recording (forced to 48 kHz to match training)

For each input, the user picks the torque label up-front so the
analyser knows which class column the segments belong to.
"""
import os
import sys
import io
import time
import tempfile
import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.constants import CLASS_NAMES, CLASS_COLORS
from utils.audio_io  import load_and_segment, detect_peaks
from utils.plots     import plot_feature_comparison
from utils.theme     import (
    inject_theme, side_nav, app_bar, hero, step_header, disclaimer,
)


st.set_page_config(page_title="Signal Analysis · Flange Detection",
                   page_icon="③", layout="wide",
                   initial_sidebar_state="expanded")
inject_theme()
side_nav(active="analyze")
app_bar(crumb="Signal Analysis", status="Analyser ready")
hero(
    eyebrow="Section ③  •  Educational",
    title_html=(
        "<span class='accent'>Signal Processing</span> &amp; Analysis"
    ),
    subtitle=(
        "See how 0, 25 and 50 ft-lb hammer strikes differ in the time, "
        "frequency, and time-frequency domains."
    ),
    meta=(
        "Provide recordings and pick the torque level for each &mdash; "
        "the analyser segments hammer strikes and renders side-by-side "
        "feature plots across the three classes."
    ),
)


# =============================================================================
# Session state -- the staged-input list
# =============================================================================
# Each item is a dict:
#     {'class_idx': int, 'segments': list of np.ndarray, 'sr': int,
#      'n_hits': int, 'source': str, 'audio_path': str (optional)}
if 'sa_inputs' not in st.session_state:
    st.session_state['sa_inputs'] = []

# Counter to force-rerender the file uploader after we consume a file.
if 'sa_uploader_key' not in st.session_state:
    st.session_state['sa_uploader_key'] = 0
if 'sa_recorder_key' not in st.session_state:
    st.session_state['sa_recorder_key'] = 0


# =============================================================================
# Step 1 -- pick torque + source, stage the recording
# =============================================================================
step_header(
    "1",
    "Stage your recordings",
    "Pick a torque level, then add a file or record live. Repeat as "
    "many times as you want before analysing.",
)

CLASS_OPTIONS = {
    f"{name}  ({['loose', 'partial', 'tight'][i]})": i
    for i, name in enumerate(CLASS_NAMES)
}

cA, cB = st.columns([1, 2.0], gap="large")

with cA:
    st.markdown(
        "<div style='font-weight:600;color:var(--ink-2);"
        "margin-bottom:0.3em;'>Torque level</div>",
        unsafe_allow_html=True,
    )
    chosen_label = st.radio(
        "Torque level",
        list(CLASS_OPTIONS.keys()),
        index=2,
        label_visibility='collapsed',
        key="sa_class_pick",
    )
    chosen_class = CLASS_OPTIONS[chosen_label]

    st.markdown(
        "<div style='font-weight:600;color:var(--ink-2);"
        "margin:1.0em 0 0.3em 0;'>Source</div>",
        unsafe_allow_html=True,
    )
    source_mode = st.radio(
        "Source",
        ["📁 Upload audio file", "🎙 Record live (browser, 48 kHz)"],
        label_visibility='collapsed',
        key="sa_source_mode",
    )

with cB:
    if source_mode.startswith("📁"):
        # ---- Upload mode -----------------------------------------------
        up = st.file_uploader(
            "Drop one .m4a or .wav recording (any number of strikes inside)",
            type=['m4a', 'wav'],
            accept_multiple_files=False,
            key=f"sa_upload_{st.session_state['sa_uploader_key']}",
        )
        if up is not None:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(up.name)[1])
            tmp.write(up.getbuffer())
            tmp.close()
            try:
                segments, sr, n_peaks, _ = load_and_segment(tmp.name)
                st.session_state['sa_inputs'].append({
                    'class_idx': chosen_class,
                    'segments' : segments,
                    'sr'       : sr,
                    'n_hits'   : len(segments),
                    'source'   : f"📁 {up.name}",
                })
                st.session_state['sa_uploader_key'] += 1
                st.success(
                    f"Added: **{up.name}** &mdash; {len(segments)} hammer "
                    f"strikes detected, labelled as **{CLASS_NAMES[chosen_class]}**."
                )
                time.sleep(0.4)
                st.rerun()
            except Exception as e:
                st.error(f"Could not load {up.name}: {e}")
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    else:
        # ---- Live recording mode ---------------------------------------
        disclaimer(
            "<b>Browser recording at 48&nbsp;kHz.</b> The recorder below "
            "asks your browser for the highest sample rate it supports "
            "(48&nbsp;kHz on every modern laptop). For best accuracy "
            "during the competition, an iPhone <code>.m4a</code> upload "
            "is still the gold standard.",
            kind="info",
        )

        # Backend probe -- prefer st.audio_input (Streamlit >= 1.31),
        # fall back to streamlit-mic-recorder.
        rec_bytes = None
        rec_key = f"sa_rec_{st.session_state['sa_recorder_key']}"

        if hasattr(st, 'audio_input'):
            rec = st.audio_input(
                "Record one or more hammer strikes:", key=rec_key,
            )
            if rec is not None:
                rec_bytes = rec.getvalue()
        else:
            try:
                from streamlit_mic_recorder import mic_recorder
                rec = mic_recorder(
                    start_prompt="🎙 Start recording",
                    stop_prompt="⏹ Stop",
                    just_once=False,
                    use_container_width=True,
                    format="wav",
                    key=rec_key,
                )
                if rec and isinstance(rec, dict) and rec.get('bytes'):
                    rec_bytes = rec['bytes']
            except ImportError:
                st.error(
                    "Live recording is not available in this environment.\n\n"
                    "**Either upgrade Streamlit** (`pip install --upgrade "
                    "streamlit`) or **install the community recorder** "
                    "(`pip install streamlit-mic-recorder`)."
                )

        if rec_bytes and len(rec_bytes) > 1024:
            # Decode + force-resample to 48 kHz for parity with training.
            try:
                import soundfile as sf
                import librosa as _librosa

                tmp_in = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.wav')
                tmp_in.write(rec_bytes); tmp_in.close()

                # Try soundfile first (handles most browser WAVs cleanly);
                # fall back to librosa if soundfile complains about the
                # WebM/Opus container some browsers use.
                try:
                    data, sr_in = sf.read(tmp_in.name, dtype='float32',
                                           always_2d=False)
                except Exception:
                    data, sr_in = _librosa.load(tmp_in.name, sr=None,
                                                 mono=True)

                if data.ndim > 1:
                    data = data.mean(axis=1)

                # Force to 48 kHz mono PCM_16 -- matches training corpus.
                if int(sr_in) != 48000:
                    data = _librosa.resample(
                        data.astype(np.float32),
                        orig_sr=int(sr_in),
                        target_sr=48000,
                    )
                    sr_in = 48000

                tmp_out = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.wav')
                tmp_out.close()
                sf.write(tmp_out.name, data.astype(np.float32),
                         48000, subtype='PCM_16')

                segments, sr, n_peaks, _ = load_and_segment(tmp_out.name)

                if not segments:
                    st.warning(
                        "No hammer strikes detected in this recording. "
                        "Make sure the strikes are loud and clearly above "
                        "the background noise."
                    )
                else:
                    st.session_state['sa_inputs'].append({
                        'class_idx': chosen_class,
                        'segments' : segments,
                        'sr'       : sr,
                        'n_hits'   : len(segments),
                        'source'   : f"🎙 live ({len(data)/48000:.1f} s @ 48 kHz)",
                    })
                    st.session_state['sa_recorder_key'] += 1
                    st.success(
                        f"Added live recording &mdash; {len(segments)} "
                        f"hammer strikes detected, labelled as "
                        f"**{CLASS_NAMES[chosen_class]}**."
                    )
                    time.sleep(0.4)
                    st.rerun()
            except Exception as e:
                st.error(f"Could not decode the browser recording: {e}")


# =============================================================================
# Step 2 -- show the staged inputs
# =============================================================================
inputs = st.session_state['sa_inputs']

step_header(
    "2",
    "Staged recordings",
    f"{len(inputs)} recording{'s' if len(inputs) != 1 else ''} ready for "
    "analysis.",
)

if not inputs:
    st.markdown(
        "<div style='padding:1.4em;border:1.5px dashed var(--border-strong);"
        "border-radius:14px;background:var(--surface-2);text-align:center;"
        "color:var(--muted);font-size:0.95em;'>"
        "No recordings staged yet. Use the controls above to add at least "
        "one recording per torque class for a meaningful comparison."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    # Coverage summary
    counts = {0: 0, 1: 0, 2: 0}
    for it in inputs:
        counts[it['class_idx']] += it['n_hits']

    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.5], gap="small")
    for i, col in enumerate([col1, col2, col3]):
        c = CLASS_COLORS[i]
        with col:
            st.markdown(
                f"<div style='padding:0.85em 1.0em;border:1px solid "
                f"var(--border);border-left:4px solid {c};border-radius:"
                f"var(--radius-md);background:var(--surface);"
                f"box-shadow:var(--shadow-xs);'>"
                f"<div style='font-size:0.78em;font-weight:700;letter-spacing:"
                f"0.1em;text-transform:uppercase;color:var(--muted);'>"
                f"{CLASS_NAMES[i]}</div>"
                f"<div style='font-size:1.6em;font-weight:800;color:{c};"
                f"margin-top:0.15em;line-height:1;'>"
                f"{counts[i]}<span style='font-size:0.55em;color:var(--muted);"
                f"margin-left:0.3em;font-weight:600;'>hits</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with col4:
        if st.button("Clear all", use_container_width=True, key="sa_clear"):
            st.session_state['sa_inputs'] = []
            st.rerun()

    # Per-input expander list
    with st.expander(f"📜  Show all {len(inputs)} staged recordings",
                     expanded=False):
        for idx, it in enumerate(inputs):
            cc = CLASS_COLORS[it['class_idx']]
            cn = CLASS_NAMES[it['class_idx']]
            cl, cr = st.columns([5, 1])
            with cl:
                st.markdown(
                    f"<div style='padding:0.5em 0;'>"
                    f"<span style='display:inline-block;width:10px;"
                    f"height:10px;border-radius:50%;background:{cc};"
                    f"margin-right:0.6em;vertical-align:middle;'></span>"
                    f"<b>{cn}</b> &middot; {it['n_hits']} strikes "
                    f"&middot; <span style='color:var(--muted);'>"
                    f"{it['source']}</span></div>",
                    unsafe_allow_html=True,
                )
            with cr:
                if st.button("Remove", key=f"rm_{idx}",
                             use_container_width=True):
                    st.session_state['sa_inputs'].pop(idx)
                    st.rerun()


# =============================================================================
# Step 3 -- analyse + render side-by-side feature plots
# =============================================================================
if not inputs:
    st.stop()

step_header(
    "3",
    "Analyse",
    "Pick which feature views to render across the three torque classes.",
)

cv1, cv2 = st.columns([2, 1], gap="large")

with cv1:
    available_views = {
        'time'      : "Time series",
        'envelope'  : "Hilbert envelope",
        'psd'       : "Power Spectral Density",
        'stft'      : "STFT magnitude (dB)",
        'mfcc'      : "MFCC",
        'mfcc_delta': "MFCC delta (Δ)",
        'logmel'    : "Log-Mel spectrogram",
    }
    chosen_views = st.multiselect(
        "Feature views to render",
        list(available_views.keys()),
        default=['time', 'psd', 'stft', 'mfcc'],
        format_func=lambda k: available_views[k],
    )

with cv2:
    n_per_class = st.number_input(
        "Samples per class to plot",
        min_value=1, max_value=8, value=3, step=1,
        help="Each row of every panel is one randomly-selected hit "
             "from the staged recordings of that class.",
    )

run_btn = st.button("📈  Render side-by-side comparison",
                     type="primary", use_container_width=True)

if not run_btn:
    st.stop()

if not chosen_views:
    st.warning("Pick at least one feature view first.")
    st.stop()

# Aggregate segments per class
segments_by_class = {0: [], 1: [], 2: []}
sr_by_class       = {0: None, 1: None, 2: None}
for it in inputs:
    c = it['class_idx']
    segments_by_class[c].extend(it['segments'])
    sr_by_class[c] = it['sr']

# Need at least n_per_class hits in each class actually used in views.
populated = {c: len(segments_by_class[c]) for c in (0, 1, 2)}
empty = [CLASS_NAMES[c] for c, n in populated.items() if n == 0]
if empty:
    st.error(
        f"Cannot run side-by-side comparison: no recordings staged for "
        f"**{', '.join(empty)}**. Add at least one recording per torque "
        f"class above."
    )
    st.stop()

# Random sample n_per_class strikes per class.
rng = np.random.default_rng(seed=42)
sampled = {}
for c in (0, 1, 2):
    pool = segments_by_class[c]
    take = min(n_per_class, len(pool))
    idx  = rng.choice(len(pool), size=take, replace=False)
    sampled[c] = [pool[int(i)] for i in idx]

sr = next(s for s in sr_by_class.values() if s is not None)

# Render one figure per chosen view.
for v in chosen_views:
    st.markdown(
        f"<h3 style='margin-top:1.2em'>{available_views[v]}</h3>",
        unsafe_allow_html=True,
    )
    with st.spinner(f"Rendering {available_views[v]}..."):
        try:
            n_rows = max(len(sampled[c]) for c in (0, 1, 2))
            fig = plot_feature_comparison(
                sampled, sr, feature_type=v, n_per_class=n_rows,
            )
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render {available_views[v]}: {e}")
