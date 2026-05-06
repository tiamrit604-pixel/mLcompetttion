"""
2_Competition_Prediction.py
===========================
Section 2 of the app -- predict torque on a new recording using the
pre-trained leave-one-flange-out bundles + accuracy-weighted ensemble.

Two input modes:
    A. Upload .m4a / .wav file (recommended -- iPhone-style recording)
    B. Live browser microphone (best-effort; mic recording quality
       differs from the iPhone training data, so a clear disclaimer is
       shown)
"""
import os
import sys
import io
import tempfile
import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.constants   import CLASS_NAMES, FLANGES, AREAS, CLASS_COLORS
from utils.persistence import load_pipeline
from utils.predict     import predict_one_file, predict_from_array
from utils.plots       import (
    plot_waveform_with_peaks, plot_segment_time_series,
    plot_probability_bar,
)
from utils.audio_io    import detect_peaks
from utils.theme       import (
    inject_theme, side_nav, app_bar, hero, result_hero, disclaimer, step_header,
)


st.set_page_config(page_title="Predict · Flange Detection", page_icon="②",
                   layout="wide", initial_sidebar_state="expanded")
inject_theme()
side_nav(active="predict")
app_bar(crumb="Competition Prediction", status="Pre-trained models loaded")

# --- Page hero -------------------------------------------------------------
hero(
    eyebrow="Section ②  •  Live demo",
    title_html=(
        "<span class='accent'>Competition</span> Prediction"
    ),
    subtitle=(
        "Predict torque (0 / 25 / 50 ft-lb) on an unseen recording using "
        "the pre-trained leave-one-flange-out bundles and an "
        "accuracy-weighted ensemble."
    ),
    meta=(
        "Pipeline &nbsp;·&nbsp; <b>WFAN</b> + PCA + 9 models + "
        "per-flange ensemble &nbsp;·&nbsp; Inference time &lt; 2 s"
    ),
)


# =============================================================================
# Load the saved pipeline (cached so it only happens once)
# =============================================================================
@st.cache_resource(show_spinner="Loading pre-trained models...")
def _load():
    return load_pipeline()


try:
    bundle_dict = _load()
except Exception as e:
    st.error(
        "❌ **No trained artifacts found.**\n\n"
        f"```\n{e}\n```\n\n"
        "Either:\n"
        "1. Run `python build_models.py --data <your-folder>` once from "
        "the project root, or\n"
        "2. Use Section 1 (Train Your Own Model) and click "
        "*Save trained bundle*."
    )
    st.stop()

ind_bundles      = bundle_dict['ind_bundles']
wfan_flat        = bundle_dict['wfan_means_flat']
wfan_2d          = bundle_dict['wfan_means_2d']
wfan_seq         = bundle_dict['wfan_means_seq']
flange_top_models = bundle_dict['flange_top_models']
flange_weights   = bundle_dict['flange_weights']
fold_results     = bundle_dict['fold_results']
metadata         = bundle_dict['metadata']

# Append model bundle info to the sidebar (below the nav)
with st.sidebar:
    st.markdown(
        "<div class='side-section' style='margin-top:0.5em;'>Loaded bundle</div>",
        unsafe_allow_html=True,
    )
    if metadata:
        meta_rows = "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:0.3em 0.95em;font-size:0.84em;'>"
            f"<span style='color:var(--muted);'>{k}</span>"
            f"<span style='color:var(--ink);font-weight:600;'>{v}</span>"
            f"</div>"
            for k, v in list(metadata.items())[:8]
        )
        st.markdown(
            f"<div style='margin:0 0.6em 0.4em 0.6em;border:1px solid "
            f"var(--border);border-radius:var(--radius-md);"
            f"background:var(--surface);padding:0.4em 0;"
            f"box-shadow:var(--shadow-xs);'>{meta_rows}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div style='margin:0.6em 1.0em;font-size:0.78em;"
        "color:var(--muted);'>"
        "<b style='color:var(--ink-2);font-weight:600;'>Available flanges</b>"
        f"<div style='margin-top:0.3em;'>{', '.join(sorted(ind_bundles.keys()))}"
        "</div></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Selection: flange, area, mode
# =============================================================================
step_header(
    "1",
    "Select the recording context",
    "WFAN needs the flange ID and striking area to subtract the right "
    "(flange, area) mean before classification.",
)

c1, c2, c3 = st.columns(3)
with c1:
    flange_id = st.selectbox(
        "Flange ID",
        FLANGES,
        help="Which flange does the recording come from? "
             "WFAN means depend on this.",
    )
with c2:
    area_id = st.selectbox(
        "Striking area",
        AREAS,
        help="Which of the four marked areas was struck? "
             "Each area has its own learned mean.",
    )
with c3:
    mode = st.radio(
        "Input mode",
        ["📁 Upload audio file", "🎙 Live browser recording"],
        index=0,
    )

# Show top-N + weights for the chosen flange (transparency)
with st.expander(f"ℹ How will {flange_id} be predicted?"):
    if flange_id in flange_top_models:
        tops    = flange_top_models[flange_id]
        weights = flange_weights[flange_id]
        st.markdown(
            f"For **{flange_id}**, the prediction is a soft-probability "
            f"vote across the **top-{len(tops)}** models from the "
            "leave-one-flange-out test:"
        )
        rows = []
        for r, m in enumerate(tops, 1):
            acc = (fold_results.get(flange_id, {})
                   .get(m, {}).get('acc', None))
            acc_str = f"{acc*100:.2f}%" if acc is not None else "n/a"
            rows.append((r, m, acc_str, f"{weights[m]:.4f}"))
        st.dataframe(
            rows, hide_index=True,
            column_config={
                "0": "Rank", "1": "Model",
                "2": "Independent test acc", "3": "Weight",
            },
            use_container_width=True,
        )


# =============================================================================
# Input
# =============================================================================
step_header(
    "2",
    "Provide the recording",
    "Upload an iPhone .m4a (recommended) or record live in the browser.",
)

audio_path  = None
audio_bytes = None
audio_array = None
sr_array    = None

if mode == "📁 Upload audio file":
    f = st.file_uploader(
        "Upload `.m4a` (iPhone) or `.wav` recording",
        type=['m4a', 'wav'],
        accept_multiple_files=False,
        key="comp_upload",
    )
    if f is not None:
        audio_bytes = f.read()
        st.audio(audio_bytes)
        # Persist to a temp path because audioread needs a file
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(f.name)[-1] or '.m4a')
        tmp.write(audio_bytes); tmp.close()
        audio_path = tmp.name

else:  # Live browser recording
    disclaimer(
        "<b>⚠ Disclaimer.</b> The training data was recorded with an iPhone "
        "at 48 kHz. Browser-microphone recordings often use a different "
        "sample rate, gain, and frequency response, which can degrade "
        "prediction accuracy. For competition predictions, an iPhone "
        "<code>.m4a</code> upload is strongly recommended.",
        kind="warn",
    )

    # ---- pick the best available recorder backend --------------------------
    # Priority order:
    #   1. st.audio_input  (Streamlit >= 1.31, native, no extra dep)
    #   2. streamlit-mic-recorder  (community, works on older Streamlit)
    # If neither is available, we surface a clear install instruction
    # rather than crashing with an AttributeError.
    audio_bytes = None
    audio_format_hint = 'audio/wav'

    if hasattr(st, 'audio_input'):
        # ---- Backend 1: native st.audio_input ----------------------------
        st.caption("Recorder: native Streamlit `st.audio_input` "
                   "(Streamlit ≥ 1.31).")
        rec = st.audio_input(
            "Record one or more hammer strikes (start, hit, stop):",
            key="comp_rec_native",
        )
        if rec is not None:
            audio_bytes = rec.getvalue()
    else:
        # ---- Backend 2: streamlit-mic-recorder ---------------------------
        try:
            from streamlit_mic_recorder import mic_recorder
            st.caption("Recorder: `streamlit-mic-recorder` "
                       "(your Streamlit version is < 1.31).")
            rec = mic_recorder(
                start_prompt="🎙 Start recording",
                stop_prompt="⏹ Stop",
                just_once=False,
                use_container_width=True,
                format="wav",
                key="comp_rec_mic",
            )
            if rec is not None and isinstance(rec, dict) and rec.get('bytes'):
                audio_bytes = rec['bytes']
        except ImportError:
            st.error(
                "**Live browser recording is unavailable in this "
                "environment.**\n\n"
                "Your installed Streamlit version is older than 1.31, "
                "which is when `st.audio_input` was introduced.\n\n"
                "**Pick one of these to enable live recording:**\n\n"
                "**Option A — Upgrade Streamlit (recommended):**\n"
                "```bash\n"
                "pip install --upgrade streamlit>=1.32\n"
                "```\n\n"
                "**Option B — Install the community recorder:**\n"
                "```bash\n"
                "pip install streamlit-mic-recorder\n"
                "```\n\n"
                "Then restart the app: `streamlit run app.py`.\n\n"
                "**Or just use the *Upload audio file* mode** above — "
                "it works on every Streamlit version, and an iPhone "
                ".m4a upload gives the best accuracy anyway."
            )

    # ---- decode whatever bytes we ended up with ---------------------------
    if audio_bytes is not None:
        if len(audio_bytes) < 1024:
            st.warning("Recording is empty or too short. Please record again.")
        else:
            st.audio(audio_bytes, format=audio_format_hint)

            # Write to a temp file with .wav suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()

            # Some browsers send WebM/Opus wrapped in an audio_input,
            # others send proper WAV. Decode with soundfile (permissive),
            # fall back to librosa, then resample to 48 kHz (matches the
            # iPhone-recorded training corpus exactly) and re-encode to
            # clean PCM_16 mono WAV.
            try:
                import soundfile as sf
                import librosa as _librosa
                data, sr_in = sf.read(tmp.name, dtype='float32',
                                       always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)

                # Force-resample to 48 kHz if browser captured at a
                # different rate (most laptops can do 48 kHz natively
                # but Safari sometimes returns 44.1 kHz).
                resampled_note = ""
                if int(sr_in) != 48000:
                    data = _librosa.resample(
                        data.astype(np.float32),
                        orig_sr=int(sr_in), target_sr=48000,
                    )
                    resampled_note = f" (resampled from {sr_in} Hz)"
                    sr_in = 48000

                clean_path = tmp.name.replace('.wav', '_clean.wav')
                sf.write(clean_path, data.astype(np.float32),
                         48000, subtype='PCM_16')
                audio_path = clean_path
                st.caption(
                    f"✅ Recorded: {len(data) / 48000:.2f} s at "
                    f"48 kHz, mono{resampled_note}."
                )
            except Exception as decode_err:
                try:
                    import librosa
                    import soundfile as sf
                    data, sr_in = librosa.load(tmp.name, sr=48000, mono=True)
                    clean_path = tmp.name.replace('.wav', '_clean.wav')
                    sf.write(clean_path, data.astype('float32'),
                             48000, subtype='PCM_16')
                    audio_path = clean_path
                    st.caption(
                        f"✅ Recorded: {len(data) / 48000:.2f} s at "
                        f"48 kHz (librosa fallback)."
                    )
                except Exception as e2:
                    st.error(
                        "Could not decode the browser recording.\n\n"
                        f"soundfile said: `{decode_err}`\n\n"
                        f"librosa said: `{e2}`\n\n"
                        "**Fix:** use the *Upload audio file* mode instead, "
                        "or check that your browser microphone is enabled."
                    )


# =============================================================================
# Predict
# =============================================================================
predict_btn = st.button(
    "🔍 Run prediction",
    type="primary",
    disabled=audio_path is None,
    use_container_width=True,
)

if predict_btn and audio_path is not None:
    if flange_id not in ind_bundles:
        st.error(f"No trained bundle for {flange_id}.")
        st.stop()

    bundle = ind_bundles[flange_id]
    tops    = flange_top_models.get(flange_id, [])
    weights = flange_weights.get(flange_id, {})

    with st.spinner("Running pipeline..."):
        try:
            res = predict_one_file(
                audio_path, flange_id, area_id, bundle,
                wfan_flat, wfan_2d, wfan_seq,
                tops, weights,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    if res is None:
        st.error(
            "No hammer-strike peaks could be detected in the recording. "
            "Make sure the file actually contains a clean strike."
        )
        st.stop()

    # =====================================================================
    # Headline result
    # =====================================================================
    final = res['final_class_name']
    conf  = res['confidence'] * 100

    result_hero(
        class_index=int(res['final_class']),
        predicted_label=final,
        pills=[
            ("Confidence",        f"{conf:.1f}%"),
            ("Hammer strikes",    str(res['n_hits'])),
            ("Flange",            flange_id),
            ("Area",              area_id),
        ],
        status_text="Live",
    )

    # =====================================================================
    # Probability bar
    # =====================================================================
    fig = plot_probability_bar(
        res['weighted_proba'], res['final_class'],
        title="Accuracy-weighted ensemble probabilities",
    )
    st.pyplot(fig, use_container_width=False)

    # =====================================================================
    # Per-model breakdown
    # =====================================================================
    st.subheader("Per-model votes")
    st.caption(
        "Shows what each top-N model predicts for every detected hit. "
        "The weighted vote above is the accuracy-weighted soft-probability "
        "average of all of these."
    )

    rows = []
    for m in tops:
        if m not in res['per_hit_preds']:
            continue
        preds = res['per_hit_preds'][m]
        probs = res['per_hit_probs'][m]
        per_class = np.bincount(preds, minlength=3)
        mean_prob = probs.mean(axis=0)
        rows.append({
            'Model'      : m,
            'Weight'     : f"{weights[m]:.3f}",
            '0 ft-lb'    : f"{per_class[0]} hits ({mean_prob[0]*100:.1f}%)",
            '25 ft-lb'   : f"{per_class[1]} hits ({mean_prob[1]*100:.1f}%)",
            '50 ft-lb'   : f"{per_class[2]} hits ({mean_prob[2]*100:.1f}%)",
            'Majority'   : CLASS_NAMES[int(np.argmax(np.bincount(preds, minlength=3)))],
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)

    # =====================================================================
    # Waveform + segments preview
    # =====================================================================
    st.subheader("Waveform & extracted segments")
    peaks_arr = detect_peaks(res['signal'], res['sr'])
    fig_wav = plot_waveform_with_peaks(
        res['signal'], res['sr'], peaks_arr,
        title=f"Full recording -- {res['n_hits']} strikes detected",
    )
    st.pyplot(fig_wav, use_container_width=True)

    show_n = min(4, len(res['segments']))
    if show_n:
        st.caption(f"First {show_n} extracted strike segments:")
        cols = st.columns(show_n)
        for i, (col, seg) in enumerate(zip(cols, res['segments'][:show_n])):
            with col:
                fig_s = plot_segment_time_series(
                    seg, res['sr'], title=f"Strike #{i+1}",
                    color=CLASS_COLORS[res['final_class']],
                )
                st.pyplot(fig_s, use_container_width=True)

    # Cleanup the temp file
    try:
        os.unlink(audio_path)
    except OSError:
        pass


# =============================================================================
# Batch / "competition mode" -- predict all 4 flanges at once
# =============================================================================
st.divider()
with st.expander(
    "🏆 Batch prediction -- predict all 4 flanges from competition files",
    expanded=False,
):
    st.write(
        "Upload all your competition files at once (one or more files per "
        "flange). The app groups them by flange, runs the prediction "
        "pipeline on each, and shows one row per flange in the final "
        "table -- exactly the format requested in the project brief."
    )

    batch_files = st.file_uploader(
        "Upload competition files (must be named like `F1A1.m4a`, "
        "`F2A2.m4a`, etc.)",
        type=['m4a', 'wav'],
        accept_multiple_files=True,
        key="comp_batch_uploader",
    )

    if batch_files and st.button("Run batch prediction",
                                  type="primary",
                                  use_container_width=True):
        # Group by (flange, area) inferred from filename
        import re
        rgx = re.compile(r'(F[1-4])(A[1-4])', re.IGNORECASE)

        per_flange_results = {f: [] for f in FLANGES}
        progress = st.progress(0.0)

        tmp_dir = tempfile.mkdtemp(prefix='comp_batch_')

        for i, f in enumerate(batch_files):
            progress.progress((i + 1) / len(batch_files),
                              text=f"Processing {f.name}...")
            m = rgx.search(f.name)
            if not m:
                st.warning(f"Skipping {f.name}: cannot infer flange/area "
                           "from filename.")
                continue
            flange = m.group(1).upper()
            area   = m.group(2).upper()

            tmp_path = os.path.join(tmp_dir, f.name)
            with open(tmp_path, 'wb') as out:
                out.write(f.getbuffer())

            bundle = ind_bundles[flange]
            tops   = flange_top_models[flange]
            wts    = flange_weights[flange]

            try:
                r = predict_one_file(
                    tmp_path, flange, area, bundle,
                    wfan_flat, wfan_2d, wfan_seq, tops, wts)
                per_flange_results[flange].append({
                    'file'         : f.name,
                    'area'         : area,
                    'proba'        : r['weighted_proba'],
                    'pred'         : r['final_class'],
                    'n_hits'       : r['n_hits'],
                    'per_hit_preds': r['per_hit_preds'],   # for the grid plot
                    'per_hit_probs': r['per_hit_probs'],
                })
            except Exception as e:
                st.warning(f"Failed on {f.name}: {e}")

        progress.empty()

        # =====================================================================
        # Headline table (per-flange final prediction)
        # =====================================================================
        st.markdown("### 🏁 Final per-flange predictions")
        rows = []
        for flange in FLANGES:
            entries = per_flange_results[flange]
            if not entries:
                rows.append({
                    'Flange': flange,
                    'Files': 0,
                    'Total hits': 0,
                    'P(0 ft-lb)': '-',
                    'P(25 ft-lb)': '-',
                    'P(50 ft-lb)': '-',
                    'Predicted': 'no data',
                })
                continue
            mean_proba = np.mean([e['proba'] for e in entries], axis=0)
            pred       = int(np.argmax(mean_proba))
            rows.append({
                'Flange'     : flange,
                'Files'      : len(entries),
                'Total hits' : sum(e['n_hits'] for e in entries),
                'P(0 ft-lb)' : f"{mean_proba[0]*100:.1f}%",
                'P(25 ft-lb)': f"{mean_proba[1]*100:.1f}%",
                'P(50 ft-lb)': f"{mean_proba[2]*100:.1f}%",
                'Predicted'  : CLASS_NAMES[pred],
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

        # =====================================================================
        # The headline grid plot (4 flanges x top-N models)
        # =====================================================================
        # Only flanges that actually got data
        flanges_with_data = {
            f: per_flange_results[f]
            for f in FLANGES
            if per_flange_results[f]
        }
        if flanges_with_data:
            from utils.plots import plot_competition_grid
            st.markdown("### 📊 Competition prediction grid")
            st.caption(
                "Each row is a flange. Each column is one of the top-5 "
                "models that voted for that flange (rank-1 marked with ★). "
                "Each bar shows segmented hit counts per striking area, "
                "stacked by predicted torque class. The boxed annotation "
                "in the rightmost panel of each row is the final accuracy-"
                "weighted ensemble decision."
            )
            with st.spinner("Rendering competition grid..."):
                fig_grid = plot_competition_grid(
                    flanges_with_data,
                    flange_top_models,
                    flange_weights,
                )
            st.pyplot(fig_grid, use_container_width=True)

            # Download as PNG -- handy for the poster file
            buf_img = io.BytesIO()
            fig_grid.savefig(buf_img, format='png',
                              dpi=200, bbox_inches='tight',
                              facecolor='white')
            buf_img.seek(0)
            st.download_button(
                "⬇ Download grid as PNG",
                data=buf_img.getvalue(),
                file_name="competition_prediction_grid.png",
                mime="image/png",
                use_container_width=False,
            )
