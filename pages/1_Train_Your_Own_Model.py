"""
1_Train_Your_Own_Model.py
=========================
Section 1 of the app -- upload a labelled corpus and run the full
training pipeline:

    Load files -> WFAN -> PCA -> dependent test -> independent test
                                        -> save trained bundle for download
"""
import os
import sys
import io
import time
import zipfile
import tempfile
import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.constants   import CLASS_NAMES, FLANGES, AREAS
from utils.pipeline    import (
    load_corpus, run_dependent_test, run_independent_test,
    compute_competition_wfan_means,
)
from utils.predict     import build_per_flange_weights
from utils.persistence import save_pipeline
from utils.plots       import (
    plot_confusion_matrix, plot_results_bar, plot_per_flange_accuracy,
)
from utils.theme       import (
    inject_theme, side_nav, app_bar, hero, disclaimer,
)


st.set_page_config(page_title="Train · Flange Detection", page_icon="①",
                   layout="wide", initial_sidebar_state="expanded")
inject_theme()
side_nav(active="train")
app_bar(crumb="Train Your Own Model", status="Pipeline ready")


# =============================================================================
# Header
# =============================================================================
hero(
    eyebrow="Section ①  •  Training workflow",
    title_html="<span class='accent'>Train</span> Your Own Model",
    subtitle=(
        "Run the full training pipeline on your own labelled audio: "
        "feature extraction, WFAN, PCA, 9-model training, and "
        "leave-one-flange-out validation."
    ),
    meta=(
        "Output &nbsp;·&nbsp; trained bundle saved to "
        "<code>artifacts/</code>, used by Section ② for prediction"
    ),
)

disclaimer(
    "<b>Heavy &amp; time-consuming.</b> Training all nine models across "
    "the leave-one-flange-out folds typically takes "
    "<b>5–20&nbsp;minutes on CPU</b> and 1–3&nbsp;minutes on GPU. "
    "Use the toggle below to skip deep models if you only need a quick "
    "smoke-test.",
    kind="warn",
)


# =============================================================================
# File-naming convention helper
# =============================================================================
with st.expander("📋 Required filename convention -- click to view", expanded=False):
    st.markdown("""
        Each uploaded file must be named in the format
        `<torque>ftlb<Fx><Ay>.<ext>`:

        | Component | Allowed values            | Example |
        |-----------|---------------------------|---------|
        | torque    | `0`, `25`, `50`           | `25`    |
        | flange    | `F1`, `F2`, `F3`, `F4`    | `F2`    |
        | area      | `A1`, `A2`, `A3`, `A4`    | `A3`    |
        | extension | `.m4a` or `.wav`          | `.m4a`  |

        ✅ `0ftlbF1A1.m4a` &nbsp;&nbsp;&nbsp; ✅ `25ftlbF2A3.m4a` &nbsp;&nbsp;&nbsp; ✅ `50ftlbF4A4.m4a`

        ✗ `0ftLB_F1_A1.m4a` (underscores) &nbsp;&nbsp;&nbsp;
        ✗ `looseF1A1.m4a` (no torque value)

        For the full corpus you need **3 torque levels x 4 flanges x 4 areas = 48 files**.
    """)


# =============================================================================
# Settings
# =============================================================================
st.subheader("Training Settings")

c1, c2, c3 = st.columns(3)
with c1:
    n_epochs = st.number_input(
        "Max epochs per deep model", min_value=10, max_value=300,
        value=80, step=10,
        help="Early stopping uses the patience setting below.")
with c2:
    patience = st.number_input(
        "Early-stopping patience", min_value=3, max_value=50,
        value=15, step=1)
with c3:
    top_n = st.number_input(
        "Top-N models per flange (ensemble)",
        min_value=2, max_value=9, value=5, step=1,
        help="Used for the per-flange accuracy-weighted soft vote.")

train_dl = st.toggle(
    "Include deep-learning models (BPNN + CNN + BiLSTM)",
    value=True,
    help="Disable for a faster classical-only run."
)


# =============================================================================
# Upload
# =============================================================================
st.subheader("Upload Your Labelled Corpus")
uploaded_files = st.file_uploader(
    "Select all your `.m4a` / `.wav` files (you can multi-select).",
    type=['m4a', 'wav'],
    accept_multiple_files=True,
    key="train_uploader",
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} files queued.")

    # Quick label preview
    from utils.audio_io import parse_flange_label
    parsed_summary = {}
    for f in uploaded_files:
        p = parse_flange_label(f.name)
        if p is None:
            continue
        torque, flange, area, _ = p
        key = (torque, flange)
        parsed_summary[key] = parsed_summary.get(key, 0) + 1
    bad = sum(1 for f in uploaded_files
              if parse_flange_label(f.name) is None)
    if bad:
        st.warning(
            f"{bad} file(s) do not match the naming convention and will be "
            "skipped."
        )


# =============================================================================
# Run training
# =============================================================================
run_btn = st.button(
    "🚀 Run Full Pipeline",
    type="primary",
    disabled=not uploaded_files,
    use_container_width=True,
)

if run_btn and uploaded_files:
    # Persist uploads to a temp dir so they have real paths
    tmp_dir = tempfile.mkdtemp(prefix="flange_train_")
    file_pairs = []
    for f in uploaded_files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, 'wb') as out:
            out.write(f.getbuffer())
        file_pairs.append((path, f.name))

    progress = st.progress(0.0, text="Starting...")
    status   = st.empty()

    def cb(i, n, msg=""):
        if isinstance(i, float):
            frac = max(0.0, min(1.0, i))
        else:
            frac = (i + 1) / max(n, 1)
        progress.progress(frac, text=msg)
        status.caption(msg)

    # ---- Phase 1: load corpus
    status.info("Phase 1/4 -- Loading audio + extracting features...")
    try:
        corpus = load_corpus(file_pairs, progress_callback=cb)
    except Exception as e:
        progress.empty(); status.empty()
        st.error(f"❌ Loading failed: {e}")
        st.stop()

    st.success(
        f"Loaded {corpus['n_loaded']} files, "
        f"extracted {corpus['X_flat'].shape[0]} segments. "
        f"Skipped {corpus['n_skipped']} files."
    )

    # ---- Phase 2: dependent test
    status.info("Phase 2/4 -- Dependent test (70/30 stratified split)...")
    progress.progress(0.05, text="Dependent test...")
    t0 = time.time()
    dep = run_dependent_test(
        corpus, n_epochs=n_epochs, patience=patience,
        progress_callback=cb, train_dl=train_dl,
    )
    st.session_state['dep_results'] = dep['results']
    dep_time = time.time() - t0
    st.success(f"Dependent test done in {dep_time:.1f}s.")

    # ---- Decide whether to run independent test
    do_ind = st.session_state.get('do_independent', True)

    # ---- Phase 3: independent test
    status.info("Phase 3/4 -- Independent test (leave-one-flange-out)...")
    t0 = time.time()
    fold_results, ind_bundles = run_independent_test(
        corpus, n_epochs=n_epochs, patience=patience,
        progress_callback=cb, train_dl=train_dl,
    )
    ind_time = time.time() - t0
    st.success(f"Independent test done in {ind_time:.1f}s.")

    # ---- Phase 4: WFAN means + per-flange weights
    status.info("Phase 4/4 -- Competition WFAN means + per-flange weights...")
    wfan_flat, wfan_2d, wfan_seq = compute_competition_wfan_means(corpus)
    flange_top_models, flange_weights = build_per_flange_weights(
        fold_results, top_n=top_n)

    # Stash everything in session state for downstream tabs
    st.session_state.update({
        'fold_results'   : fold_results,
        'ind_bundles'    : ind_bundles,
        'dep_results'    : dep['results'],
        'wfan_flat'      : wfan_flat,
        'wfan_2d'        : wfan_2d,
        'wfan_seq'       : wfan_seq,
        'flange_top_models': flange_top_models,
        'flange_weights' : flange_weights,
        'corpus_meta'    : {
            'n_files'    : corpus['n_loaded'],
            'n_segments' : int(corpus['X_flat'].shape[0]),
        },
        'training_done'  : True,
    })

    progress.progress(1.0, text="Done.")
    status.success("All phases complete -- inspect results below.")


# =============================================================================
# Results dashboard
# =============================================================================
if st.session_state.get('training_done'):
    st.divider()
    st.subheader("📊 Results Dashboard")

    tab_dep, tab_ind, tab_cm, tab_save = st.tabs([
        "Dependent Test (70/30)",
        "Independent Test (LOFO)",
        "Confusion Matrices",
        "Save / Download",
    ])

    # -----------------------------------------------------------------------
    # Dependent test tab
    # -----------------------------------------------------------------------
    with tab_dep:
        dep_results = st.session_state['dep_results']

        st.write(
            "Pooled 70/30 stratified split across all 4 flanges. The "
            "training and test sets share the same physical flanges, so "
            "this is the **upper bound** on accuracy -- not a measure of "
            "generalisation."
        )

        # Bar chart
        fig = plot_results_bar(dep_results,
                                title="Dependent test accuracy (3-class)")
        st.pyplot(fig, use_container_width=True)

        # Binary version
        fig_bin = plot_results_bar(dep_results,
                                    title="Dependent test accuracy "
                                          "(binary loose vs tight)",
                                    binary=True)
        st.pyplot(fig_bin, use_container_width=True)

        # Numeric table
        rows = [(m, f"{r['acc']*100:.2f}%") for m, r in
                sorted(dep_results.items(), key=lambda kv: -kv[1]['acc'])]
        st.dataframe(rows, column_config={
            "0": "Model",
            "1": "Test accuracy",
        }, hide_index=True, use_container_width=True)

    # -----------------------------------------------------------------------
    # Independent test tab
    # -----------------------------------------------------------------------
    with tab_ind:
        fold_results = st.session_state['fold_results']

        st.write(
            "Leave-one-flange-out: in each fold, one full flange is held "
            "out and the model is trained on the other three. This is the "
            "**meaningful generalisation test** -- whether the model works "
            "on a flange it has never seen."
        )

        c_left, c_right = st.columns([2, 1])

        with c_left:
            fig = plot_per_flange_accuracy(
                fold_results,
                title="Independent test accuracy (3-class) per held-out flange",
            )
            st.pyplot(fig, use_container_width=True)

        with c_right:
            # Mean per model
            all_models = set()
            for fr in fold_results.values():
                all_models.update(fr.keys())
            rows = []
            for m in sorted(all_models):
                accs = [fold_results[f][m]['acc']
                        for f in fold_results
                        if m in fold_results[f]]
                if accs:
                    rows.append((m,
                                 f"{np.mean(accs)*100:.2f}%",
                                 f"{np.std(accs)*100:.2f}%"))
            rows.sort(key=lambda r: -float(r[1].rstrip('%')))
            st.markdown("**Mean ± std across folds**")
            st.dataframe(
                rows,
                column_config={
                    "0": "Model", "1": "Mean acc", "2": "Std",
                },
                hide_index=True,
                use_container_width=True,
            )

        # Per-flange detail
        st.markdown("#### Drill-down: per-flange / per-model accuracy")
        flange_pick = st.radio(
            "Held-out flange",
            FLANGES,
            horizontal=True,
            key="ind_flange_pick",
        )
        if flange_pick in fold_results:
            fr = fold_results[flange_pick]
            rows = [(m, f"{r['acc']*100:.2f}%") for m, r in
                    sorted(fr.items(), key=lambda kv: -kv[1]['acc'])]
            st.dataframe(rows, column_config={
                "0": "Model", "1": "Test accuracy",
            }, hide_index=True, use_container_width=True)

            # Show top-N + weights
            tops    = st.session_state['flange_top_models'][flange_pick]
            weights = st.session_state['flange_weights'][flange_pick]
            st.markdown(
                f"**Top-{len(tops)} models for {flange_pick} "
                "(used in the ensemble):**"
            )
            for r, m in enumerate(tops, 1):
                st.markdown(
                    f"&nbsp;&nbsp; `{r}.` **{m}** &nbsp;&nbsp; "
                    f"acc = {fr[m]['acc']*100:.2f}% &nbsp;&nbsp; "
                    f"weight = {weights[m]:.3f}",
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # Confusion matrix tab
    # -----------------------------------------------------------------------
    with tab_cm:
        st.write(
            "Pick a fold and a model below. The confusion matrix is "
            "rendered for that model on the held-out flange."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            cm_fold = st.selectbox(
                "Held-out flange",
                FLANGES,
                key="cm_fold",
            )
        with c2:
            available_models = list(
                st.session_state['fold_results'][cm_fold].keys())
            cm_model = st.selectbox(
                "Model",
                available_models,
                index=available_models.index('CNN')
                if 'CNN' in available_models else 0,
                key="cm_model",
            )
        with c3:
            binary = st.toggle(
                "Binary (loose vs tight)",
                value=False,
                key="cm_binary",
            )

        r = st.session_state['fold_results'][cm_fold][cm_model]
        title = (f"{cm_model} on held-out {cm_fold}  "
                 f"({'binary' if binary else '3-class'}, "
                 f"acc = {r['acc']*100:.2f}%)")
        fig = plot_confusion_matrix(
            r['true'], r['pred'],
            title=title,
            binary=binary,
        )
        st.pyplot(fig, use_container_width=False)

    # -----------------------------------------------------------------------
    # Save / download tab
    # -----------------------------------------------------------------------
    with tab_save:
        st.write(
            "After saving, the **Competition Prediction** page can load "
            "these models and use them to predict torque on new recordings."
        )

        c_left, c_right = st.columns(2)

        with c_left:
            if st.button("💾 Save trained bundle to "
                         "`artifacts/` (overwrite)",
                         type="primary",
                         use_container_width=True):
                from utils.constants import ARTIFACTS_DIR
                save_pipeline(
                    ind_bundles      = st.session_state['ind_bundles'],
                    fold_results     = st.session_state['fold_results'],
                    dep_results      = st.session_state['dep_results'],
                    wfan_means_flat  = st.session_state['wfan_flat'],
                    wfan_means_2d    = st.session_state['wfan_2d'],
                    wfan_means_seq   = st.session_state['wfan_seq'],
                    flange_top_models= st.session_state['flange_top_models'],
                    flange_weights   = st.session_state['flange_weights'],
                    metadata         = st.session_state['corpus_meta'],
                    out_dir          = ARTIFACTS_DIR,
                )
                st.success(
                    f"Bundle saved to `{ARTIFACTS_DIR}/`. The Competition "
                    "Prediction page will now use this trained bundle."
                )

        with c_right:
            if st.button("📦 Build a zip of the artifacts to download",
                         use_container_width=True):
                with tempfile.TemporaryDirectory() as zip_tmp:
                    save_pipeline(
                        ind_bundles      = st.session_state['ind_bundles'],
                        fold_results     = st.session_state['fold_results'],
                        dep_results      = st.session_state['dep_results'],
                        wfan_means_flat  = st.session_state['wfan_flat'],
                        wfan_means_2d    = st.session_state['wfan_2d'],
                        wfan_means_seq   = st.session_state['wfan_seq'],
                        flange_top_models= st.session_state['flange_top_models'],
                        flange_weights   = st.session_state['flange_weights'],
                        metadata         = st.session_state['corpus_meta'],
                        out_dir          = zip_tmp,
                    )
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, 'w',
                                          zipfile.ZIP_DEFLATED) as zf:
                        for fn in os.listdir(zip_tmp):
                            zf.write(os.path.join(zip_tmp, fn), fn)
                    st.download_button(
                        "⬇ Download flange_artifacts.zip",
                        data=buf.getvalue(),
                        file_name="flange_artifacts.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
