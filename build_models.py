"""
build_models.py
===============
One-time training script.  Run this ONCE on your labelled corpus to
populate the `artifacts/` folder.  After that, the Streamlit app loads
the artifacts directly and can predict instantly without retraining.

Usage
-----
    python build_models.py --data /path/to/your/data
    python build_models.py --data ./Data/Iphone --epochs 80 --top-n 5

Folder structure expected
-------------------------
The data folder should contain .m4a (or .wav) files named
<torque>ftlb<Fx><Ay>.<ext>, e.g.

    0ftlbF1A1.m4a   25ftlbF1A1.m4a   50ftlbF1A1.m4a
    0ftlbF1A2.m4a   25ftlbF1A2.m4a   50ftlbF1A2.m4a
    ...
    0ftlbF4A4.m4a   25ftlbF4A4.m4a   50ftlbF4A4.m4a

48 files total per session, 16 areas (4 flanges x 4 areas) per torque level.
"""
import os
import sys
import argparse
import time
import warnings

# Make `utils` importable regardless of where the script is invoked from
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.pipeline    import (
    load_corpus, run_dependent_test, run_independent_test,
    compute_competition_wfan_means,
)
from utils.predict     import build_per_flange_weights
from utils.persistence import save_pipeline, ARTIFACTS_DIR


def collect_files(data_dir):
    """Walk the data folder and gather all .m4a / .wav paths."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.lower().endswith(('.m4a', '.wav')):
                paths.append(os.path.join(root, fn))
    return sorted(paths)


def cli_progress(i, n, msg=""):
    """Tiny progress printer for terminal use."""
    if isinstance(i, float):
        pct = i * 100
    else:
        pct = (i + 1) / max(n, 1) * 100
    sys.stdout.write(f"\r  [{pct:6.1f}%] {msg[:80]:<80}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Train all LOFO models and save artifacts.")
    parser.add_argument(
        '--data', required=True,
        help="Folder containing labelled .m4a / .wav files."
    )
    parser.add_argument(
        '--epochs', type=int, default=80,
        help="Max epochs per deep model (default 80).")
    parser.add_argument(
        '--patience', type=int, default=15,
        help="Early-stopping patience (default 15).")
    parser.add_argument(
        '--top-n', type=int, default=5,
        help="Top-N models per flange in the weighted ensemble (default 5).")
    parser.add_argument(
        '--out', default=ARTIFACTS_DIR,
        help=f"Output artifacts directory (default {ARTIFACTS_DIR}).")
    parser.add_argument(
        '--no-dl', action='store_true',
        help="Skip deep learning models (much faster, classical only).")
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    print("=" * 70)
    print("  Bolted-Flange Looseness Detection -- Offline Training Pipeline")
    print("=" * 70)
    print(f"  Data folder : {args.data}")
    print(f"  Output dir  : {args.out}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Top-N       : {args.top_n}")
    print(f"  Deep models : {'disabled' if args.no_dl else 'enabled'}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1 -- ingest
    # ------------------------------------------------------------------
    file_paths = collect_files(args.data)
    if not file_paths:
        print(f"\n[ERROR] No .m4a / .wav files found in {args.data}")
        sys.exit(1)
    print(f"\nFound {len(file_paths)} audio files.")

    print("\nPhase 1/4 -- Loading and feature extraction...")
    t0 = time.time()
    corpus = load_corpus(file_paths, progress_callback=cli_progress)
    print(f"\n  Loaded {corpus['n_loaded']} files, "
          f"{corpus['X_flat'].shape[0]} segments  "
          f"({time.time()-t0:.1f}s)")
    print(f"  Skipped {corpus['n_skipped']} files (bad name or load error)")

    if corpus['n_skipped']:
        print("\n  First few skipped names:")
        for s in corpus['skipped_names'][:5]:
            print(f"    - {s}")

    # ------------------------------------------------------------------
    # Phase 2 -- dependent test (all-flanges-pooled 70/30)
    # ------------------------------------------------------------------
    print("\nPhase 2/4 -- Dependent test (70/30 stratified split)...")
    t0 = time.time()
    dep = run_dependent_test(corpus,
                              n_epochs=args.epochs,
                              patience=args.patience,
                              progress_callback=cli_progress,
                              train_dl=not args.no_dl)
    print(f"\n  Dependent test done ({time.time()-t0:.1f}s)")
    print(f"  Per-model accuracy:")
    for m, r in sorted(dep['results'].items(),
                       key=lambda kv: -kv[1]['acc']):
        print(f"    {m:<10}: {r['acc']*100:6.2f}%")

    # ------------------------------------------------------------------
    # Phase 3 -- independent test (leave-one-flange-out)
    # ------------------------------------------------------------------
    print("\nPhase 3/4 -- Independent test (leave-one-flange-out)...")
    t0 = time.time()
    fold_results, ind_bundles = run_independent_test(
        corpus,
        n_epochs=args.epochs,
        patience=args.patience,
        progress_callback=cli_progress,
        train_dl=not args.no_dl,
    )
    print(f"\n  Independent test done ({time.time()-t0:.1f}s)")
    print(f"  Per-fold mean accuracy:")
    for flange, results in fold_results.items():
        mean_acc = sum(r['acc'] for r in results.values()) / len(results)
        print(f"    {flange} held-out : mean={mean_acc*100:6.2f}%")

    # ------------------------------------------------------------------
    # Phase 4 -- competition WFAN means + per-flange weights + save
    # ------------------------------------------------------------------
    print("\nPhase 4/4 -- Computing competition WFAN means + saving...")
    wfan_flat, wfan_2d, wfan_seq = compute_competition_wfan_means(corpus)

    flange_top_models, flange_weights = build_per_flange_weights(
        fold_results, top_n=args.top_n)

    print(f"\n  Top-{args.top_n} models per flange:")
    for flange in flange_top_models:
        print(f"    {flange}: {flange_top_models[flange]}")

    metadata = {
        'n_files'      : corpus['n_loaded'],
        'n_segments'   : int(corpus['X_flat'].shape[0]),
        'n_flanges'    : 4,
        'n_areas'      : 4,
        'top_n'        : args.top_n,
        'epochs'       : args.epochs,
        'training_time': time.time() - t0,
    }

    save_pipeline(
        ind_bundles      = ind_bundles,
        fold_results     = fold_results,
        dep_results      = dep['results'],
        wfan_means_flat  = wfan_flat,
        wfan_means_2d    = wfan_2d,
        wfan_means_seq   = wfan_seq,
        flange_top_models= flange_top_models,
        flange_weights   = flange_weights,
        metadata         = metadata,
        out_dir          = args.out,
    )

    print("\n" + "=" * 70)
    print(f"  All artifacts saved to {args.out}")
    print("=" * 70)
    print("\nNext step: launch the Streamlit app:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
