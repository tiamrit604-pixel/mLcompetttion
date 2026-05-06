"""
persistence.py
==============
Save / load every artifact the app needs at run time.

Files written
-------------
artifacts/
    ind_bundles.pkl       leave-one-flange-out bundles (4 of them).
                          Each bundle stores: classical models, scaler,
                          PCA, area means.  Torch state_dicts are stored
                          separately because joblib can't pickle CUDA
                          tensors safely.
    bpnn_F1.pth ... BiLSTM_F4.pth  state_dicts (12 files)
    bpnn_input_dims.pkl   per-fold BPNN input dimension after PCA
    fold_results.pkl      independent test accuracies per (fold, model)
    dep_results.pkl       dependent-test 70/30 accuracies per model
    wfan_means_*.pkl      competition WFAN means
    weights.pkl           per-flange top-N model weights for the ensemble
    metadata.pkl          random misc info shown in the UI
"""
import os
import pickle
import joblib
import torch
import numpy as np

from .constants import ARTIFACTS_DIR, FLANGES, INPUT_SIZE, CNN_CHANNELS
from .models    import BPNN, CNN, BiLSTM


DEEP_NAMES = ('BPNN', 'CNN', 'BiLSTM')


def ensure_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# =============================================================================
# Save
# =============================================================================
def save_pipeline(ind_bundles, fold_results, dep_results,
                  wfan_means_flat, wfan_means_2d, wfan_means_seq,
                  flange_top_models, flange_weights,
                  metadata=None, out_dir=ARTIFACTS_DIR):
    """Persist everything needed for later prediction."""
    os.makedirs(out_dir, exist_ok=True)

    # 1.  Per-fold deep model state_dicts and bundle skeletons
    bpnn_input_dims = {}
    bundles_for_pickle = {}
    for flange, bundle in ind_bundles.items():
        models = bundle['models']

        # Save state_dicts of any deep models
        for dname in DEEP_NAMES:
            if dname in models:
                sd_path = os.path.join(out_dir, f"{dname}_{flange}.pth")
                torch.save(models[dname].state_dict(), sd_path)

        # Track BPNN input dim (PCA output size; fold-specific)
        if 'BPNN' in models:
            bpnn_input_dims[flange] = bundle['pca'].n_components_

        # Build a picklable bundle: classical models + transforms + means
        classical = {k: v for k, v in models.items() if k not in DEEP_NAMES}
        bundles_for_pickle[flange] = {
            'classical'      : classical,
            'scaler'         : bundle['scaler'],
            'pca'            : bundle['pca'],
            'area_means_flat': bundle['area_means_flat'],
            'area_means_2d'  : bundle['area_means_2d'],
            'area_means_seq' : bundle['area_means_seq'],
        }

    joblib.dump(bundles_for_pickle, os.path.join(out_dir, 'ind_bundles.pkl'))
    joblib.dump(bpnn_input_dims,    os.path.join(out_dir, 'bpnn_input_dims.pkl'))

    # 2.  Test results (used to display tables in the app)
    fold_results_safe = _strip_tensors(fold_results)
    dep_results_safe  = _strip_tensors(dep_results)
    joblib.dump(fold_results_safe, os.path.join(out_dir, 'fold_results.pkl'))
    joblib.dump(dep_results_safe,  os.path.join(out_dir, 'dep_results.pkl'))

    # 3.  WFAN means
    joblib.dump(wfan_means_flat, os.path.join(out_dir, 'wfan_means_flat.pkl'))
    joblib.dump(wfan_means_2d,   os.path.join(out_dir, 'wfan_means_2d.pkl'))
    joblib.dump(wfan_means_seq,  os.path.join(out_dir, 'wfan_means_seq.pkl'))

    # 4.  Per-flange weights
    joblib.dump(flange_top_models,
                os.path.join(out_dir, 'flange_top_models.pkl'))
    joblib.dump(flange_weights,
                os.path.join(out_dir, 'flange_weights.pkl'))

    if metadata is None:
        metadata = {}
    joblib.dump(metadata, os.path.join(out_dir, 'metadata.pkl'))


def _strip_tensors(d):
    """Drop training history (tensors) before pickling for the UI."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _strip_tensors(v)
        elif k == 'history':
            continue
        else:
            try:
                pickle.dumps(v)
                out[k] = v
            except Exception:
                continue
    return out


# =============================================================================
# Load
# =============================================================================
def load_pipeline(artifacts_dir=ARTIFACTS_DIR):
    """
    Load everything saved by :func:`save_pipeline`.  Returns a dict ready
    to be passed straight to :func:`predict.predict_one_file`.
    """
    if not os.path.isdir(artifacts_dir):
        raise FileNotFoundError(
            f"Artifacts directory not found: {artifacts_dir}\n"
            f"Run `python build_models.py --data <folder>` first."
        )

    required = ['ind_bundles.pkl', 'wfan_means_flat.pkl',
                'flange_top_models.pkl', 'flange_weights.pkl']
    for name in required:
        path = os.path.join(artifacts_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artifact: {path}\n"
                "Did the training script finish?  Re-run "
                "`python build_models.py --data <folder>`."
            )

    bundles_skeleton = joblib.load(
        os.path.join(artifacts_dir, 'ind_bundles.pkl'))
    bpnn_input_dims  = joblib.load(
        os.path.join(artifacts_dir, 'bpnn_input_dims.pkl'))
    wfan_means_flat  = joblib.load(
        os.path.join(artifacts_dir, 'wfan_means_flat.pkl'))
    wfan_means_2d    = joblib.load(
        os.path.join(artifacts_dir, 'wfan_means_2d.pkl'))
    wfan_means_seq   = joblib.load(
        os.path.join(artifacts_dir, 'wfan_means_seq.pkl'))
    flange_top_models = joblib.load(
        os.path.join(artifacts_dir, 'flange_top_models.pkl'))
    flange_weights   = joblib.load(
        os.path.join(artifacts_dir, 'flange_weights.pkl'))

    fold_results = (joblib.load(os.path.join(artifacts_dir, 'fold_results.pkl'))
                    if os.path.exists(os.path.join(artifacts_dir,
                                                    'fold_results.pkl'))
                    else {})
    dep_results  = (joblib.load(os.path.join(artifacts_dir, 'dep_results.pkl'))
                    if os.path.exists(os.path.join(artifacts_dir,
                                                    'dep_results.pkl'))
                    else {})
    metadata     = (joblib.load(os.path.join(artifacts_dir, 'metadata.pkl'))
                    if os.path.exists(os.path.join(artifacts_dir,
                                                    'metadata.pkl'))
                    else {})

    # Re-instantiate deep models
    ind_bundles = {}
    from .constants import DEVICE
    for flange, sk in bundles_skeleton.items():
        models = dict(sk['classical'])

        # BPNN
        bpnn_path = os.path.join(artifacts_dir, f'BPNN_{flange}.pth')
        if os.path.exists(bpnn_path):
            in_dim = bpnn_input_dims.get(flange,
                                          sk['pca'].n_components_)
            m = BPNN(input_dim=in_dim).to(DEVICE)
            m.load_state_dict(torch.load(bpnn_path,
                                          map_location=DEVICE))
            m.eval()
            models['BPNN'] = m

        # CNN
        cnn_path = os.path.join(artifacts_dir, f'CNN_{flange}.pth')
        if os.path.exists(cnn_path):
            m = CNN(in_channels=CNN_CHANNELS).to(DEVICE)
            m.load_state_dict(torch.load(cnn_path,
                                          map_location=DEVICE))
            m.eval()
            models['CNN'] = m

        # BiLSTM
        rnn_path = os.path.join(artifacts_dir, f'BiLSTM_{flange}.pth')
        if os.path.exists(rnn_path):
            m = BiLSTM(input_size=INPUT_SIZE, hidden_size=32).to(DEVICE)
            m.load_state_dict(torch.load(rnn_path,
                                          map_location=DEVICE))
            m.eval()
            models['BiLSTM'] = m

        ind_bundles[flange] = {
            'models'         : models,
            'scaler'         : sk['scaler'],
            'pca'            : sk['pca'],
            'area_means_flat': sk['area_means_flat'],
            'area_means_2d'  : sk['area_means_2d'],
            'area_means_seq' : sk['area_means_seq'],
        }

    return {
        'ind_bundles'      : ind_bundles,
        'fold_results'     : fold_results,
        'dep_results'      : dep_results,
        'wfan_means_flat'  : wfan_means_flat,
        'wfan_means_2d'    : wfan_means_2d,
        'wfan_means_seq'   : wfan_means_seq,
        'flange_top_models': flange_top_models,
        'flange_weights'   : flange_weights,
        'metadata'         : metadata,
    }
