"""
predict.py
==========
Prediction utilities for the *competition* and *live* pages of the app.

The pipeline a single new recording goes through:

    raw audio -> peak detection -> N segments
              -> (flat 221-d, 3-channel image, 64-step MFCC sequence)
              -> WFAN: subtract the (flange, area) mean
              -> standardise + PCA  (flat representation)
              -> top-N models by per-flange independent test accuracy
              -> weighted soft-probability vote -> final class

The accuracy weights themselves come from the leave-one-flange-out test
results.  Models that historically struggle on a given flange contribute
proportionally less when predicting on that flange.
"""
import numpy as np
import torch

from .constants import NUM_CLASSES, CLASS_NAMES, DEVICE, FLANGES
from .audio_io  import load_and_segment, detect_peaks, segment_around_peaks
from .features  import extract_features, extract_mfcc_2d, extract_mfcc_sequence
from .wfan      import apply_wfan


def featurise_segments(segments, sr):
    """Compute all three feature representations for a list of segments."""
    flat = np.asarray([extract_features(s, sr) for s in segments],
                      dtype=np.float32)
    d2   = np.asarray([extract_mfcc_2d(s, sr) for s in segments],
                      dtype=np.float32)
    seq  = np.asarray([extract_mfcc_sequence(s, sr) for s in segments],
                      dtype=np.float32)
    return flat, d2, seq


def predict_one_file(filepath, flange_id, area_id, bundle,
                     wfan_means_flat, wfan_means_2d, wfan_means_seq,
                     top_models, weights):
    """
    Run the full prediction pipeline on one audio file.

    Parameters
    ----------
    filepath        : str
    flange_id       : 'F1'..'F4'  (selects the per-flange bundle)
    area_id         : 'A1'..'A4'  (selects the WFAN mean to subtract)
    bundle          : dict with keys 'models', 'scaler', 'pca'
                      (per-flange bundle from ind_bundles)
    wfan_means_flat : dict {area_key: np.ndarray}
    top_models      : list of model names contributing to the vote
    weights         : dict {model_name: weight}

    Returns
    -------
    dict with keys:
        n_hits           : int
        per_hit_preds    : dict {model_name: (n_hits,) preds}
        per_hit_probs    : dict {model_name: (n_hits, 3) probs}
        weighted_proba   : (3,) weighted-mean probability
        final_class      : int
        final_class_name : str
        confidence       : float (max prob)
        segments         : list of np.ndarray (waveform segments, for plotting)
        sr               : int
    """
    segments, sr, _, signal = load_and_segment(filepath)
    if not segments:
        return None
    flat, d2, seq = featurise_segments(segments, sr)

    area_key = f"{flange_id}{area_id}"

    flat_wfan = apply_wfan(flat, area_key, wfan_means_flat)
    d2_wfan   = apply_wfan(d2,   area_key, wfan_means_2d)
    seq_wfan  = apply_wfan(seq,  area_key, wfan_means_seq)

    # PCA on flat representation
    flat_sc  = bundle['scaler'].transform(flat_wfan)
    flat_pca = bundle['pca'].transform(flat_sc)

    per_hit_preds = {}
    per_hit_probs = {}

    for mname in top_models:
        if mname in {'BPNN', 'CNN', 'BiLSTM'}:
            model = bundle['models'].get(mname)
            if model is None:
                continue
            model.eval()
            with torch.no_grad():
                if mname == 'BPNN':
                    t = torch.tensor(flat_pca, dtype=torch.float32).to(DEVICE)
                elif mname == 'CNN':
                    t = torch.tensor(d2_wfan, dtype=torch.float32).to(DEVICE)
                else:                           # BiLSTM
                    t = torch.tensor(seq_wfan, dtype=torch.float32).to(DEVICE)
                logits = model(t)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
        else:
            model = bundle['models'].get(mname)
            if model is None:
                continue
            preds = model.predict(flat_pca)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(flat_pca)
            else:
                # one-hot fallback
                probs = np.eye(NUM_CLASSES)[preds]

        per_hit_preds[mname] = preds
        per_hit_probs[mname] = probs

    # Weighted soft probability across hits AND models
    n_hits = len(segments)
    weighted_proba = np.zeros(NUM_CLASSES)
    weight_total   = 0.0
    for mname in top_models:
        if mname not in per_hit_probs:
            continue
        w = weights.get(mname, 0.0)
        # average over hits, then scale by weight
        weighted_proba += w * per_hit_probs[mname].mean(axis=0)
        weight_total   += w
    if weight_total > 0:
        weighted_proba = weighted_proba / weight_total
    final_class = int(np.argmax(weighted_proba))
    confidence  = float(weighted_proba[final_class])

    return {
        'n_hits'          : n_hits,
        'per_hit_preds'   : per_hit_preds,
        'per_hit_probs'   : per_hit_probs,
        'weighted_proba'  : weighted_proba,
        'final_class'     : final_class,
        'final_class_name': CLASS_NAMES[final_class],
        'confidence'      : confidence,
        'segments'        : segments,
        'sr'              : sr,
        'signal'          : signal,
    }


def predict_from_array(signal, sr, flange_id, area_id, bundle,
                        wfan_means_flat, wfan_means_2d, wfan_means_seq,
                        top_models, weights):
    """Same as :func:`predict_one_file` but takes a raw numpy waveform."""
    peaks = detect_peaks(signal, sr)
    segments = segment_around_peaks(signal, peaks, sr)
    if not segments:
        return None

    flat, d2, seq = featurise_segments(segments, sr)
    area_key = f"{flange_id}{area_id}"
    flat_wfan = apply_wfan(flat, area_key, wfan_means_flat)
    d2_wfan   = apply_wfan(d2,   area_key, wfan_means_2d)
    seq_wfan  = apply_wfan(seq,  area_key, wfan_means_seq)
    flat_sc   = bundle['scaler'].transform(flat_wfan)
    flat_pca  = bundle['pca'].transform(flat_sc)

    per_hit_preds, per_hit_probs = {}, {}
    for mname in top_models:
        if mname in {'BPNN', 'CNN', 'BiLSTM'}:
            model = bundle['models'].get(mname)
            if model is None:
                continue
            model.eval()
            with torch.no_grad():
                if mname == 'BPNN':
                    t = torch.tensor(flat_pca, dtype=torch.float32).to(DEVICE)
                elif mname == 'CNN':
                    t = torch.tensor(d2_wfan, dtype=torch.float32).to(DEVICE)
                else:
                    t = torch.tensor(seq_wfan, dtype=torch.float32).to(DEVICE)
                logits = model(t)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
        else:
            model = bundle['models'].get(mname)
            if model is None:
                continue
            preds = model.predict(flat_pca)
            probs = (model.predict_proba(flat_pca)
                     if hasattr(model, 'predict_proba')
                     else np.eye(NUM_CLASSES)[preds])
        per_hit_preds[mname] = preds
        per_hit_probs[mname] = probs

    weighted_proba = np.zeros(NUM_CLASSES)
    weight_total   = 0.0
    for mname in top_models:
        if mname not in per_hit_probs:
            continue
        w = weights.get(mname, 0.0)
        weighted_proba += w * per_hit_probs[mname].mean(axis=0)
        weight_total   += w
    if weight_total > 0:
        weighted_proba = weighted_proba / weight_total
    final_class = int(np.argmax(weighted_proba))

    return {
        'n_hits'          : len(segments),
        'per_hit_preds'   : per_hit_preds,
        'per_hit_probs'   : per_hit_probs,
        'weighted_proba'  : weighted_proba,
        'final_class'     : final_class,
        'final_class_name': CLASS_NAMES[final_class],
        'confidence'      : float(weighted_proba[final_class]),
        'segments'        : segments,
        'sr'              : sr,
        'signal'          : signal,
    }


# =============================================================================
# Build the per-flange "top-N models + weights" lookup
# =============================================================================
def build_per_flange_weights(fold_results, top_n=5):
    """
    Build top-N per flange from leave-one-flange-out results.

    Parameters
    ----------
    fold_results : dict {flange: {model_name: {'acc': ...}}}
    top_n        : int

    Returns
    -------
    flange_top_models : dict {flange: list of model names (descending acc)}
    flange_weights    : dict {flange: {model_name: weight (sums to 1)}}
    """
    flange_top_models, flange_weights = {}, {}
    for flange in FLANGES:
        if flange not in fold_results:
            continue
        accs   = {m: r['acc'] for m, r in fold_results[flange].items()}
        ranked = sorted(accs.items(), key=lambda kv: kv[1], reverse=True)
        top    = ranked[:top_n]
        names  = [m for m, _ in top]
        vals   = np.array([a for _, a in top], dtype=np.float64)
        if vals.sum() == 0:
            weights = {m: 1.0 / len(top) for m in names}
        else:
            ws = vals / vals.sum()
            weights = {m: float(w) for m, w in zip(names, ws)}
        flange_top_models[flange] = names
        flange_weights[flange]    = weights
    return flange_top_models, flange_weights
