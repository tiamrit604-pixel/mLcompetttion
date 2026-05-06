"""
wfan.py
=======
Within-Flange-Area Normalisation (WFAN).

Each (flange, area) combination has its own acoustic fingerprint -- the
shape of the flange, the local material thickness, and even the surface
finish at the strike point shift the spectrum.  Subtracting the per-area
mean removes those nuisance factors and leaves the torque-dependent
component that the classifier should actually learn.

Mirrors Section 9B of the notebook.  Three parallel functions because the
three feature representations (flat, 2-D image, sequence) must each be
normalised in their own native shape.
"""
import numpy as np


def within_area_normalize(X, flange_ids, area_ids,
                           fit_flanges=None, fit_areas=None):
    """
    Subtract per-(flange, area) mean from every sample.

    Each area in each flange gets its own mean.  16 separate means for
    4 flanges * 4 areas.

    Parameters
    ----------
    X           : (N, D) feature matrix
    flange_ids  : (N,) array of flange strings ('F1' ... 'F4')
    area_ids    : (N,) array of area strings   ('A1' ... 'A4')
    fit_flanges : list of flanges to fit on (None = all observed)
    fit_areas   : list of areas to fit on   (None = all observed)

    Returns
    -------
    X_norm     : (N, D) normalised features (float32)
    area_means : dict {'F1A1': (D,), 'F1A2': (D,), ...}
    """
    X_norm     = X.copy().astype(np.float64)
    area_means = {}
    flanges    = fit_flanges if fit_flanges is not None else np.unique(flange_ids)
    areas      = fit_areas   if fit_areas   is not None else np.unique(area_ids)

    for f in flanges:
        for a in areas:
            key  = f"{f}{a}"
            mask = (flange_ids == f) & (area_ids == a)
            if mask.sum() > 0:
                area_means[key] = X[mask].mean(axis=0)

    for i in range(len(X)):
        key = f"{flange_ids[i]}{area_ids[i]}"
        if key in area_means:
            X_norm[i] = X[i] - area_means[key]

    return X_norm.astype(np.float32), area_means


def within_area_normalize_2d(X_2d, flange_ids, area_ids,
                              fit_flanges=None, fit_areas=None):
    """WFAN for CNN inputs (N, C, H, W)."""
    X_norm     = X_2d.copy().astype(np.float32)
    area_means = {}
    flanges    = fit_flanges if fit_flanges is not None else np.unique(flange_ids)
    areas      = fit_areas   if fit_areas   is not None else np.unique(area_ids)
    for f in flanges:
        for a in areas:
            key  = f"{f}{a}"
            mask = (flange_ids == f) & (area_ids == a)
            if mask.sum() > 0:
                area_means[key] = X_2d[mask].mean(axis=0)
    for i in range(len(X_2d)):
        key = f"{flange_ids[i]}{area_ids[i]}"
        if key in area_means:
            X_norm[i] = X_2d[i] - area_means[key]
    return X_norm, area_means


def within_area_normalize_seq(X_seq, flange_ids, area_ids,
                               fit_flanges=None, fit_areas=None):
    """WFAN for BiLSTM inputs (N, T, F)."""
    X_norm     = X_seq.copy().astype(np.float32)
    area_means = {}
    flanges    = fit_flanges if fit_flanges is not None else np.unique(flange_ids)
    areas      = fit_areas   if fit_areas   is not None else np.unique(area_ids)
    for f in flanges:
        for a in areas:
            key  = f"{f}{a}"
            mask = (flange_ids == f) & (area_ids == a)
            if mask.sum() > 0:
                area_means[key] = X_seq[mask].mean(axis=0)
    for i in range(len(X_seq)):
        key = f"{flange_ids[i]}{area_ids[i]}"
        if key in area_means:
            X_norm[i] = X_seq[i] - area_means[key]
    return X_norm, area_means


def apply_wfan(features, area_key, area_means):
    """
    Apply a stored area-specific mean to a single sample or batch.
    Used at prediction time when we know the (flange, area) of the
    incoming recording.
    """
    if area_key not in area_means:
        raise KeyError(
            f"Area key '{area_key}' not found in trained WFAN means. "
            f"Available keys: {sorted(area_means.keys())}"
        )
    return features - area_means[area_key]
