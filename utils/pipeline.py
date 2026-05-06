"""
pipeline.py
===========
End-to-end pipeline orchestration -- this is what the "Train your own
model" page runs and what the offline ``build_models.py`` script runs.

A single call to :func:`run_full_pipeline` reproduces every step of the
notebook (Sections 6 - 26):

    1. Load all labelled files; segment around peaks; extract features.
    2. Apply WFAN (within-flange-area normalisation).
    3. Standardise + PCA on training partition only (no leakage).
    4. Dependent test  : 70/30 stratified split, 9 models.
    5. Independent test: leave-one-flange-out, 9 models per fold.
    6. Pre-compute competition WFAN means and per-flange model bundles.
"""
import os
import io
import joblib
import numpy as np
import torch
import torch.nn as nn
import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from .constants import (
    NUM_CLASSES, FLANGES, AREAS, BATCH_SIZE, DEVICE, SEED,
    CNN_CHANNELS, N_MFCC, FIXED_FRAMES, SEQ_LEN, INPUT_SIZE, FEATURE_DIM,
)
from .audio_io  import load_and_segment, parse_flange_label
from .features  import extract_features, extract_mfcc_2d, extract_mfcc_sequence
from .wfan      import (
    within_area_normalize, within_area_normalize_2d, within_area_normalize_seq,
)
from .models    import BPNN, CNN, BiLSTM, train_model, evaluate

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# Phase 1 -- ingest a labelled corpus
# =============================================================================
def load_corpus(file_paths, progress_callback=None):
    """
    Load every file, segment around peaks, extract all three feature
    representations.  Files whose names don't match the labelling
    convention are skipped silently.

    Parameters
    ----------
    file_paths : list of str | list of (path, original_name) tuples
        If tuples are given, ``original_name`` is used for label parsing
        (because Streamlit gives random temp names to uploaded files).
    progress_callback : callable(i, n, message) | None

    Returns
    -------
    dict with keys
        X_flat, X_2d, X_seq, y, flange_labels, area_labels,
        n_loaded, n_skipped, n_peaks_skipped
    """
    X_flat, X_2d, X_seq = [], [], []
    y_all, flange_labels, area_labels = [], [], []
    n_loaded = n_skipped = n_peaks_skipped = 0
    skipped_names = []

    for i, item in enumerate(file_paths):
        if isinstance(item, tuple):
            path, original_name = item
        else:
            path, original_name = item, os.path.basename(item)

        if progress_callback is not None:
            progress_callback(i, len(file_paths),
                              f"Loading {original_name}...")

        parsed = parse_flange_label(original_name)
        if parsed is None:
            n_skipped += 1
            skipped_names.append(original_name)
            continue
        torque, flange_id, area, label = parsed

        try:
            segments, sr, _, _ = load_and_segment(path)
        except Exception:
            n_skipped += 1
            skipped_names.append(original_name)
            continue

        if not segments:
            n_skipped += 1
            skipped_names.append(original_name)
            continue

        for seg in segments:
            try:
                X_flat.append(extract_features(seg, sr))
                X_2d.append(extract_mfcc_2d(seg, sr))
                X_seq.append(extract_mfcc_sequence(seg, sr))
                y_all.append(label)
                flange_labels.append(flange_id)
                area_labels.append(area)
            except Exception:
                n_peaks_skipped += 1
                continue
        n_loaded += 1

    if not X_flat:
        raise RuntimeError(
            "No valid labelled audio files were found. Filenames must "
            "follow the format <torque>ftlb<Fx><Ay>.m4a (e.g. "
            "0ftlbF1A1.m4a)."
        )

    return {
        'X_flat'         : np.asarray(X_flat,        dtype=np.float32),
        'X_2d'           : np.asarray(X_2d,          dtype=np.float32),
        'X_seq'          : np.asarray(X_seq,         dtype=np.float32),
        'y'              : np.asarray(y_all,         dtype=np.int64),
        'flange_labels'  : np.asarray(flange_labels),
        'area_labels'    : np.asarray(area_labels),
        'n_loaded'       : n_loaded,
        'n_skipped'      : n_skipped,
        'n_peaks_skipped': n_peaks_skipped,
        'skipped_names'  : skipped_names,
    }


# =============================================================================
# Loader helpers
# =============================================================================
def _make_loader(X, y, shuffle, batch_size=BATCH_SIZE):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle)


def _make_loader_pair(X_tr, X_ts, y_tr, y_ts, batch_size=BATCH_SIZE):
    return (
        _make_loader(X_tr, y_tr, shuffle=True,  batch_size=batch_size),
        _make_loader(X_ts, y_ts, shuffle=False, batch_size=batch_size),
    )


# =============================================================================
# Phase 2 -- dependent test (70/30 stratified split, all 4 flanges pooled)
# =============================================================================
def run_dependent_test(corpus, n_epochs=80, patience=15,
                        progress_callback=None, train_dl=True):
    """
    Pool all 4 flanges, split 70/30 with stratification, train every
    model.  This is the optimistic upper bound; it does NOT measure
    generalisation to a new flange.
    """
    X_flat = corpus['X_flat']
    X_2d   = corpus['X_2d']
    X_seq  = corpus['X_seq']
    y      = corpus['y']
    flanges = corpus['flange_labels']
    areas   = corpus['area_labels']

    indices = np.arange(len(y))
    idx_tr, idx_ts = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=y)

    flanges_tr, areas_tr = flanges[idx_tr], areas[idx_tr]
    flanges_ts, areas_ts = flanges[idx_ts], areas[idx_ts]

    # WFAN on flat / 2d / seq -- means computed from training only
    X_flat_tr_wfan, area_means_flat = within_area_normalize(
        X_flat[idx_tr], flanges_tr, areas_tr)
    X_flat_ts_wfan, _ = _apply_means(X_flat[idx_ts], flanges_ts, areas_ts,
                                     area_means_flat)

    X_2d_tr_wfan, area_means_2d = within_area_normalize_2d(
        X_2d[idx_tr], flanges_tr, areas_tr)
    X_2d_ts_wfan, _ = _apply_means_nd(X_2d[idx_ts], flanges_ts, areas_ts,
                                       area_means_2d)

    X_seq_tr_wfan, area_means_seq = within_area_normalize_seq(
        X_seq[idx_tr], flanges_tr, areas_tr)
    X_seq_ts_wfan, _ = _apply_means_nd(X_seq[idx_ts], flanges_ts, areas_ts,
                                        area_means_seq)

    # Standardise + PCA on flat (fit on training only)
    scaler = StandardScaler()
    X_flat_tr_sc = scaler.fit_transform(X_flat_tr_wfan)
    X_flat_ts_sc = scaler.transform(X_flat_ts_wfan)

    pca = PCA(n_components=0.99, random_state=SEED)
    X_flat_tr_pca = pca.fit_transform(X_flat_tr_sc)
    X_flat_ts_pca = pca.transform(X_flat_ts_sc)

    y_tr, y_ts = y[idx_tr], y[idx_ts]
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_tr)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))

    results, models = {}, {}

    # --- classical models on PCA features ---------------------------------
    classical = _train_classical(X_flat_tr_pca, y_tr, X_flat_ts_pca, y_ts,
                                  results, models)

    # --- LDA --------------------------------------------------------------
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(X_flat_tr_pca, y_tr)
    pred = lda.predict(X_flat_ts_pca)
    results['LDA'] = {
        'acc': accuracy_score(y_ts, pred),
        'pred': pred, 'true': y_ts,
    }
    models['LDA'] = lda

    # --- Deep models -------------------------------------------------------
    if train_dl:
        results, models = _train_deep(
            X_flat_tr_pca, X_flat_ts_pca,
            X_2d_tr_wfan,  X_2d_ts_wfan,
            X_seq_tr_wfan, X_seq_ts_wfan,
            y_tr, y_ts, crit,
            n_epochs, patience, progress_callback,
            results, models,
        )

    return {
        'results'          : results,
        'models'           : models,
        'scaler'           : scaler,
        'pca'              : pca,
        'area_means_flat'  : area_means_flat,
        'area_means_2d'    : area_means_2d,
        'area_means_seq'   : area_means_seq,
        'idx_tr'           : idx_tr,
        'idx_ts'           : idx_ts,
        'y_tr'             : y_tr,
        'y_ts'             : y_ts,
    }


def _apply_means(X, flange_ids, area_ids, area_means):
    """Apply pre-computed means to a flat array."""
    X_norm = X.copy().astype(np.float64)
    for i in range(len(X)):
        key = f"{flange_ids[i]}{area_ids[i]}"
        if key in area_means:
            X_norm[i] = X[i] - area_means[key]
    return X_norm.astype(np.float32), area_means


def _apply_means_nd(X, flange_ids, area_ids, area_means):
    """Apply pre-computed means to a multi-dim array."""
    X_norm = X.copy().astype(np.float32)
    for i in range(len(X)):
        key = f"{flange_ids[i]}{area_ids[i]}"
        if key in area_means:
            X_norm[i] = X[i] - area_means[key]
    return X_norm, area_means


def _train_classical(X_tr, y_tr, X_ts, y_ts, results, models):
    """Train KNN, DT, LR, SVM, XGBoost with grid-searched hyperparameters."""
    grids = {
        'KNN': (KNeighborsClassifier(),
                {'n_neighbors': [3, 5, 7, 9],
                 'weights': ['uniform', 'distance'],
                 'metric': ['euclidean', 'manhattan']}),
        'DT' : (DecisionTreeClassifier(random_state=SEED),
                {'max_depth': [3, 5, 8, 12],
                 'criterion': ['gini', 'entropy'],
                 'min_samples_split': [2, 5, 10]}),
        'LR' : (LogisticRegression(max_iter=2000, solver='lbfgs',
                                    random_state=SEED),
                {'C': [0.01, 0.1, 1, 10, 100]}),
        'SVM': (SVC(probability=True, random_state=SEED),
                [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                 {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100],
                  'gamma': ['scale', 0.01, 0.1]}]),
    }
    for name, (estimator, grid) in grids.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gs = GridSearchCV(estimator, grid, cv=5,
                              scoring='f1_macro', n_jobs=-1)
            gs.fit(X_tr, y_tr)
        pred = gs.best_estimator_.predict(X_ts)
        results[name] = {
            'acc' : accuracy_score(y_ts, pred),
            'pred': pred,
            'true': y_ts,
            'best_params': gs.best_params_,
        }
        models[name] = gs.best_estimator_

    if XGBOOST_AVAILABLE:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gxgb = GridSearchCV(
                XGBClassifier(eval_metric='mlogloss',
                              random_state=SEED, n_jobs=-1),
                {'n_estimators': [200, 400], 'max_depth': [4, 6],
                 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]},
                cv=5, scoring='f1_macro', n_jobs=-1)
            gxgb.fit(X_tr, y_tr)
        pred = gxgb.best_estimator_.predict(X_ts)
        results['XGBoost'] = {
            'acc' : accuracy_score(y_ts, pred),
            'pred': pred,
            'true': y_ts,
        }
        models['XGBoost'] = gxgb.best_estimator_
    return results, models


def _train_deep(X_flat_tr, X_flat_ts, X_2d_tr, X_2d_ts,
                X_seq_tr, X_seq_ts, y_tr, y_ts, crit,
                n_epochs, patience, progress_callback,
                results, models):
    """Train BPNN, CNN, BiLSTM."""
    flat_tr_dl, flat_ts_dl = _make_loader_pair(X_flat_tr, X_flat_ts, y_tr, y_ts)
    cnn_tr_dl,  cnn_ts_dl  = _make_loader_pair(X_2d_tr,   X_2d_ts,   y_tr, y_ts)
    seq_tr_dl,  seq_ts_dl  = _make_loader_pair(X_seq_tr,  X_seq_ts,  y_tr, y_ts)

    deep_specs = [
        ('BPNN',
         BPNN(input_dim=X_flat_tr.shape[1]).to(DEVICE),
         flat_tr_dl, flat_ts_dl, 1e-3, 1e-4),
        ('CNN',
         CNN(in_channels=CNN_CHANNELS).to(DEVICE),
         cnn_tr_dl, cnn_ts_dl, 5e-4, 1e-3),
        ('BiLSTM',
         BiLSTM(input_size=INPUT_SIZE, hidden_size=32).to(DEVICE),
         seq_tr_dl, seq_ts_dl, 3e-4, 1e-3),
    ]

    def _wrap_progress(model_name, total_models, model_idx):
        def cb(epoch, n_epochs_, tr_acc, ts_acc):
            if progress_callback is None:
                return
            base = model_idx / total_models
            frac = (epoch / n_epochs_) / total_models
            progress_callback(
                base + frac,
                1.0,
                f"{model_name}: epoch {epoch}/{n_epochs_}  "
                f"train {tr_acc*100:.1f}%  test {ts_acc*100:.1f}%",
            )
        return cb

    for i, (name, model, tr_dl, ts_dl, lr, wd) in enumerate(deep_specs):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-5)
        model, hist = train_model(
            model, tr_dl, ts_dl, opt, crit, n_epochs, patience, DEVICE,
            scheduler=sch,
            progress_callback=_wrap_progress(name, len(deep_specs), i),
            model_name=name,
        )
        ts_acc, _, pred, true = evaluate(model, ts_dl, crit, DEVICE)
        results[name] = {
            'acc' : ts_acc, 'pred': np.asarray(pred), 'true': np.asarray(true),
            'history': hist,
        }
        models[name] = model

    return results, models


# =============================================================================
# Phase 3 -- independent test (leave-one-flange-out)
# =============================================================================
def run_independent_test(corpus, n_epochs=80, patience=15,
                          progress_callback=None, train_dl=True):
    """
    Leave-one-flange-out cross-validation.  Returns one bundle per held-out
    flange containing the trained models, scaler, PCA and WFAN means
    needed to reproduce the per-flange test scenario at prediction time.
    """
    X_flat = corpus['X_flat']
    X_2d   = corpus['X_2d']
    X_seq  = corpus['X_seq']
    y      = corpus['y']
    flanges = corpus['flange_labels']
    areas   = corpus['area_labels']

    fold_bundles = {}
    fold_results = {}

    n_folds = len(FLANGES)
    for fi, test_flange in enumerate(FLANGES):
        train_flanges = [f for f in FLANGES if f != test_flange]
        train_mask = np.isin(flanges, train_flanges)
        test_mask  = (flanges == test_flange)

        # WFAN means computed on training partition only
        X_flat_tr_wfan, area_means_flat = within_area_normalize(
            X_flat[train_mask], flanges[train_mask], areas[train_mask])
        X_2d_tr_wfan, area_means_2d = within_area_normalize_2d(
            X_2d[train_mask], flanges[train_mask], areas[train_mask])
        X_seq_tr_wfan, area_means_seq = within_area_normalize_seq(
            X_seq[train_mask], flanges[train_mask], areas[train_mask])

        # For the test side we MUST also subtract a per-area mean.  We
        # approximate the held-out flange's area means using the *test*
        # samples themselves (one mean per area on the held-out flange).
        # This is exactly what the notebook does via per-flange WFAN
        # means -- the held-out flange's area centroid is unknown, so we
        # use its own samples to centre.
        X_flat_ts_wfan, ts_means_flat = within_area_normalize(
            X_flat[test_mask], flanges[test_mask], areas[test_mask])
        X_2d_ts_wfan, ts_means_2d = within_area_normalize_2d(
            X_2d[test_mask], flanges[test_mask], areas[test_mask])
        X_seq_ts_wfan, ts_means_seq = within_area_normalize_seq(
            X_seq[test_mask], flanges[test_mask], areas[test_mask])

        # Combine training and test means in a single dictionary so
        # competition-time predictions can retrieve any (flange, area)
        # mean by key.
        combined_flat = {**area_means_flat, **ts_means_flat}
        combined_2d   = {**area_means_2d,   **ts_means_2d}
        combined_seq  = {**area_means_seq,  **ts_means_seq}

        # Standardise + PCA fitted on training only
        scaler = StandardScaler()
        X_flat_tr_sc = scaler.fit_transform(X_flat_tr_wfan)
        X_flat_ts_sc = scaler.transform(X_flat_ts_wfan)

        pca = PCA(n_components=0.99, random_state=SEED)
        X_flat_tr_pca = pca.fit_transform(X_flat_tr_sc)
        X_flat_ts_pca = pca.transform(X_flat_ts_sc)

        y_tr, y_ts = y[train_mask], y[test_mask]

        cw = compute_class_weight('balanced',
                                  classes=np.arange(NUM_CLASSES), y=y_tr)
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))

        results, models = {}, {}

        if progress_callback is not None:
            progress_callback(fi / n_folds, 1.0,
                              f"Fold {test_flange}: classical models...")

        _train_classical(X_flat_tr_pca, y_tr, X_flat_ts_pca, y_ts,
                          results, models)

        # LDA
        lda = LinearDiscriminantAnalysis(solver='svd')
        lda.fit(X_flat_tr_pca, y_tr)
        pred = lda.predict(X_flat_ts_pca)
        results['LDA'] = {
            'acc' : accuracy_score(y_ts, pred),
            'pred': pred, 'true': y_ts,
        }
        models['LDA'] = lda

        if train_dl:
            def fold_cb(frac, total, msg):
                if progress_callback is not None:
                    progress_callback(
                        (fi + frac) / n_folds, 1.0,
                        f"Fold {test_flange}: {msg}")

            _train_deep(
                X_flat_tr_pca, X_flat_ts_pca,
                X_2d_tr_wfan,  X_2d_ts_wfan,
                X_seq_tr_wfan, X_seq_ts_wfan,
                y_tr, y_ts, crit,
                n_epochs, patience, fold_cb,
                results, models,
            )

        fold_results[test_flange] = results
        fold_bundles[test_flange] = {
            'models'         : models,
            'scaler'         : scaler,
            'pca'            : pca,
            'area_means_flat': combined_flat,
            'area_means_2d'  : combined_2d,
            'area_means_seq' : combined_seq,
            'y_tr'           : y_tr,
            'y_ts'           : y_ts,
        }

    return fold_results, fold_bundles


# =============================================================================
# Phase 4 -- competition WFAN means (for unseen recordings)
# =============================================================================
def compute_competition_wfan_means(corpus):
    """
    Compute (flange, area)-specific means across the full labelled corpus.

    These means are used at competition time to centre each unseen recording
    by the same (flange, area) population mean that was used during training.
    """
    X_flat = corpus['X_flat']
    X_2d   = corpus['X_2d']
    X_seq  = corpus['X_seq']
    flange_labels = corpus['flange_labels']
    area_labels   = corpus['area_labels']

    means_flat, means_2d, means_seq = {}, {}, {}
    for f in FLANGES:
        for a in AREAS:
            key  = f"{f}{a}"
            mask = (flange_labels == f) & (area_labels == a)
            if mask.sum() == 0:
                continue
            means_flat[key] = X_flat[mask].mean(axis=0)
            means_2d[key]   = X_2d[mask].mean(axis=0)
            means_seq[key]  = X_seq[mask].mean(axis=0)
    return means_flat, means_2d, means_seq
