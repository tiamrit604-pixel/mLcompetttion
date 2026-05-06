"""
plots.py
========
Matplotlib helpers for every chart used in the app.  All return a
matplotlib Figure (so the page can either display it or save it to disk).
"""
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # headless backend safe for Streamlit Cloud
import seaborn as sns
import librosa
import librosa.display
from scipy.signal import welch, stft

from .constants import CLASS_NAMES, CLASS_COLORS, BLOCK_COLORS, NUM_CLASSES


def _set_clean_style():
    plt.rcParams.update({
        'font.family'   : 'DejaVu Sans',
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'axes.grid'        : True,
        'grid.alpha'       : 0.3,
        'figure.dpi'       : 110,
    })


def plot_waveform_with_peaks(signal, sr, peaks=None, title="Waveform"):
    """Time-series waveform with optional peak markers."""
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(10, 2.6))
    t = np.arange(len(signal)) / sr
    ax.plot(t, signal, linewidth=0.5, color='#2563eb')
    if peaks is not None and len(peaks) > 0:
        ax.scatter(peaks / sr, signal[peaks],
                   color='#dc2626', s=24, zorder=5, label=f'{len(peaks)} peaks')
        ax.legend(loc='upper right')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_segment_time_series(segment, sr, title="Segment time series",
                              color='#2563eb'):
    """Time-domain plot of one extracted segment."""
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 2.4))
    t = np.arange(len(segment)) / sr
    ax.plot(t, segment, linewidth=0.6, color=color)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_psd(segment, sr, title="Power Spectral Density", color='#2563eb',
              normalized=True):
    """
    Welch PSD plot.

    If normalized=True (default) the x-axis is Normalized Frequency (Hz)
    in [0, 0.2], matching the reference style used in this project.
    Set normalized=False to get true Hz on the x-axis.
    """
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 3.0))
    if normalized:
        f, pxx = welch(segment, fs=1.0, nperseg=2048,
                        noverlap=1024, nfft=4096)
        ax.plot(f, pxx, color=color, linewidth=1.0)
        ax.set_xlim(0, 0.2)
        ax.set_xlabel("Normalized Frequency (Hz)")
        ax.set_ylabel("Power / Hz")
    else:
        f, pxx = welch(segment, fs=sr, nperseg=2048,
                        noverlap=1024, nfft=4096)
        ax.semilogy(f, pxx, color=color, linewidth=0.8)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_stft(segment, sr, title="STFT magnitude (dB)"):
    """Short-time Fourier transform spectrogram."""
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    f, t, Z = stft(segment, fs=sr, nperseg=512, noverlap=384)
    Zdb = 20 * np.log10(np.abs(Z) + 1e-9)
    im = ax.pcolormesh(t, f, Zdb, shading='gouraud', cmap='magma')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='dB')
    fig.tight_layout()
    return fig


def plot_mfcc(segment, sr, n_mfcc=32, title="MFCC matrix"):
    """MFCC heat-map."""
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    mfcc = librosa.feature.mfcc(y=segment.astype(np.float64),
                                 sr=sr, n_mfcc=n_mfcc)
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr,
                                    cmap='coolwarm', ax=ax)
    ax.set_title(title); ax.set_ylabel("MFCC #")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def plot_mfcc_delta(segment, sr, n_mfcc=32, title="Delta-MFCC"):
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    mfcc  = librosa.feature.mfcc(y=segment.astype(np.float64),
                                  sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    img = librosa.display.specshow(delta, x_axis='time', sr=sr,
                                    cmap='coolwarm', ax=ax)
    ax.set_title(title); ax.set_ylabel("Delta-MFCC #")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def plot_logmel(segment, sr, n_mels=64, title="Log-Mel spectrogram"):
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    mel = librosa.feature.melspectrogram(y=segment.astype(np.float64),
                                          sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(log_mel, x_axis='time', y_axis='mel',
                                    sr=sr, cmap='viridis', ax=ax)
    ax.set_title(title); fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_envelope(segment, sr, title="Hilbert envelope"):
    """Hilbert envelope shows ring-down behaviour clearly."""
    from scipy.signal import hilbert
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    t  = np.arange(len(segment)) / sr
    env = np.abs(hilbert(segment))
    ax.plot(t, segment, color='#9ca3af', linewidth=0.5, label='Signal')
    ax.plot(t, env,    color='#dc2626', linewidth=1.2, label='Envelope')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_title(title); ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


# =============================================================================
# Confusion matrices and result tables
# =============================================================================
def plot_confusion_matrix(true, pred, class_names=None, title=None,
                          binary=False):
    """Single confusion matrix heat-map."""
    from sklearn.metrics import confusion_matrix
    if binary:
        true = np.where(np.asarray(true) == 0, 0, 1)
        pred = np.where(np.asarray(pred) == 0, 0, 1)
        class_names = ['Loose (0)', 'Tight (25+50)']
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(true, pred)
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_results_bar(results_dict, title="Test accuracy by model",
                     binary=False):
    """Horizontal bar chart of per-model accuracies."""
    _set_clean_style()
    if binary:
        accs = []
        names = []
        for m, r in results_dict.items():
            true = np.where(np.asarray(r['true']) == 0, 0, 1)
            pred = np.where(np.asarray(r['pred']) == 0, 0, 1)
            from sklearn.metrics import accuracy_score
            accs.append(accuracy_score(true, pred))
            names.append(m)
    else:
        names = list(results_dict.keys())
        accs  = [r['acc'] for r in results_dict.values()]

    order = np.argsort(accs)
    names = [names[i] for i in order]
    accs  = [accs[i]  for i in order]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(names, [a * 100 for a in accs],
                   color='#2563eb', edgecolor='white')
    for bar, a in zip(bars, accs):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                f'{a*100:.1f}%', va='center', fontsize=9)
    ax.set_xlim(0, max(105, max(accs) * 100 + 5))
    ax.set_xlabel("Accuracy (%)"); ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_per_flange_accuracy(fold_results, models_to_show=None,
                              title="Independent test accuracy"):
    """Grouped bar chart: x = flange, y = accuracy, hue = model."""
    _set_clean_style()
    flanges = list(fold_results.keys())
    if models_to_show is None:
        all_models = set()
        for fr in fold_results.values():
            all_models.update(fr.keys())
        models_to_show = sorted(all_models)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    width = 0.8 / max(1, len(models_to_show))
    x = np.arange(len(flanges))

    cmap = plt.get_cmap('tab10')
    for i, m in enumerate(models_to_show):
        vals = []
        for f in flanges:
            r = fold_results.get(f, {}).get(m)
            vals.append(r['acc'] * 100 if r else 0)
        ax.bar(x + i * width, vals, width, label=m,
               color=cmap(i % 10), edgecolor='white')
    ax.set_xticks(x + width * (len(models_to_show) - 1) / 2)
    ax.set_xticklabels([f"{f} held-out" for f in flanges])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# =============================================================================
# Side-by-side feature comparison (Section 3 of the app)
# =============================================================================
def plot_feature_comparison(segments_by_class, sr,
                             feature_type='time', n_per_class=1,
                             colors=None):
    """
    Side-by-side comparison of one feature type across classes.

    segments_by_class : dict {class_idx: list of segments}
    feature_type      : 'time' | 'psd' | 'mfcc' | 'mfcc_delta' | 'stft'
                        | 'logmel' | 'envelope'
    """
    _set_clean_style()
    classes_present = sorted(segments_by_class.keys())
    n_cols = len(classes_present)
    if colors is None:
        colors = [CLASS_COLORS[c] for c in classes_present]

    fig, axes = plt.subplots(n_per_class, n_cols,
                              figsize=(4.5 * n_cols, 2.8 * n_per_class),
                              squeeze=False)

    for col, c in enumerate(classes_present):
        for row in range(n_per_class):
            ax = axes[row][col]
            if row >= len(segments_by_class[c]):
                ax.axis('off')
                continue
            seg = segments_by_class[c][row]

            if feature_type == 'time':
                t = np.arange(len(seg)) / sr
                ax.plot(t, seg, linewidth=0.5, color=colors[col])
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
            elif feature_type == 'psd':
                # Normalized-frequency PSD (matches the reference figure):
                # x-axis = normalized frequency (Hz), 0 .. 0.2
                # y-axis = Power / Hz (linear)
                # Setting fs=1.0 yields normalized freq in [0, 0.5];
                # we then clip the view to [0, 0.2] which is where the
                # interesting hammer-strike content lives for our 48 kHz
                # signal (0.2 * 48000 = 9.6 kHz).
                f, pxx = welch(seg, fs=1.0, nperseg=2048,
                                noverlap=1024, nfft=4096)
                ax.plot(f, pxx, color=colors[col], linewidth=1.0)
                ax.set_xlim(0, 0.2)
                ax.set_xlabel("Normalized Frequency (Hz)")
                ax.set_ylabel("Power / Hz")
                ax.grid(True, which='both', alpha=0.4, linewidth=0.6)
                ax.set_axisbelow(True)
            elif feature_type == 'mfcc':
                mfcc = librosa.feature.mfcc(y=seg.astype(np.float64),
                                             sr=sr, n_mfcc=32)
                librosa.display.specshow(mfcc, x_axis='time', sr=sr,
                                          cmap='coolwarm', ax=ax)
            elif feature_type == 'mfcc_delta':
                mfcc  = librosa.feature.mfcc(y=seg.astype(np.float64),
                                              sr=sr, n_mfcc=32)
                delta = librosa.feature.delta(mfcc)
                librosa.display.specshow(delta, x_axis='time', sr=sr,
                                          cmap='coolwarm', ax=ax)
            elif feature_type == 'stft':
                f, t, Z = stft(seg, fs=sr, nperseg=512, noverlap=384)
                Zdb = 20 * np.log10(np.abs(Z) + 1e-9)
                ax.pcolormesh(t, f, Zdb, shading='gouraud', cmap='magma')
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
            elif feature_type == 'logmel':
                mel = librosa.feature.melspectrogram(
                    y=seg.astype(np.float64), sr=sr, n_mels=64)
                log_mel = librosa.power_to_db(mel, ref=np.max)
                librosa.display.specshow(log_mel, x_axis='time', y_axis='mel',
                                          sr=sr, cmap='viridis', ax=ax)
            elif feature_type == 'envelope':
                from scipy.signal import hilbert
                t = np.arange(len(seg)) / sr
                env = np.abs(hilbert(seg))
                ax.plot(t, seg, color='#9ca3af', linewidth=0.4)
                ax.plot(t, env, color=colors[col], linewidth=1.2)
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")

            if row == 0:
                ax.set_title(CLASS_NAMES[c],
                              color=colors[col], fontweight='bold')

    fig.tight_layout()
    return fig


def plot_probability_bar(weighted_proba, final_class, title="Final prediction"):
    """Show 3-class probability bar with final class highlighted."""
    _set_clean_style()
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    colors = [CLASS_COLORS[c] if c == final_class else '#cbd5e1'
              for c in range(NUM_CLASSES)]
    bars = ax.bar(CLASS_NAMES, weighted_proba * 100,
                   color=colors, edgecolor='white')
    for b, p in zip(bars, weighted_proba):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f"{p*100:.1f}%", ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110); ax.set_ylabel("Probability (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# =============================================================================
# Competition batch grid -- the headline visual for the poster
# =============================================================================
def plot_competition_grid(per_flange_results, flange_top_models,
                           flange_weights, areas=("A1", "A2", "A3", "A4")):
    """
    Replicate the 4 (flange) x 5 (top-N model) panel grid used on the poster.
    Each subplot is a stacked bar chart over the four areas of a flange,
    with stack heights = number of hits each model votes for each torque.

    Parameters
    ----------
    per_flange_results : dict
        flange_id -> list of dicts:
            {'area': str, 'per_hit_preds': {model_name: 1-D np.ndarray of {0,1,2}}}
    flange_top_models : dict
        flange_id -> [list of top-N model names, rank 1 first]
    flange_weights : dict
        flange_id -> {model_name: float}
    """
    _set_clean_style()

    flanges = sorted(per_flange_results.keys())
    if not flanges:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No batch results to display.",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Assume top-N is consistent across flanges; take max
    top_n = max(len(flange_top_models.get(f, [])) for f in flanges)
    if top_n == 0:
        top_n = 5

    fig, axes = plt.subplots(
        len(flanges), top_n,
        figsize=(3.2 * top_n, 3.0 * len(flanges)),
        squeeze=False,
    )

    # Class color order: 0->blue, 1->orange, 2->green per CLASS_COLORS
    class_colors = [CLASS_COLORS[i] for i in range(NUM_CLASSES)]

    # Compute per-flange final ensemble pred for the boxed annotation
    # IMPORTANT: this must match the headline table calculation in
    # pages/2_Competition_Prediction.py exactly. The table uses
    #     mean_proba = np.mean([entry['proba'] for entry in entries], axis=0)
    # which is the simple mean of the per-FILE accuracy-weighted soft
    # probabilities (one entry per uploaded file/area). We do the same
    # here so the boxed Final on the grid never disagrees with the
    # Final per-flange predictions table.
    final_per_flange = {}
    for f in flanges:
        entries = per_flange_results[f]
        if not entries:
            final_per_flange[f] = np.zeros(NUM_CLASSES, dtype=np.float64)
            continue
        # Each entry has a 'proba' field iff the caller wired it in
        # (the new path in 2_Competition_Prediction.py). Fall back to
        # the old per-hit-vote-histogram method if it's missing, so we
        # don't break older callers.
        if all('proba' in e for e in entries):
            stack = np.stack([np.asarray(e['proba'], dtype=np.float64)
                              for e in entries], axis=0)
            final_per_flange[f] = stack.mean(axis=0)
        else:
            # Legacy fallback (kept for backward compat).
            wts = flange_weights.get(f, {})
            acc = np.zeros(NUM_CLASSES, dtype=np.float64)
            wsum = 0.0
            for entry in entries:
                for m, preds in entry['per_hit_preds'].items():
                    if m not in wts or len(preds) == 0:
                        continue
                    hist = np.bincount(preds, minlength=NUM_CLASSES) / len(preds)
                    acc += wts[m] * hist
                    wsum += wts[m]
            if wsum > 0:
                acc = acc / wsum
            final_per_flange[f] = acc

    # Master row/column layout
    for ri, flange in enumerate(flanges):
        tops = flange_top_models.get(flange, [])
        weights = flange_weights.get(flange, {})
        entries = per_flange_results[flange]
        # entries: ordered list of {area, per_hit_preds}
        # Sort by area for predictable column order
        entries_by_area = {e['area']: e for e in entries}
        ordered_areas = [a for a in areas if a in entries_by_area]
        if not ordered_areas:
            ordered_areas = list(entries_by_area.keys())

        for ci in range(top_n):
            ax = axes[ri][ci]
            if ci >= len(tops):
                ax.axis('off')
                continue
            mname = tops[ci]
            w = weights.get(mname, 0.0)

            # Build per-area stacked counts for this model
            counts_per_class = {c: [] for c in range(NUM_CLASSES)}
            n_per_area = []
            for a in ordered_areas:
                preds = entries_by_area[a]['per_hit_preds'].get(mname,
                                                                np.array([]))
                hist = np.bincount(preds.astype(int), minlength=NUM_CLASSES) \
                    if len(preds) else np.zeros(NUM_CLASSES)
                for c in range(NUM_CLASSES):
                    counts_per_class[c].append(int(hist[c]))
                n_per_area.append(int(len(preds)))

            x = np.arange(len(ordered_areas))
            bottoms = np.zeros(len(ordered_areas))
            for c in range(NUM_CLASSES):
                heights = np.array(counts_per_class[c])
                ax.bar(x, heights, bottom=bottoms,
                       color=class_colors[c], edgecolor='white',
                       linewidth=0.5, width=0.7)
                # write count labels in the middle of each segment if big
                # enough
                for xi, h in enumerate(heights):
                    if h >= 3:
                        ax.text(xi, bottoms[xi] + h / 2, str(int(h)),
                                ha='center', va='center', fontsize=10,
                                fontweight='bold', color='white')
                bottoms = bottoms + heights

            # n=XX label on top of each bar
            for xi, n in enumerate(n_per_area):
                ax.text(xi, bottoms[xi] + max(1, max(n_per_area) * 0.04),
                        f"n={n}", ha='center', va='bottom', fontsize=8,
                        color='#374151')

            # Title with rank, model, weight, ensemble decision per area block
            #   -> majority class for this model summed across areas
            total_hist = np.zeros(NUM_CLASSES, dtype=int)
            for c in range(NUM_CLASSES):
                total_hist[c] = sum(counts_per_class[c])
            total_n = total_hist.sum()
            if total_n > 0:
                pred_c = int(np.argmax(total_hist))
                pred_pct = 100 * total_hist[pred_c] / total_n
                pred_str = f"→ {CLASS_NAMES[pred_c]} ({pred_pct:.0f}%)"
                title_color = class_colors[pred_c]
            else:
                pred_str = "→ no data"
                title_color = '#6b7280'

            rank_marker = "★" if ci == 0 else f"#{ci+1}"
            ax.set_title(
                f"{rank_marker}  {mname}   w={w:.2f}\n{pred_str}",
                fontsize=10, color=title_color, fontweight='bold'
            )

            ax.set_xticks(x)
            ax.set_xticklabels([f"{flange}{a}" for a in ordered_areas],
                               fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            top = max(bottoms.max() if len(bottoms) else 1, 1) * 1.25
            ax.set_ylim(0, top)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)

            # Row label on the left-most column
            if ci == 0:
                ax.set_ylabel(f"{flange}   Segments",
                              fontsize=11, fontweight='bold', labelpad=10)

            # On the right-most column, show the final per-flange box
            if ci == top_n - 1:
                proba = final_per_flange[flange]
                final_c = int(np.argmax(proba))
                box_color = class_colors[final_c]
                final_text = (
                    f"Final: {CLASS_NAMES[final_c]}\n"
                    f"P$_0$={proba[0]:.2f}  "
                    f"P$_{{25}}$={proba[1]:.2f}  "
                    f"P$_{{50}}$={proba[2]:.2f}"
                )
                ax.text(
                    0.98, 0.02, final_text,
                    transform=ax.transAxes,
                    ha='right', va='bottom',
                    fontsize=8.5, fontweight='bold', color=box_color,
                    bbox=dict(boxstyle='round,pad=0.45',
                              facecolor='white',
                              edgecolor=box_color, linewidth=1.2,
                              alpha=0.95),
                )

    # Legend at the top
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=class_colors[c], edgecolor='white',
              label=CLASS_NAMES[c])
        for c in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_handles,
               loc='upper center', ncol=NUM_CLASSES,
               bbox_to_anchor=(0.5, 1.005),
               title="Predicted torque class",
               frameon=True, fontsize=10)

    fig.suptitle(
        "Competition Prediction — Accuracy-Weighted Ensemble  "
        "(Top-N per flange, WFAN)\n"
        "Columns = top-N models per flange   |   ★ = rank-1   |   "
        "weight (w) shown in title",
        fontsize=12, fontweight='bold', y=1.045,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig
