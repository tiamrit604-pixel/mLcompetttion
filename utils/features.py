"""
features.py
===========
Feature extraction.  Three complementary representations:

    1. Flat 221-d vector  : 80 log-PSD bands | 128 MFCC stats | 13 physics
    2. CNN 3-channel image: (3, 32, 64) -- MFCC | delta-MFCC | log-Mel
    3. BiLSTM sequence    : (64, 32) -- 64 time frames x 32 MFCC coefficients

Mirrors Section 4 of the notebook exactly so that features computed here
are bit-identical to those used to train the saved models.
"""
import numpy as np
import librosa
from scipy.signal import find_peaks, welch
from scipy.signal import hilbert

from .constants import (
    N_MFCC, FIXED_FRAMES, N_PSD_BANDS, PSD_F_MIN, PSD_F_MAX,
    PHYSICS_DIM, FEATURE_DIM, SEQ_LEN, INPUT_SIZE, CNN_CHANNELS,
)

# Pre-computed log-spaced PSD band edges
_PSD_BAND_EDGES = np.logspace(
    np.log10(PSD_F_MIN), np.log10(PSD_F_MAX), N_PSD_BANDS + 1
)


# =============================================================================
# 13-d physics-motivated feature vector
# =============================================================================
def extract_physics_features(x, sr):
    """
    13-d physics-motivated feature vector encoding the acoustic
    consequences of bolt looseness via damping and resonance.

    Layout
    ------
        [0]    damping_rate         slope of log-envelope after peak
        [1]    t_half               time to half-peak amplitude
        [2]    ring_energy          area under envelope after peak
        [3]    lh_ratio             log low/high spectral energy ratio
        [4]    spec_centroid        spectral centroid (Hz)
        [5..8] peak_freq_0..3       top-4 resonant peak frequencies (norm)
        [9..12] peak_amp_0..3       top-4 resonant peak amplitudes (norm)
    """
    envelope  = np.abs(hilbert(x))
    peak_idx  = int(np.argmax(envelope))
    t         = np.arange(len(x)) / sr

    # 1. Damping rate (slope of log-envelope after peak)
    decay_len = min(int(0.15 * sr), len(envelope) - peak_idx - 1)
    if decay_len > 10:
        decay_env = envelope[peak_idx : peak_idx + decay_len]
        decay_t   = np.arange(decay_len) / sr
        poly         = np.polyfit(decay_t, np.log(decay_env + 1e-8), 1)
        damping_rate = poly[0]
    else:
        damping_rate = 0.0

    # 2. Half-power time
    peak_amp = envelope[peak_idx]
    after    = envelope[peak_idx:]
    below    = np.where(after < 0.5 * peak_amp)[0]
    t_half   = float(below[0]) / sr if len(below) > 0 else float(len(after)) / sr

    # 3. Ring-down energy
    ring_energy = float(np.trapz(envelope[peak_idx:],
                                  t[peak_idx : peak_idx + len(envelope) - peak_idx]))

    # 4 & 5. Low-to-high energy ratio and spectral centroid
    f_w, pxx_w = welch(x, fs=sr, nperseg=min(2048, len(x)), nfft=4096)
    low_energy   = np.sum(pxx_w[f_w <  3000])
    high_energy  = np.sum(pxx_w[(f_w >= 3000) & (f_w < 15000)])
    lh_ratio     = np.log((low_energy + 1e-12) / (high_energy + 1e-12))
    spec_centroid = np.sum(f_w * pxx_w) / (np.sum(pxx_w) + 1e-12)

    # 6. Top-4 resonant peaks
    peaks_idx, _ = find_peaks(pxx_w, height=np.percentile(pxx_w, 85), distance=15)
    if len(peaks_idx) > 0:
        order     = np.argsort(pxx_w[peaks_idx])[::-1]
        peaks_idx = peaks_idx[order][:4]
        top_freqs = f_w[peaks_idx]
        top_amps  = pxx_w[peaks_idx]
    else:
        top_freqs = np.zeros(4)
        top_amps  = np.zeros(4)
    top_freqs = np.pad(top_freqs, (0, 4 - len(top_freqs)))
    top_amps  = np.pad(top_amps,  (0, 4 - len(top_amps)))
    top_freqs_norm = top_freqs / (sr / 2.0)
    top_amps_norm  = top_amps / (top_amps.sum() + 1e-12)

    return np.array([damping_rate, t_half, ring_energy,
                     lh_ratio, spec_centroid,
                     *top_freqs_norm, *top_amps_norm], dtype=np.float64)


# =============================================================================
# 221-d flat feature vector
# =============================================================================
def extract_features(segment, sr):
    """
    221-d flat feature vector from one percussion segment.

    Layout
    ------
        [0:80]    log-PSD over 80 log-spaced bands (50 Hz – 24 kHz)
        [80:208]  MFCC + delta-MFCC mean/std (32 coeffs each = 128-d)
        [208:221] 13 physics-motivated features
    """
    x = segment.astype(np.float64)

    # ---- BLOCK 1: Log-PSD in 80 log-spaced bands ----------------------------
    f_welch, pxx = welch(x, fs=sr, window='hann', nperseg=2048,
                         noverlap=1024, nfft=4096, scaling='density')
    psd_features = np.zeros(N_PSD_BANDS)
    for i in range(N_PSD_BANDS):
        lo, hi = _PSD_BAND_EDGES[i], _PSD_BAND_EDGES[i + 1]
        mask = (f_welch >= lo) & (f_welch < hi)
        psd_features[i] = np.log(pxx[mask].mean() + 1e-10) if mask.any() else -23.0

    # ---- BLOCK 2: MFCC statistics ------------------------------------------
    mfcc_matrix = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc  = librosa.feature.delta(mfcc_matrix)
    mfcc_mean  = mfcc_matrix.mean(axis=1)
    mfcc_std   = mfcc_matrix.std(axis=1)
    delta_mean = delta_mfcc.mean(axis=1)
    delta_std  = delta_mfcc.std(axis=1)

    # ---- BLOCK 3: Physics features -----------------------------------------
    phys = extract_physics_features(x, sr)

    return np.concatenate([psd_features, mfcc_mean, mfcc_std,
                           delta_mean, delta_std, phys])


# =============================================================================
# CNN: 3-channel 2-D image (MFCC | delta-MFCC | log-Mel spectrogram)
# =============================================================================
def extract_mfcc_2d(segment, sr, n_mfcc=N_MFCC, fixed_frames=FIXED_FRAMES):
    """3-channel CNN image. Shape: (3, 32, 64)."""
    x = segment.astype(np.float64)

    def pad_or_trim(m):
        if m.shape[1] >= fixed_frames:
            return m[:, :fixed_frames]
        return np.pad(m, ((0, 0), (0, fixed_frames - m.shape[1])))

    mfcc     = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    delta    = librosa.feature.delta(mfcc)
    mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mfcc)
    log_mel  = librosa.power_to_db(mel_spec, ref=np.max)

    return np.stack([
        pad_or_trim(mfcc).astype(np.float32),
        pad_or_trim(delta).astype(np.float32),
        pad_or_trim(log_mel).astype(np.float32),
    ], axis=0)


# =============================================================================
# BiLSTM: genuine MFCC time-series (64 frames, 32 coefficients)
# =============================================================================
def extract_mfcc_sequence(segment, sr, n_mfcc=N_MFCC, fixed_frames=FIXED_FRAMES):
    """
    Genuine MFCC time-series for BiLSTM: (fixed_frames, n_mfcc) = (64, 32).

    Per-coefficient z-score normalisation prevents LSTM gate saturation
    (raw MFCC values ~-400 to +200 cause vanishing gradients at epoch 1).
    """
    x    = segment.astype(np.float64)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] >= fixed_frames:
        mfcc = mfcc[:, :fixed_frames]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, fixed_frames - mfcc.shape[1])))
    mean = mfcc.mean(axis=1, keepdims=True)
    std  = mfcc.std(axis=1,  keepdims=True) + 1e-8
    mfcc = (mfcc - mean) / std
    return mfcc.T.astype(np.float32)


def extract_all_features(segment, sr):
    """One-stop convenience: returns (flat_221, image_3x32x64, seq_64x32)."""
    return (
        extract_features(segment, sr),
        extract_mfcc_2d(segment, sr),
        extract_mfcc_sequence(segment, sr),
    )
