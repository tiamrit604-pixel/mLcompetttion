"""
audio_io.py
===========
Audio loading and segmentation utilities.
Mirrors Sections 3, 5 and 6 of the notebook (audioread loader,
filename parser, peak-based segmentation).
"""
import os
import numpy as np
import audioread
from scipy.signal import find_peaks

from .constants import (
    TORQUE_CLASSES, FLANGES, AREAS,
    PRE_TIME, POST_TIME, MIN_SEGMENT_DURATION,
    PEAK_HEIGHT_RATIO, PEAK_MIN_DISTANCE,
)


def audioread_mono_load(filepath):
    """
    Load .m4a or .wav as mono float32 array, stereo-safe.

    Why audioread instead of librosa.load?
    librosa.load can silently resample .m4a files on some FFmpeg builds.
    audioread gives the native sample rate, which is critical because all
    feature ranges (PSD bands, MFCC, etc.) were tuned for the iPhone's
    native 48 kHz.
    """
    audio_blocks = []
    with audioread.audio_open(filepath) as f:
        sr         = f.samplerate
        n_channels = getattr(f, 'channels', 1)
        for buf in f:
            block = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
            if n_channels > 1:
                block = block.reshape(-1, n_channels).mean(axis=1)
            audio_blocks.append(block)
    audio = np.concatenate(audio_blocks) / 32768.0
    return audio, sr


def parse_flange_label(filename):
    """
    Parse a labelled flange filename → (torque, flange, area, label) or None.

    Expected format:  {torque}ftlb{Fx}{Ay}.m4a
    Examples: 0ftlbF1A1.m4a, 25ftlbF2A3.m4a, 50ftlbF4A4.m4a
    Case-insensitive on 'ftlb'; longest-first matching prevents 25→2 collisions.
    """
    name = os.path.splitext(os.path.basename(filename))[0].strip()
    for torque in [50, 25, 0]:               # longest-first
        prefix = f"{torque}ftlb"
        if name.lower().startswith(prefix.lower()):
            rest = name[len(prefix):]
            if len(rest) < 4:
                return None
            flange = rest[:2].upper()
            area   = rest[2:4].upper()
            if flange not in FLANGES or area not in AREAS:
                return None
            return (torque, flange, area, TORQUE_CLASSES[torque])
    return None


def detect_peaks(signal, sr,
                 height_ratio=PEAK_HEIGHT_RATIO,
                 min_distance_s=PEAK_MIN_DISTANCE):
    """
    Detect hammer-strike peaks in the rectified signal.

    Two thresholds prevent both noise spikes and double-triggering:
      - height: ≥ 30% of the file's global maximum
      - distance: ≥ 0.5 s between consecutive peaks
    """
    peaks, _ = find_peaks(
        np.abs(signal),
        height   = height_ratio * np.max(np.abs(signal)),
        distance = int(min_distance_s * sr),
    )
    return peaks


def segment_around_peaks(signal, peaks, sr,
                         pre_time=PRE_TIME,
                         post_time=POST_TIME,
                         min_duration=MIN_SEGMENT_DURATION):
    """
    Cut [peak - pre_time, peak + post_time] segments around each peak.

    Segments shorter than min_duration are dropped (boundary clipping
    near the start or end of the file can produce stubs that have no
    meaningful decay information).
    """
    pre_s  = int(pre_time  * sr)
    post_s = int(post_time * sr)
    min_s  = int(min_duration * sr)

    segments = []
    for peak in peaks:
        start = max(0, peak - pre_s)
        end   = min(len(signal), peak + post_s)
        seg   = signal[start:end].copy()
        if len(seg) >= min_s:
            segments.append(seg)
    return segments


def load_and_segment(filepath):
    """
    Convenience wrapper: load file → detect peaks → segment.

    Returns
    -------
    segments : list of np.ndarray
    sr       : int (sample rate, expected 48000)
    n_peaks  : int (peaks found before duration filtering)
    """
    signal, sr = audioread_mono_load(filepath)
    peaks      = detect_peaks(signal, sr)
    segments   = segment_around_peaks(signal, peaks, sr)
    return segments, sr, len(peaks), signal
