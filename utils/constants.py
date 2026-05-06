"""
constants.py
============
Single source of truth for all constants used across the pipeline.
Mirrors Section 1 of the notebook.
"""
import os
import torch

# =============================================================================
# CLASSIFICATION CONSTANTS
# =============================================================================
TORQUE_CLASSES  = {0: 0, 25: 1, 50: 2}
CLASS_NAMES     = ['0 ft-lbs', '25 ft-lbs', '50 ft-lbs']
CLASS_NAMES_BIN = ['Loose (0 ft-lbs)', 'Tight (25 + 50 ft-lbs)']
NUM_CLASSES     = 3
FLANGES = ['F1', 'F2', 'F3', 'F4']
AREAS   = ['A1', 'A2', 'A3', 'A4']

# =============================================================================
# FEATURE DIMENSIONS
# =============================================================================
N_MFCC       = 32
FIXED_FRAMES = 64
MFCC_DIM     = N_MFCC * 4              # 128 (mean + std of MFCC and delta-MFCC)
N_PSD_BANDS  = 80
PSD_F_MIN    = 50.0
PSD_F_MAX    = 24000.0
PHYSICS_DIM  = 13
FEATURE_DIM  = N_PSD_BANDS + MFCC_DIM + PHYSICS_DIM   # 80 + 128 + 13 = 221
SEQ_LEN      = FIXED_FRAMES            # 64 genuine MFCC time steps
INPUT_SIZE   = N_MFCC                  # 32 MFCC coefficients per time step
CNN_CHANNELS = 3                       # MFCC + delta-MFCC + log-Mel spectrogram

# =============================================================================
# AUDIO PROCESSING CONSTANTS
# =============================================================================
PRE_TIME             = 0.02   # seconds BEFORE the detected peak
POST_TIME            = 0.40   # seconds AFTER the peak (full ring-down)
MIN_SEGMENT_DURATION = 0.10   # seconds, skip shorter segments
PEAK_HEIGHT_RATIO    = 0.30   # peak ≥ 30% of file's global maximum
PEAK_MIN_DISTANCE    = 0.50   # min seconds between consecutive peaks

# =============================================================================
# REPRODUCIBILITY & DEVICE
# =============================================================================
SEED       = 42
BATCH_SIZE = 32
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# PATH HELPERS
# =============================================================================
APP_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(APP_DIR, 'artifacts')
ASSETS_DIR    = os.path.join(APP_DIR, 'assets')
SAMPLE_DIR    = os.path.join(APP_DIR, 'sample_data')

# Color palette used throughout the app (consistent with poster)
CLASS_COLORS = {
    0: '#1A5FA8',   # 0 ft-lbs (loose) - blue
    1: '#C45E00',   # 25 ft-lbs - orange
    2: '#0A7A4F',   # 50 ft-lbs - green
}
BLOCK_COLORS = {
    'PSD'    : '#378ADD',
    'MFCC'   : '#BA7517',
    'Physics': '#1D9E75',
}
