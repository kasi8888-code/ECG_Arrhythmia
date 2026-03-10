"""
Configuration settings for ECG Arrhythmia Detection System
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories
for dir_path in [DATA_DIR, MODEL_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA SETTINGS
# =============================================================================
SAMPLING_RATE = 125  # Hz (MIT-BIH is 360Hz, but Kaggle version is resampled to 125Hz)
BEAT_LENGTH = 187    # samples per beat in Kaggle dataset

# Class mapping: Original MIT-BIH to 5-class AAMI standard
# N: Normal, S: Supraventricular, V: Ventricular, F: Fusion, Q: Unknown
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
NUM_CLASSES = 5

# Patient-wise split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# MODEL SETTINGS
# =============================================================================
# CNN Architecture
CNN_FILTERS = [32, 64, 128, 256]
CNN_KERNEL_SIZE = 5
CNN_POOL_SIZE = 2
DROPOUT_RATE = 0.3

# Hybrid fusion
NUM_ENGINEERED_FEATURES = 25  # HRV + morphological features
FUSION_HIDDEN_DIM = 128

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-4

# Class weights for imbalanced data (will be computed from data)
USE_CLASS_WEIGHTS = True

# =============================================================================
# CONFIDENCE & REFERRAL SETTINGS
# =============================================================================
# Confidence threshold for referral (predictions below this are flagged for human review)
REFERRAL_THRESHOLD = 0.7

# Temperature scaling for calibration
CALIBRATION_TEMPERATURE = 1.5  # Will be optimized on validation set

# =============================================================================
# GRAD-CAM SETTINGS
# =============================================================================
TARGET_LAYER_NAME = "conv_block_3"  # Layer for Grad-CAM visualization

# =============================================================================
# RANDOM SEED
# =============================================================================
RANDOM_SEED = 42
