"""
Global constants for NIDS system
"""

# Class labels for binary classification
BINARY_LABELS = {
    0: "Normal",
    1: "Attack"
}

# Reverse mapping
BINARY_LABELS_REVERSE = {v: k for k, v in BINARY_LABELS.items()}

# Random seed for reproducibility
RANDOM_STATE = 42

# Train-test split ratio
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2

# Feature scaling bounds
FEATURE_SCALE_MIN = 0.0
FEATURE_SCALE_MAX = 1.0

# Ensemble voting strategies
VOTING_HARD = "hard"
VOTING_SOFT = "soft"

# Default voting strategy
DEFAULT_VOTING = "soft"

# Class weights for imbalanced data
CLASS_WEIGHT_BALANCED = "balanced"

# Evaluation metrics thresholds
ROC_AUC_THRESHOLD = 0.7
PRECISION_THRESHOLD = 0.8
RECALL_THRESHOLD = 0.8

# Feature selection parameters
FEATURE_SELECTION_K = "all"  # Can be int or "all"
FEATURE_SELECTION_METHOD = "SelectKBest"

# Preprocessing parameters
HANDLE_MISSING_STRATEGY = "mean"  # mean, median, drop
OUTLIER_DETECTION_METHOD = "iqr"  # iqr, zscore
OUTLIER_IQR_MULTIPLIER = 1.5

# Model artifact paths (relative to project root)
MODEL_SAVE_DIR = "models/artifacts"
PIPELINE_SAVE_DIR = "pipelines/artifacts"
LOGS_DIR = "logs"
RESULTS_DIR = "results"
