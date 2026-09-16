"""
Configuration settings for NIDS system
"""
from pathlib import Path
from config.constants import *

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NIDS_ROOT = PROJECT_ROOT
DATA_DIR = NIDS_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = NIDS_ROOT / "data" / "processed"
MODEL_ARTIFACTS_DIR = NIDS_ROOT / MODEL_SAVE_DIR
PIPELINE_ARTIFACTS_DIR = NIDS_ROOT / PIPELINE_SAVE_DIR
LOGS_DIR_PATH = NIDS_ROOT / LOGS_DIR
RESULTS_DIR_PATH = NIDS_ROOT / RESULTS_DIR

# Create directories if they don't exist
for directory in [
    DATA_DIR, PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR,
    PIPELINE_ARTIFACTS_DIR, LOGS_DIR_PATH, RESULTS_DIR_PATH
]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR_PATH / "nids.log"

# Data configuration
DATA_CONFIG = {
    "test_size": TRAIN_TEST_SPLIT,
    "random_state": RANDOM_STATE,
    "stratify": True,  # For imbalanced classification
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "missing_value_strategy": HANDLE_MISSING_STRATEGY,
    "outlier_detection": None,  # Do not discard legitimate attack traffic as statistical outliers
    "outlier_iqr_multiplier": OUTLIER_IQR_MULTIPLIER,
    "scale_features": True,
    "scaling_method": "StandardScaler",  # StandardScaler, MinMaxScaler, RobustScaler
}

# Feature engineering configuration
FEATURE_ENGINEERING_CONFIG = {
    "selection_method": FEATURE_SELECTION_METHOD,
    "k_features": FEATURE_SELECTION_K,
    "create_interaction_features": False,
    "polynomial_features": False,
}

# XGBoost model configuration
XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "scale_pos_weight": 1,  # Adjust for class imbalance
    "eval_metric": "logloss",
}

# LightGBM model configuration
LIGHTGBM_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "scale_pos_weight": 1,
    "verbose": -1,
}

# Random Forest configuration
RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": CLASS_WEIGHT_BALANCED,
}

# Support Vector Machine configuration
SVM_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,  # Required for soft voting
    "random_state": RANDOM_STATE,
}

# Logistic Regression configuration
LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "solver": "lbfgs",
}

# Ensemble voting configuration
ENSEMBLE_CONFIG = {
    "voting": DEFAULT_VOTING,  # "hard" or "soft"
    "weights": None,  # None for equal weights, or list of weights
    "n_jobs": -1,
}

# Signature engine configuration (misuse/rule-mining tier of the hybrid model)
SIGNATURE_ENGINE_CONFIG = {
    "categorical_columns": ["protocol_type", "service", "flag"],
    "binary_columns": ["land", "root_shell", "su_attempted", "is_guest_login"],
    "max_combo_size": 2,        # mine single-column and 2-column combinations
    "min_support": 0.002,       # rule must cover at least 0.2% of training rows
    "min_confidence": 0.98,     # rule must be >=98% attack within its coverage
    "max_rules": 50,            # cap on number of signatures retained
}

# Fusion configuration (combines signature tier + ensemble tier)
FUSION_CONFIG = {
    "fusion_policy": "signature_override",  # signature hit -> Attack, else defer to ensemble
    "use_signature_fusion": True,
    "threshold": 0.5,          # toggle hybrid fusion on/off in TrainingPipeline
}

# Model hyperparameter tuning (GridSearchCV)
HYPERPARAMETER_TUNING = {
    "enabled": False,
    "cv_folds": 5,
    "scoring": "roc_auc",
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "cv_folds": 5,
    "threshold_optimization": True,
}
