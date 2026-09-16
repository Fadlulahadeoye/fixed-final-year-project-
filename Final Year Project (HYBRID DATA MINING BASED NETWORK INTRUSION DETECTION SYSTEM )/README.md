# Hybrid Network Intrusion Detection System (NIDS)

A production-grade, modular network intrusion detection system using hybrid data mining with sequential pipeline and ensemble voting architecture.

## Features

- **Sequential Pipeline**: Preprocessing → Feature Engineering → Ensemble Voting
- **Hybrid Models**: XGBoost, LightGBM, Random Forest, SVM, Logistic Regression
- **Ensemble Voting**: Hard/Soft voting with optional weighted predictions
- **Binary Classification**: Normal vs. Attack (easily scalable to multi-class)
- **Production-Grade**: Configuration-driven, comprehensive logging, model serialization
- **Modular Design**: Each component independent and reusable

## Project Structure

```
nids-hybrid/
├── config/
│   ├── constants.py              # Magic numbers, class labels
│   ├── settings.py               # Paths, hyperparameters, configurations
│   └── __init__.py
├── data/
│   ├── loader.py                 # Dataset loading & EDA
│   ├── preprocessor.py           # Data cleaning, normalization
│   └── __init__.py
├── features/
│   ├── engineer.py               # Feature extraction & selection
│   ├── multiclass.py             # Multi-class extensions
│   └── __init__.py
├── models/
│   ├── base_model.py             # Abstract base class
│   ├── classifiers.py            # Individual models (XGB, LGB, etc.)
│   ├── ensemble.py               # Ensemble voting classifier
│   └── __init__.py
├── pipeline/
│   ├── sequential_pipeline.py    # Preprocessing + feature pipeline
│   ├── training_pipeline.py      # End-to-end training orchestration
│   └── __init__.py
├── evaluation/
│   ├── metrics.py                # ROC-AUC, Precision, Recall, F1, etc.
│   ├── visualization.py          # ROC curves, confusion matrices
│   └── __init__.py
├── utils/
│   ├── logger.py                 # Logging configuration
│   ├── file_handler.py           # Model saving/loading
│   ├── data_utils.py             # Train-test split, data loading
│   └── __init__.py
├── tests/
│   ├── test_preprocessing.py     # Preprocessing unit tests
│   ├── test_features.py          # Feature engineering tests
│   └── test_ensemble.py          # Ensemble voting tests
├── main.py                       # Main training script
├── predict.py                    # Inference script
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.10+
- pip (Python package manager)

### Install Dependencies

```bash
cd nids-hybrid
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
```

## Quick Start

### 1. Training

```bash
python main.py
```

This will:
- Load NSL-KDD dataset (binary classification)
- Preprocess and engineer features
- Train all 5 models
- Create ensemble with soft voting
- Evaluate on test set
- Save artifacts to `models/artifacts/` and `pipelines/artifacts/`

### 2. Inference

```bash
python predict.py
```

This will:
- Load trained ensemble and pipelines
- Make predictions on sample data
- Show prediction confidence scores

### 3. Custom Training

```python
from pipeline.training_pipeline import TrainingPipeline
from config.settings import *

# Create training pipeline
training_pipeline = TrainingPipeline(
    preprocessing_config=PREPROCESSING_CONFIG,
    feature_config=FEATURE_ENGINEERING_CONFIG,
    model_configs={
        "xgboost": XGBOOST_CONFIG,
        "lightgbm": LIGHTGBM_CONFIG,
        # ... other models
    },
    ensemble_config=ENSEMBLE_CONFIG
)

# Train
results = training_pipeline.train(X_train, y_train, X_test, y_test)

# Predict
y_pred = training_pipeline.predict(X_test)
y_proba = training_pipeline.predict_proba(X_test)
```

## Configuration

### Preprocessing (`config/settings.py`)

```python
PREPROCESSING_CONFIG = {
    "missing_value_strategy": "mean",      # mean, median, drop
    "outlier_detection": "iqr",            # iqr, zscore
    "outlier_iqr_multiplier": 1.5,
    "scale_features": True,
    "scaling_method": "StandardScaler",    # StandardScaler, MinMaxScaler
}
```

### Feature Engineering

```python
FEATURE_ENGINEERING_CONFIG = {
    "selection_method": "SelectKBest",
    "k_features": "all",                   # int or "all"
    "create_interaction_features": False,
    "polynomial_features": False,
}
```

### Models

Each model has configurable hyperparameters:

```python
XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
```

### Ensemble

```python
ENSEMBLE_CONFIG = {
    "voting": "soft",                      # "hard" or "soft"
    "weights": None,                       # None for equal, or list of weights
    "n_jobs": -1,
}
```

## Data Format

### Expected Features
NSL-KDD dataset with 41 features:
- `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`
- `land`, `wrong_fragment`, `urgent`, `hot`, `num_failed_logins`, `logged_in`
- `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`
- `num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`
- `is_guest_login`, `count`, `srv_count`, `serror_rate`, `srv_serror_rate`
- `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`
- `srv_diff_host_rate`, `dst_host_count`, `dst_host_srv_count`
- `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`
- `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`
- `dst_host_rerror_rate`, `dst_host_srv_rerror_rate`, `label`, `difficulty`

### Target Variable
- **Binary**: `normal` (0) vs. `attack` (1)
- **Multi-class**: Specific attack type (adaptable)

## Models

### Individual Classifiers

1. **XGBoost** - Gradient Boosting
   - Fast training, excellent for tabular data
   - Feature importance available

2. **LightGBM** - Light Gradient Boosting
   - Memory efficient, fast inference
   - Handles large datasets well

3. **Random Forest** - Ensemble Trees
   - Robust, interpretable
   - Good baseline model

4. **SVM** - Support Vector Machine
   - Non-linear classification
   - Handles high-dimensional data

5. **Logistic Regression** - Linear Baseline
   - Interpretable, fast
   - Good for baseline comparison

### Ensemble Strategy

The ensemble combines predictions using:

- **Hard Voting**: Majority vote among models
- **Soft Voting**: Average of predicted probabilities (recommended)
- **Weighted Voting**: Weighted average based on model importance

## Evaluation Metrics

### Binary Classification

- **Accuracy**: Overall correctness
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: TP, TN, FP, FN breakdown

### Visualizations

- ROC curves for each model
- Precision-recall curves
- Confusion matrices
- Feature importance
- Class distribution

## Multi-Class Support

For multi-class attack type classification:

```python
from features.multiclass import MultiClassNIDS

multi_nids = MultiClassNIDS()
y_encoded, label_mapping = multi_nids.prepare_multiclass_labels(y)

# Handle class imbalance
X_balanced, y_balanced = multi_nids.handle_class_imbalance(X, y_encoded)

# Get per-class metrics
metrics = multi_nids.get_per_class_metrics(y_test, y_pred)
```

## Logging

All operations are logged to:
- **Console**: Real-time display
- **File**: `logs/nids.log`

Set log level in `config/settings.py`:
```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
```

## Model Artifacts

After training, artifacts are saved to:

```
nids-hybrid/
├── models/artifacts/
│   ├── ensemble_nids.joblib
│   ├── model_xgboost.joblib
│   ├── model_lightgbm.joblib
│   └── ...
└── pipelines/artifacts/
    ├── preprocessor.joblib
    └── feature_engineer.joblib
```

Load and use:
```python
from utils.file_handler import ModelFileHandler

handler = ModelFileHandler()
ensemble = handler.load_ensemble("models/artifacts/ensemble_nids.joblib")
preprocessor = handler.load_model("pipelines/artifacts/preprocessor.joblib")
```

## Performance

### Expected Results (NSL-KDD 20% Binary)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| XGBoost | 0.96+ | 0.95+ | 0.96+ | 0.96 | 0.99 |
| LightGBM | 0.96+ | 0.95+ | 0.96+ | 0.96 | 0.99 |
| Random Forest | 0.95+ | 0.94+ | 0.95+ | 0.95 | 0.98 |
| **Ensemble** | **0.97+** | **0.96+** | **0.97+** | **0.97** | **0.99+** |

*Results may vary with hyperparameter tuning*

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Test coverage:
- `test_preprocessing.py`: Data preprocessing (8 tests)
- `test_features.py`: Feature engineering (5 tests)
- `test_ensemble.py`: Ensemble voting (6 tests)

## Troubleshooting

### Memory Issues
- Reduce `n_estimators` in model configs
- Use `LightGBM` instead of `XGBoost` (more memory efficient)
- Enable feature selection (`k_features` in config)

### Slow Training
- Use `LightGBM` for faster training
- Reduce dataset size (use 20% sample)
- Enable parallel processing: set `n_jobs=-1` in configs

### Poor Performance
- Check data quality (missing values, outliers)
- Adjust class weight for imbalanced data
- Tune hyperparameters (use GridSearchCV)
- Add more features or engineer new ones

## Advanced Usage

### Custom Data Loading

```python
from data.loader import DataLoader

loader = DataLoader("path/to/dataset")
df = loader.load_dataset("my_data.csv")
stats = loader.explore_dataset(target_col="label")
```

### Feature Importance Analysis

```python
from features.engineer import FeatureEngineer

engineer = FeatureEngineer(config)
importance = engineer.get_feature_importance(X, y, method="f_classif")

# Top 10 features
sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
```

### Threshold Optimization

```python
from evaluation.metrics import NIDSEvaluator

evaluator = NIDSEvaluator()
optimal_threshold, best_value = evaluator.optimize_threshold(
    y_test, y_pred_proba, metric="f1"
)
```

## Future Enhancements

- [ ] Distributed training with Spark
- [ ] Deep learning models (LSTM, Autoencoder)
- [ ] Real-time streaming detection
- [ ] Web dashboard for monitoring
- [ ] Automated hyperparameter tuning
- [ ] Explainability analysis (SHAP)

## References

### Datasets
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html

### Papers
- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- Ensemble Learning: Schapire & Freund (2012)

## License

MIT License

## Author

Hybrid NIDS Development Team

## Support

For issues and questions, refer to the documentation or check logs in `logs/nids.log`.

---

**Last Updated**: 2026-05-30
**Version**: 1.0.0
