"""Main training script for the Hybrid Data-Mining NIDS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import pandas as pd
from config.settings import *
from pipeline.training_pipeline import TrainingPipeline
from utils.data_utils import load_nsl_kdd, convert_to_binary_classification, split_train_test
from utils.file_handler import ModelFileHandler
from utils.logger import get_logger

logger = get_logger(__name__)


def find_dataset_dir() -> Path:
    candidates = [
        DATA_DIR,
        Path.home() / "Downloads" / "data-sets" / "archive" / "nsl-kdd",
        Path.home() / "Downloads" / "nsl-kdd",
    ]
    for path in candidates:
        if (path / "KDDTrain+_20Percent.txt").exists():
            return path
    raise FileNotFoundError(
        "NSL-KDD training file not found. Put KDDTrain+_20Percent.txt and, "
        "preferably, KDDTest+.txt in data/raw/ or your Downloads/nsl-kdd folder."
    )


def build_pipeline():
    model_configs = {
        "xgboost": XGBOOST_CONFIG,
        "lightgbm": LIGHTGBM_CONFIG,
        "random_forest": RANDOM_FOREST_CONFIG,
        "svm": SVM_CONFIG,
        "logistic_regression": LOGISTIC_REGRESSION_CONFIG,
    }
    return TrainingPipeline(
        PREPROCESSING_CONFIG, FEATURE_ENGINEERING_CONFIG, model_configs,
        ENSEMBLE_CONFIG, SIGNATURE_ENGINE_CONFIG, FUSION_CONFIG
    )


def save_artifacts(pipeline: TrainingPipeline):
    handler = ModelFileHandler()
    handler.save_model(pipeline.final_model, MODEL_ARTIFACTS_DIR, "hybrid_nids")
    # Keep the plain ensemble available for research/model-comparison work.
    handler.save_ensemble(pipeline.ensemble, MODEL_ARTIFACTS_DIR, "ensemble_nids")
    for name, model in pipeline.models.items():
        handler.save_model(model.model, MODEL_ARTIFACTS_DIR, f"model_{name.lower()}")
    handler.save_model(pipeline.preprocessor, PIPELINE_ARTIFACTS_DIR, "preprocessor")
    handler.save_model(pipeline.feature_engineer, PIPELINE_ARTIFACTS_DIR, "feature_engineer")
    with open(PIPELINE_ARTIFACTS_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"classification_threshold": pipeline.classification_threshold}, f, indent=2)


def main():
    logger.info("=" * 80)
    logger.info("HYBRID DATA-MINING BASED NETWORK INTRUSION DETECTION SYSTEM")
    logger.info("=" * 80)
    try:
        data_dir = find_dataset_dir()
        train_df = convert_to_binary_classification(load_nsl_kdd(data_dir / "KDDTrain+_20Percent.txt"))
        train_df = train_df.drop(columns=["difficulty"])
        y_full = train_df.pop("label")

        # Training -> validation. KDDTest+ is kept completely untouched as final test.
        X_train, X_val, y_train, y_val = split_train_test(
            train_df, y_full, test_size=VALIDATION_SPLIT,
            random_state=RANDOM_STATE, stratify=True
        )

        test_path = data_dir / "KDDTest+.txt"
        if test_path.exists():
            test_df = convert_to_binary_classification(load_nsl_kdd(test_path)).drop(columns=["difficulty"])
            y_test = test_df.pop("label")
            X_test = test_df
            test_description = "official NSL-KDD KDDTest+"
        else:
            logger.warning("KDDTest+.txt not found; creating a held-out test split from the training file.")
            X_train, X_test, y_train, y_test = split_train_test(
                X_train, y_train, test_size=0.20,
                random_state=RANDOM_STATE, stratify=True
            )
            X_train, X_val, y_train, y_val = split_train_test(
                X_train, y_train, test_size=0.20,
                random_state=RANDOM_STATE, stratify=True
            )
            test_description = "held-out split"

        pipeline = build_pipeline()
        results = pipeline.train(X_train, y_train, X_val, y_val, X_test, y_test)

        logger.info("\nFinal test results (%s):", test_description)
        for name, metrics in results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                logger.info("%s: %s", name, ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        logger.info("Selected validation threshold: %.3f", pipeline.classification_threshold)

        save_artifacts(pipeline)
        logger.info("Artifacts saved in %s and %s", MODEL_ARTIFACTS_DIR, PIPELINE_ARTIFACTS_DIR)
        logger.info("Training completed successfully.")
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
