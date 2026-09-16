"""Inference for the complete hybrid NIDS detector."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from config.settings import MODEL_ARTIFACTS_DIR, PIPELINE_ARTIFACTS_DIR, DATA_DIR
from utils.file_handler import ModelFileHandler
from utils.logger import get_logger

logger = get_logger(__name__)


class NIDSInference:
    """Loads the same hybrid artifact produced by main.py."""
    def __init__(self, hybrid_path=None, preprocessor_path=None, feature_engineer_path=None):
        handler = ModelFileHandler()
        self.hybrid = handler.load_model(hybrid_path or (MODEL_ARTIFACTS_DIR / "hybrid_nids.joblib"))
        self.preprocessor = handler.load_model(preprocessor_path or (PIPELINE_ARTIFACTS_DIR / "preprocessor.joblib"))
        self.feature_engineer = handler.load_model(feature_engineer_path or (PIPELINE_ARTIFACTS_DIR / "feature_engineer.joblib"))
        threshold_file = PIPELINE_ARTIFACTS_DIR / "threshold.json"
        self.threshold = 0.5
        if threshold_file.exists():
            self.threshold = float(json.loads(threshold_file.read_text(encoding="utf-8")).get("classification_threshold", 0.5))
        logger.info("Hybrid inference engine loaded; threshold=%.3f", self.threshold)

    def _prepare(self, X):
        X_ml = self.feature_engineer.transform(self.preprocessor.transform(X))
        return X_ml

    def predict_proba(self, X):
        X_ml = self._prepare(X)
        return self.hybrid.predict_proba(X, X_ml)

    def predict(self, X, threshold=None):
        threshold = self.threshold if threshold is None else float(threshold)
        X_ml = self._prepare(X)
        return self.hybrid.predict(X, X_ml, threshold=threshold)

    def predict_with_confidence(self, X, threshold=None):
        proba = self.predict_proba(X)
        pred = self.predict(X, threshold)
        return {
            "predictions": pred,
            "confidence": np.max(proba, axis=1),
            "probabilities": proba,
            "labels": ["Attack" if p else "Normal" for p in pred],
        }


if __name__ == "__main__":
    from utils.data_utils import load_nsl_kdd, convert_to_binary_classification
    test_path = DATA_DIR / "KDDTest+.txt"
    if not test_path.exists():
        raise FileNotFoundError("Put KDDTest+.txt in data/raw/ before running predict.py")
    df = convert_to_binary_classification(load_nsl_kdd(test_path))
    y = df.pop("label")
    df = df.drop(columns=["difficulty"])
    engine = NIDSInference()
    result = engine.predict_with_confidence(df.iloc[:100])
    logger.info("Predicted attacks: %d / %d", int((result["predictions"] == 1).sum()), len(df.iloc[:100]))
