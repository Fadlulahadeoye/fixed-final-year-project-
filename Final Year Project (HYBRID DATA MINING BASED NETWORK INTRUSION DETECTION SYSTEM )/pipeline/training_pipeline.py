"""End-to-end, consistent training and inference pipeline for the hybrid NIDS."""
from typing import Dict, Optional
import numpy as np
import pandas as pd
from utils.logger import get_logger
from data.preprocessor import DataPreprocessor
from features.engineer import FeatureEngineer
from models.classifiers import (XGBoostNIDSModel, LightGBMNIDSModel,
    RandomForestNIDSModel, SVMNIDSModel, LogisticRegressionNIDSModel)
from models.ensemble import EnsembleVotingClassifier
from models.signature_engine import SignatureEngine
from models.fusion import SignatureEnsembleFusion
from evaluation.metrics import NIDSEvaluator

logger = get_logger(__name__)


class TrainingPipeline:
    """One canonical pipeline used by training, evaluation and inference."""
    def __init__(self, preprocessing_config: dict, feature_config: dict,
                 model_configs: dict, ensemble_config: dict,
                 signature_config: Optional[dict] = None,
                 fusion_config: Optional[dict] = None):
        self.preprocessing_config = preprocessing_config
        self.feature_config = feature_config
        self.model_configs = model_configs
        self.ensemble_config = ensemble_config
        self.signature_config = signature_config or {}
        self.fusion_config = fusion_config or {"use_signature_fusion": False, "threshold": 0.5}
        self.preprocessor = None
        self.feature_engineer = None
        self.models = {}
        self.ensemble = None
        self.signature_engine = None
        self.fusion_model = None
        self.evaluator = NIDSEvaluator()
        self.results = {}
        self.classification_threshold = float(self.fusion_config.get("threshold", 0.5))
        self.is_trained = False

    @property
    def use_signature_fusion(self):
        return bool(self.fusion_config.get("use_signature_fusion", False))

    @property
    def final_model(self):
        return self.fusion_model if self.use_signature_fusion else self.ensemble

    def _prepare_components(self):
        self.preprocessor = DataPreprocessor(self.preprocessing_config)
        self.feature_engineer = FeatureEngineer(self.feature_config)

    def _create_models(self):
        self.models = {
            "XGBoost": XGBoostNIDSModel(self.model_configs["xgboost"]),
            "LightGBM": LightGBMNIDSModel(self.model_configs["lightgbm"]),
            "RandomForest": RandomForestNIDSModel(self.model_configs["random_forest"]),
            "SVM": SVMNIDSModel(self.model_configs["svm"]),
            "LogisticRegression": LogisticRegressionNIDSModel(self.model_configs["logistic_regression"]),
        }

    def _create_ensemble(self):
        self.ensemble = EnsembleVotingClassifier(
            list(self.models.values()),
            voting=self.ensemble_config.get("voting", "soft"),
            weights=self.ensemble_config.get("weights"),
        )

    def _create_fusion_model(self):
        cfg = dict(self.fusion_config)
        cfg["threshold"] = self.classification_threshold
        self.signature_engine = SignatureEngine(self.signature_config)
        self.fusion_model = SignatureEnsembleFusion(self.signature_engine, self.ensemble, cfg)

    def _transform_for_ml(self, X):
        pre = self.preprocessor.transform(X)
        return self.feature_engineer.transform(pre)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None) -> Dict:
        """Train. Validation is used only for threshold selection; test stays untouched."""
        self._prepare_components()
        X_train_raw = X_train.copy()
        X_train_ml = self.preprocessor.fit_transform(X_train_raw)
        y_train = y_train.loc[X_train_ml.index]
        X_train_raw = X_train_raw.loc[X_train_ml.index]
        X_train_ml = self.feature_engineer.fit_transform(X_train_ml, y_train)

        X_val_ml = self._transform_for_ml(X_val) if X_val is not None else None
        X_test_ml = self._transform_for_ml(X_test) if X_test is not None else None

        self._create_models()
        for name, model in self.models.items():
            model.train(X_train_ml, y_train)
            if X_test_ml is not None and y_test is not None:
                self.results[name] = model.evaluate(X_test_ml, y_test)

        self._create_ensemble()
        # Ensemble reuses the five models already trained above.
        self.ensemble.train(X_train_ml, y_train)

        if self.use_signature_fusion:
            self._create_fusion_model()
            self.fusion_model.train(X_train_raw, y_train, X_train_ml)
        else:
            self.fusion_model = None

        # Select threshold on validation only. Never tune on final test data.
        if X_val is not None and y_val is not None:
            val_raw = X_val
            val_ml = X_val_ml
            val_proba = self._predict_proba_internal(val_raw, val_ml)
            threshold, value = self.evaluator.optimize_threshold(y_val.values, val_proba, metric="f1")
            self.classification_threshold = float(threshold)
            if self.fusion_model is not None:
                self.fusion_model.threshold = self.classification_threshold
            val_pred = self._threshold_predictions(val_proba, val_raw, val_ml)
            self.results["Validation"] = self.evaluator.evaluate(y_val.values, val_pred, val_proba)
            self.results["ValidationThreshold"] = {"threshold": threshold, "f1": value}

        if X_test_ml is not None and y_test is not None:
            test_proba = self._predict_proba_internal(X_test, X_test_ml)
            test_pred = self._threshold_predictions(test_proba, X_test, X_test_ml)
            self.results["HybridFusion" if self.use_signature_fusion else "Ensemble"] = self.evaluator.evaluate(
                y_test.values, test_pred, test_proba
            )
            if self.fusion_model is not None:
                self.results["FusionReport"] = self.fusion_model.get_fusion_report(
                    X_test, y_test, X_test_ml
                )
        self.is_trained = True
        return self.results

    def _predict_proba_internal(self, X_raw, X_ml):
        if self.use_signature_fusion:
            return self.fusion_model.predict_proba(X_raw, X_ml)
        return self.ensemble.predict_proba(X_ml)

    def _threshold_predictions(self, proba, X_raw=None, X_ml=None):
        if self.use_signature_fusion:
            return self.fusion_model.predict(X_raw, X_ml, threshold=self.classification_threshold)
        return (proba[:, 1] >= self.classification_threshold).astype(int)

    def predict(self, X: pd.DataFrame, model_name: str = "Ensemble") -> np.ndarray:
        X_ml = self._transform_for_ml(X)
        if model_name == "Ensemble":
            return self._threshold_predictions(self._predict_proba_internal(X, X_ml), X, X_ml)
        if model_name == "all":
            return self.ensemble.get_model_predictions(X_ml)
        return self.models[model_name].predict(X_ml)

    def predict_proba(self, X: pd.DataFrame, model_name: str = "Ensemble") -> np.ndarray:
        X_ml = self._transform_for_ml(X)
        if model_name == "Ensemble":
            return self._predict_proba_internal(X, X_ml)
        if model_name == "all":
            return self.ensemble.get_model_probabilities(X_ml)
        return self.models[model_name].predict_proba(X_ml)

    def get_summary(self) -> str:
        final_name = "Hybrid Signature + Ensemble" if self.use_signature_fusion else "Ensemble"
        return (f"Training Pipeline Summary\n"
                f"=========================\n"
                f"Final detector: {final_name}\n"
                f"Classification threshold: {self.classification_threshold:.3f}\n"
                f"Models: {list(self.models.keys())}\n\n"
                f"{self.ensemble.get_ensemble_summary() if self.ensemble else 'Not trained'}")
