"""Decision fusion for the hybrid NIDS.

Tier 1 (signature misuse detection) sees RAW NSL-KDD values so rules such as
flag='S0' remain meaningful. Tier 2 (ML ensemble) sees the fitted numerical
preprocessing/feature-engineering representation. A signature hit overrides
the ensemble; otherwise the ensemble probability and configured threshold
decide the verdict.
"""
from typing import Optional
import numpy as np
import pandas as pd
from models.base_model import BaseNIDSModel
from models.signature_engine import SignatureEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class SignatureEnsembleFusion(BaseNIDSModel):
    """Two-tier hybrid detector: raw signatures + ML ensemble."""
    def __init__(self, signature_engine: SignatureEngine,
                 ensemble: BaseNIDSModel, config: Optional[dict] = None):
        self.signature_engine = signature_engine
        self.ensemble = ensemble
        self.threshold = float((config or {}).get("threshold", 0.5))
        super().__init__("SignatureEnsembleFusion", None, config or {"fusion_policy": "signature_override"})

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_ensemble_train: Optional[pd.DataFrame] = None):
        """Fit signatures on RAW data and the ensemble on processed data."""
        X_ensemble_train = X_train if X_ensemble_train is None else X_ensemble_train
        self.signature_engine.fit(X_train, y_train)
        self.ensemble.train(X_ensemble_train, y_train)
        self.is_trained = True
        return self

    def _signature_and_ensemble(self, X_signature, X_ensemble=None):
        X_ensemble = X_signature if X_ensemble is None else X_ensemble
        return self.signature_engine.match_mask(X_signature), self.ensemble.predict_proba(X_ensemble)

    def predict_proba(self, X: pd.DataFrame, X_ensemble: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Fusion model not trained. Call train() first.")
        signature_mask, ensemble_proba = self._signature_and_ensemble(X, X_ensemble)
        fused = ensemble_proba.copy()
        fused[signature_mask] = [0.01, 0.99]
        return fused

    def predict(self, X: pd.DataFrame, X_ensemble: Optional[pd.DataFrame] = None,
                threshold: Optional[float] = None) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Fusion model not trained. Call train() first.")
        threshold = self.threshold if threshold is None else float(threshold)
        signature_mask, ensemble_proba = self._signature_and_ensemble(X, X_ensemble)
        ensemble_pred = (ensemble_proba[:, 1] >= threshold).astype(int)
        return np.where(signature_mask, 1, ensemble_pred)

    def get_fusion_report(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                          X_ensemble: Optional[pd.DataFrame] = None) -> dict:
        signature_mask, ensemble_proba = self._signature_and_ensemble(X, X_ensemble)
        final_pred = np.where(signature_mask, 1,
                               (ensemble_proba[:, 1] >= self.threshold).astype(int))
        report = {
            "n_records": len(X),
            "n_signature_hits": int(signature_mask.sum()),
            "signature_hit_rate": float(signature_mask.mean()) if len(X) else 0.0,
            "n_flagged_attack_total": int(final_pred.sum()),
            "n_flagged_by_ensemble_only": int(np.sum((~signature_mask) & (final_pred == 1))),
            "threshold": self.threshold,
        }
        if y is not None:
            y = np.asarray(y)
            attack_idx = y == 1
            report["recall_within_signature_hits"] = float(signature_mask[attack_idx].mean()) if attack_idx.any() else 0.0
            report["overall_accuracy"] = float((final_pred == y).mean())
        return report

    def get_ensemble_summary(self) -> str:
        return (f"\nHybrid Signature + Ensemble Fusion\n"
                f"===================================\n"
                f"Fusion policy: {self.config.get('fusion_policy', 'signature_override')}\n"
                f"Threshold: {self.threshold:.3f}\n"
                f"Signature tier: {len(self.signature_engine.rules)} mined signature(s)\n"
                + (self.ensemble.get_ensemble_summary() if hasattr(self.ensemble, "get_ensemble_summary") else ""))
