"""
Unit tests for the signature engine and signature+ensemble fusion layer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from models.signature_engine import SignatureEngine
from models.fusion import SignatureEnsembleFusion
from models.base_model import BaseNIDSModel


def _make_synthetic_dataset(n=2000, seed=42):
    """
    Synthetic dataset where:
      - flag == 'S0' and land == 1 are perfect, learnable "signatures"
      - a handful of attacks have no categorical signature at all, so a
        signature-only detector must miss them (that's the anomaly
        engine's job in the real system)
    """
    rng = np.random.default_rng(seed)
    protocol = rng.choice(["tcp", "udp", "icmp"], size=n, p=[0.6, 0.3, 0.1])
    service = rng.choice(["http", "ftp", "private", "smtp"], size=n)
    flag = rng.choice(["SF", "S0", "REJ"], size=n, p=[0.7, 0.2, 0.1])
    land = rng.choice([0, 1], size=n, p=[0.995, 0.005])

    y = np.zeros(n, dtype=int)
    y[flag == "S0"] = 1
    y[land == 1] = 1
    unknown_idx = rng.choice(np.where(y == 0)[0], size=50, replace=False)
    y[unknown_idx] = 1

    X = pd.DataFrame({
        "protocol_type": protocol, "service": service,
        "flag": flag, "land": land,
    })
    return X, pd.Series(y)


class _DummyEnsemble(BaseNIDSModel):
    """Stand-in anomaly engine that always predicts 'Normal', so tests can
    isolate what the signature tier and fusion override logic are doing."""

    def __init__(self):
        super().__init__("DummyEnsemble", None, {})

    def train(self, X, y):
        self.is_trained = True

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class TestSignatureEngine:
    """Test cases for SignatureEngine"""

    @pytest.fixture
    def data(self):
        X, y = _make_synthetic_dataset()
        return X.iloc[:1500], y.iloc[:1500], X.iloc[1500:], y.iloc[1500:]

    def test_fit_mines_expected_signatures(self, data):
        X_train, y_train, _, _ = data
        engine = SignatureEngine({"min_support": 0.005, "min_confidence": 0.9})
        engine.fit(X_train, y_train)

        assert engine.is_fitted
        mined_conditions = [r.conditions for r in engine.rules]
        assert {"flag": "S0"} in mined_conditions
        assert {"land": 1} in mined_conditions

    def test_rules_have_high_precision(self, data):
        X_train, y_train, X_test, y_test = data
        engine = SignatureEngine({"min_support": 0.005, "min_confidence": 0.9})
        engine.fit(X_train, y_train)

        mask = engine.match_mask(X_test)
        assert mask.sum() > 0
        # Every signature hit should truly be an attack (precision == 1.0
        # by construction of the synthetic data and min_confidence=0.9)
        assert y_test.values[mask].mean() == 1.0

    def test_does_not_catch_unsignatured_attacks(self, data):
        """A pure signature engine must NOT achieve full recall — the
        unknown-pattern attacks should slip through, which is exactly why
        it's paired with an anomaly engine rather than used alone."""
        X_train, y_train, X_test, y_test = data
        engine = SignatureEngine({"min_support": 0.005, "min_confidence": 0.9})
        engine.fit(X_train, y_train)

        mask = engine.match_mask(X_test)
        recall = mask[y_test.values == 1].mean()
        assert 0.0 < recall < 1.0

    def test_empty_rules_before_fit_raises(self):
        engine = SignatureEngine()
        X = pd.DataFrame({"flag": ["SF", "S0"]})
        with pytest.raises(ValueError):
            engine.match_mask(X)

    def test_no_rules_mined_returns_all_false(self):
        """If nothing meets min_confidence/min_support, match_mask should be
        all-False rather than erroring."""
        X = pd.DataFrame({"flag": ["SF"] * 100})
        y = pd.Series(np.zeros(100, dtype=int))  # no attacks at all
        engine = SignatureEngine({"min_support": 0.005, "min_confidence": 0.9})
        engine.fit(X, y)
        mask = engine.match_mask(X)
        assert mask.sum() == 0


class TestSignatureEnsembleFusion:
    """Test cases for SignatureEnsembleFusion"""

    @pytest.fixture
    def data(self):
        X, y = _make_synthetic_dataset()
        return X.iloc[:1500], y.iloc[:1500], X.iloc[1500:], y.iloc[1500:]

    def test_fusion_overrides_with_signature_hits(self, data):
        """Even with an ensemble that predicts 'Normal' for everything,
        fusion should still catch every signature-matched attack."""
        X_train, y_train, X_test, y_test = data
        fusion = SignatureEnsembleFusion(
            SignatureEngine({"min_support": 0.005, "min_confidence": 0.9}),
            _DummyEnsemble(),
        )
        fusion.train(X_train, y_train)
        pred = fusion.predict(X_test)

        sig_mask = fusion.signature_engine.match_mask(X_test)
        assert (pred[sig_mask] == 1).all()
        # Non-signature rows fall through to the (always-"Normal") ensemble
        assert (pred[~sig_mask] == 0).all()

    def test_fusion_beats_ensemble_alone_on_known_attacks(self, data):
        X_train, y_train, X_test, y_test = data
        fusion = SignatureEnsembleFusion(
            SignatureEngine({"min_support": 0.005, "min_confidence": 0.9}),
            _DummyEnsemble(),
        )
        fusion.train(X_train, y_train)

        fusion_acc = (fusion.predict(X_test) == y_test.values).mean()
        ensemble_only_acc = (fusion.ensemble.predict(X_test) == y_test.values).mean()

        assert fusion_acc > ensemble_only_acc

    def test_predict_proba_shape_and_overrides(self, data):
        X_train, y_train, X_test, y_test = data
        fusion = SignatureEnsembleFusion(
            SignatureEngine({"min_support": 0.005, "min_confidence": 0.9}),
            _DummyEnsemble(),
        )
        fusion.train(X_train, y_train)
        proba = fusion.predict_proba(X_test)

        assert proba.shape == (len(X_test), 2)
        sig_mask = fusion.signature_engine.match_mask(X_test)
        assert np.allclose(proba[sig_mask], [0.01, 0.99])

    def test_fusion_report_contents(self, data):
        X_train, y_train, X_test, y_test = data
        fusion = SignatureEnsembleFusion(
            SignatureEngine({"min_support": 0.005, "min_confidence": 0.9}),
            _DummyEnsemble(),
        )
        fusion.train(X_train, y_train)
        report = fusion.get_fusion_report(X_test, y_test)

        assert report["n_records"] == len(X_test)
        assert 0.0 < report["signature_hit_rate"] < 1.0
        assert "recall_within_signature_hits" in report

    def test_predict_before_train_raises(self, data):
        X_train, y_train, X_test, _ = data
        fusion = SignatureEnsembleFusion(SignatureEngine(), _DummyEnsemble())
        with pytest.raises(ValueError):
            fusion.predict(X_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
