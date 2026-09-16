"""
Unit tests for NIDS ensemble
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from models.classifiers import XGBoostNIDSModel, LightGBMNIDSModel
from models.ensemble import EnsembleVotingClassifier
from config.settings import XGBOOST_CONFIG, LIGHTGBM_CONFIG


class TestEnsembleVoting:
    """Test cases for EnsembleVotingClassifier"""
    
    @pytest.fixture
    def sample_data_trained(self):
        """Create and train sample models"""
        # Create sample data
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        # Create models
        xgb = XGBoostNIDSModel(XGBOOST_CONFIG)
        lgb = LightGBMNIDSModel(LIGHTGBM_CONFIG)
        
        # Train models
        xgb.train(X_train, y_train)
        lgb.train(X_train, y_train)
        
        return [xgb, lgb], X_train, y_train
    
    def test_ensemble_initialization(self, sample_data_trained):
        """Test ensemble initialization"""
        models, _, _ = sample_data_trained
        
        ensemble = EnsembleVotingClassifier(models, voting="soft")
        
        assert len(ensemble.models) == 2
        assert ensemble.voting == "soft"
    
    def test_ensemble_hard_voting(self, sample_data_trained):
        """Test hard voting"""
        models, X_train, y_train = sample_data_trained
        
        ensemble = EnsembleVotingClassifier(models, voting="hard")
        ensemble.train(X_train, y_train)
        
        y_pred = ensemble.predict(X_train)
        
        assert y_pred.shape[0] == len(X_train)
        assert all(y in [0, 1] for y in y_pred)
    
    def test_ensemble_soft_voting(self, sample_data_trained):
        """Test soft voting"""
        models, X_train, y_train = sample_data_trained
        
        ensemble = EnsembleVotingClassifier(models, voting="soft")
        ensemble.train(X_train, y_train)
        
        y_pred = ensemble.predict(X_train)
        y_proba = ensemble.predict_proba(X_train)
        
        assert y_pred.shape[0] == len(X_train)
        assert y_proba.shape == (len(X_train), 2)
    
    def test_ensemble_weighted_voting(self, sample_data_trained):
        """Test weighted voting"""
        models, X_train, y_train = sample_data_trained
        
        weights = [0.6, 0.4]
        ensemble = EnsembleVotingClassifier(models, voting="soft", weights=weights)
        ensemble.train(X_train, y_train)
        
        # Weights should be normalized
        assert np.isclose(ensemble.weights.sum(), 1.0)
    
    def test_ensemble_summary(self, sample_data_trained):
        """Test ensemble summary"""
        models, X_train, y_train = sample_data_trained
        
        ensemble = EnsembleVotingClassifier(models, voting="soft")
        summary = ensemble.get_ensemble_summary()
        
        assert "EnsembleVotingClassifier" in summary
        assert "soft" in summary
        assert "2" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
