"""
Unit tests for NIDS feature engineering
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from features.engineer import FeatureEngineer
from config.settings import FEATURE_ENGINEERING_CONFIG


class TestFeatureEngineer:
    """Test cases for FeatureEngineer"""
    
    @pytest.fixture
    def sample_data_with_labels(self):
        """Create sample data with labels"""
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100) * 1000,  # Different scale
            'feature_4': np.ones(100),  # Low variance
            'feature_5': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        return X, y
    
    def test_feature_importance(self, sample_data_with_labels):
        """Test feature importance calculation"""
        X, y = sample_data_with_labels
        engineer = FeatureEngineer(FEATURE_ENGINEERING_CONFIG)
        
        importance = engineer.get_feature_importance(X, y, method="f_classif")
        
        assert len(importance) == X.shape[1]
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_remove_low_variance_features(self, sample_data_with_labels):
        """Test low variance feature removal"""
        X, y = sample_data_with_labels
        engineer = FeatureEngineer(FEATURE_ENGINEERING_CONFIG)
        
        result = engineer.remove_low_variance_features(X, threshold=0.01)
        
        # Should remove feature_4 (variance = 0)
        assert 'feature_4' not in result.columns or result['feature_4'].var() > 0.01
    
    def test_select_features(self, sample_data_with_labels):
        """Test feature selection"""
        X, y = sample_data_with_labels
        engineer = FeatureEngineer(FEATURE_ENGINEERING_CONFIG)
        
        result = engineer.select_features(X, y, k=3, fit=True)
        
        assert result.shape[1] == 3
        assert engineer.selected_features is not None
    
    def test_fit_transform(self, sample_data_with_labels):
        """Test fit_transform"""
        X, y = sample_data_with_labels
        engineer = FeatureEngineer(FEATURE_ENGINEERING_CONFIG)
        
        result = engineer.fit_transform(X, y)
        
        # Should have processed data
        assert len(result) == len(X)
        assert result.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
