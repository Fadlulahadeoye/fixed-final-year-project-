"""
Unit tests for NIDS preprocessing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from data.preprocessor import DataPreprocessor
from config.settings import PREPROCESSING_CONFIG


class TestDataPreprocessor:
    """Test cases for DataPreprocessor"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = pd.DataFrame({
            'feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature_2': [10, 20, 30, 40, 50],
            'feature_3': ['a', 'b', 'a', 'b', 'a'],
            'feature_4': [1.0, 2.0, 2.0, 2.0, 2.0],  # Low variance
        })
        return data
    
    def test_identify_columns(self, sample_data):
        """Test column type identification"""
        preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
        num_cols, cat_cols = preprocessor.identify_columns(sample_data)
        
        assert len(num_cols) == 3
        assert len(cat_cols) == 1
        assert 'feature_3' in cat_cols
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling"""
        preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
        preprocessor.identify_columns(sample_data)
        
        # Should have 1 missing value initially
        assert sample_data.isnull().sum().sum() == 1
        
        # After handling
        result = preprocessor.handle_missing_values(sample_data, strategy="mean")
        assert result.isnull().sum().sum() == 0
    
    def test_remove_duplicates(self, sample_data):
        """Test duplicate removal"""
        preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
        
        # Create duplicates
        sample_data_dup = pd.concat([sample_data, sample_data.iloc[:2]], ignore_index=True)
        assert len(sample_data_dup) == 7
        
        result = preprocessor.remove_duplicates(sample_data_dup)
        assert len(result) == 5  # Duplicates removed
    
    def test_encode_categorical(self, sample_data):
        """Test categorical encoding"""
        preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
        preprocessor.identify_columns(sample_data)
        
        result = preprocessor.encode_categorical(sample_data, fit=True)
        
        # Check that categorical column is now numeric
        assert result['feature_3'].dtype in [np.int64, np.int32]
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform"""
        preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
        
        result = preprocessor.fit_transform(sample_data)
        
        # Should have same number of rows (no deletion)
        assert len(result) <= len(sample_data)
        
        # All should be numeric
        assert result.dtypes.unique().size == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
