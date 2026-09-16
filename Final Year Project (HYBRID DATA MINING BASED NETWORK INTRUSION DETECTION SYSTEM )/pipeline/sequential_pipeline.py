"""
Sequential pipeline for NIDS
Combines preprocessing and feature engineering into a sklearn-compatible pipeline
"""
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible preprocessing transformer"""
    
    def __init__(self, preprocessor):
        """
        Initialize
        
        Args:
            preprocessor: DataPreprocessor instance
        """
        self.preprocessor = preprocessor
    
    def fit(self, X, y=None):
        """Fit preprocessor"""
        self.preprocessor.fit_transform(X)
        return self
    
    def transform(self, X):
        """Transform data"""
        return self.preprocessor.transform(X)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature engineering transformer"""
    
    def __init__(self, engineer, y_train: Optional[pd.Series] = None):
        """
        Initialize
        
        Args:
            engineer: FeatureEngineer instance
            y_train: Training target (needed for feature selection)
        """
        self.engineer = engineer
        self.y_train = y_train
    
    def fit(self, X, y=None):
        """Fit engineer"""
        if self.y_train is not None:
            self.engineer.fit_transform(X, self.y_train)
        return self
    
    def transform(self, X):
        """Transform data"""
        return self.engineer.transform(X)


class SequentialPipeline:
    """
    Sequential pipeline for preprocessing and feature engineering
    Not sklearn Pipeline - custom implementation for flexibility
    """
    
    def __init__(self, preprocessor, feature_engineer, y_train: Optional[pd.Series] = None):
        """
        Initialize pipeline
        
        Args:
            preprocessor: DataPreprocessor instance
            feature_engineer: FeatureEngineer instance
            y_train: Training target (for feature engineering)
        """
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.y_train = y_train
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the pipeline (training data)
        
        Args:
            X: Features dataframe
            y: Target series
            
        Returns:
            Self
        """
        logger.info("Fitting sequential pipeline...")
        
        self.y_train = y
        
        # Fit preprocessor
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        # Fit feature engineer
        self.feature_engineer.fit_transform(X_preprocessed, y)
        
        self.is_fitted = True
        logger.info("Pipeline fitted successfully")
        
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform data
        
        Args:
            X: Features dataframe
            y: Target series
            
        Returns:
            Transformed dataframe
        """
        self.fit(X, y)
        return self.transform(X)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline
        
        Args:
            X: Features dataframe
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        logger.info("Applying sequential pipeline...")
        
        # Apply preprocessing
        X_preprocessed = self.preprocessor.transform(X)
        
        # Apply feature engineering
        X_final = self.feature_engineer.transform(X_preprocessed)
        
        logger.info(f"Pipeline output shape: {X_final.shape}")
        
        return X_final
    
    def get_feature_names(self):
        """Get final feature names after pipeline"""
        return self.feature_engineer.selected_features


class SklearnPipeline:
    """
    Sklearn-compatible wrapper for sequential pipeline
    This allows the pipeline to work with sklearn's GridSearchCV, cross_val_score, etc.
    """
    
    def __init__(self, preprocessor, feature_engineer):
        """Initialize"""
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.pipeline = Pipeline([
            ('preprocessing', PreprocessingTransformer(preprocessor)),
            ('feature_engineering', FeatureEngineeringTransformer(feature_engineer))
        ])
    
    def fit(self, X, y=None):
        """Fit pipeline"""
        self.pipeline.fit(X, y)
        return self
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X):
        """Transform"""
        return self.pipeline.transform(X)
