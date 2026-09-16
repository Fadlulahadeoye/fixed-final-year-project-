"""
Base model interface for NIDS
"""
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseNIDSModel(ABC):
    """Abstract base class for NIDS models"""
    
    def __init__(self, name: str, model, config: dict):
        """
        Initialize base model
        
        Args:
            name: Model name
            model: Sklearn model instance
            config: Model configuration
        """
        self.name = name
        self.model = model
        self.config = config
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Get prediction probabilities"""
        pass
    
    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.name} not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
        }
        
        logger.info(f"{self.name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return self.model.coef_[0]
        else:
            return None
