"""
Multi-class extension for NIDS
Scalable to multi-class attack type classification
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiClassNIDS:
    """Multi-class version of NIDS for attack type classification"""
    
    def __init__(self):
        """Initialize multi-class NIDS"""
        self.label_encoder = None
        self.class_labels = None
        self.n_classes = 0
    
    def prepare_multiclass_labels(self, y: pd.Series) -> Tuple[np.ndarray, dict]:
        """
        Encode multi-class labels
        
        Args:
            y: Target series with attack type labels
            
        Returns:
            Tuple of (encoded_labels, label_mapping)
        """
        logger.info("Preparing multi-class labels...")
        
        unique_labels = y.unique()
        self.n_classes = len(unique_labels)
        
        logger.info(f"Found {self.n_classes} classes:")
        for i, label in enumerate(unique_labels):
            count = (y == label).sum()
            logger.info(f"  {i}: {label} ({count} samples)")
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create mapping
        label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        
        return y_encoded, label_mapping
    
    def get_class_labels(self):
        """Get class labels"""
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call prepare_multiclass_labels first.")
        
        return self.label_encoder.classes_
    
    def decode_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Decode predicted labels back to original class names
        
        Args:
            y_pred: Encoded predictions
            
        Returns:
            Original class labels
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted.")
        
        return self.label_encoder.inverse_transform(y_pred)
    
    def get_class_distribution(self, y: pd.Series) -> pd.DataFrame:
        """
        Get class distribution statistics
        
        Args:
            y: Target series
            
        Returns:
            DataFrame with distribution statistics
        """
        value_counts = y.value_counts()
        percentages = (value_counts / len(y) * 100).round(2)
        
        distribution = pd.DataFrame({
            "class": value_counts.index,
            "count": value_counts.values,
            "percentage": percentages.values
        })
        
        logger.info(f"Class Distribution:\n{distribution}")
        return distribution
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                              strategy: str = "oversample") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance
        
        Args:
            X: Features
            y: Labels
            strategy: "oversample" or "undersample"
            
        Returns:
            Balanced (X, y)
        """
        logger.info(f"Handling class imbalance using {strategy}...")
        
        if strategy == "oversample":
            from sklearn.utils import resample
            
            X_balanced = pd.DataFrame()
            y_balanced = pd.Series(dtype=y.dtype)
            
            # Get max class count
            max_count = y.value_counts().max()
            
            for class_label in y.unique():
                X_class = X[y == class_label]
                y_class = y[y == class_label]
                
                # Oversample if needed
                if len(X_class) < max_count:
                    X_resampled = resample(X_class, n_samples=max_count, random_state=42, replace=True)
                    y_resampled = pd.Series([class_label] * max_count, dtype=y.dtype)
                else:
                    X_resampled = X_class
                    y_resampled = y_class
                
                X_balanced = pd.concat([X_balanced, X_resampled], ignore_index=True)
                y_balanced = pd.concat([y_balanced, y_resampled], ignore_index=True)
        
        elif strategy == "undersample":
            from sklearn.utils import resample
            
            X_balanced = pd.DataFrame()
            y_balanced = pd.Series(dtype=y.dtype)
            
            # Get min class count
            min_count = y.value_counts().min()
            
            for class_label in y.unique():
                X_class = X[y == class_label]
                y_class = y[y == class_label]
                
                # Undersample if needed
                if len(X_class) > min_count:
                    X_resampled = resample(X_class, n_samples=min_count, random_state=42, replace=False)
                    y_resampled = pd.Series([class_label] * min_count, dtype=y.dtype)
                else:
                    X_resampled = X_class
                    y_resampled = y_class
                
                X_balanced = pd.concat([X_balanced, X_resampled], ignore_index=True)
                y_balanced = pd.concat([y_balanced, y_resampled], ignore_index=True)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Balanced dataset shape: {X_balanced.shape}")
        logger.info(f"New class distribution:\n{y_balanced.value_counts()}")
        
        return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)
    
    def get_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Get per-class performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with per-class metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = []
        
        for class_idx in range(self.n_classes):
            class_label = self.label_encoder.classes_[class_idx]
            
            # Binary metrics for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            metrics.append({
                "class": class_label,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
        
        metrics_df = pd.DataFrame(metrics)
        logger.info(f"\nPer-Class Metrics:\n{metrics_df}")
        
        return metrics_df
    
    def create_multiclass_ensemble_weights(self, y_train: np.ndarray, 
                                          strategy: str = "balanced") -> np.ndarray:
        """
        Create weights for ensemble models based on class distribution
        
        Args:
            y_train: Training labels
            strategy: "balanced" or "uniform"
            
        Returns:
            Class weights
        """
        if strategy == "balanced":
            from sklearn.utils.class_weight import compute_class_weight
            
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            logger.info(f"Balanced class weights: {weights}")
        
        elif strategy == "uniform":
            weights = np.ones(self.n_classes)
            logger.info(f"Uniform class weights: {weights}")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return weights
