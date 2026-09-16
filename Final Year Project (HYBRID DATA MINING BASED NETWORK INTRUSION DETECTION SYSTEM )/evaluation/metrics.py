"""
Evaluation metrics for NIDS
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class NIDSEvaluator:
    """Comprehensive evaluation metrics for NIDS"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            threshold: Classification threshold for probabilities
        """
        self.threshold = threshold
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model performance...")
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba),
        }
        
        logger.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get classification report"""
        return classification_report(y_true, y_pred, target_names=["Normal", "Attack"])
    
    def get_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            (fpr, tpr, thresholds)
        """
        proba = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        return fpr, tpr, thresholds
    
    def get_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get precision-recall curve"""
        proba = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        return precision, recall, thresholds
    
    def optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          metric: str = "f1") -> Tuple[float, float]:
        """
        Optimize classification threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            metric: Metric to optimize (f1, precision, recall)
            
        Returns:
            (optimal_threshold, best_metric_value)
        """
        proba = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
        
        best_threshold = 0.5
        best_value = -1
        
        for threshold in np.linspace(0.1, 0.9, 50):
            y_pred_thresholded = (proba >= threshold).astype(int)
            
            if metric == "f1":
                value = f1_score(y_true, y_pred_thresholded, zero_division=0)
            elif metric == "precision":
                value = precision_score(y_true, y_pred_thresholded, zero_division=0)
            elif metric == "recall":
                value = recall_score(y_true, y_pred_thresholded, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if value > best_value:
                best_value = value
                best_threshold = threshold
        
        logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} ({best_value:.4f})")
        self.threshold = best_threshold
        
        return best_threshold, best_value
    
    def get_metrics_summary(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: np.ndarray) -> pd.DataFrame:
        """Get metrics summary as DataFrame"""
        metrics = self.evaluate(y_true, y_pred, y_pred_proba)
        
        # Get confusion matrix
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        # Get per-class metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics["sensitivity"] = sensitivity
        metrics["specificity"] = specificity
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn
        metrics["tp"] = tp
        
        return pd.DataFrame([metrics])
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple model results
        
        Args:
            model_results: Dictionary of model names and their metrics
            
        Returns:
            Comparison DataFrame
        """
        comparison_df = pd.DataFrame(model_results).T
        logger.info(f"\nModel Comparison:\n{comparison_df}")
        return comparison_df
