"""
Visualization utilities for NIDS
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class NIDSVisualizer:
    """Visualization utilities for NIDS evaluation"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer"""
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None) -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self, model_results: dict, save_path: str = None) -> None:
        """
        Plot ROC curves for multiple models
        
        Args:
            model_results: Dictionary with model names and (fpr, tpr, auc) tuples
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, (fpr, tpr, auc) in model_results.items():
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curves(self, model_results: dict, save_path: str = None) -> None:
        """
        Plot precision-recall curves for multiple models
        
        Args:
            model_results: Dictionary with model names and (precision, recall) arrays
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, (precision, recall) in model_results.items():
            plt.plot(recall, precision, label=model_name, linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, metrics_df, save_path: str = None) -> None:
        """
        Plot metrics comparison across models
        
        Args:
            metrics_df: DataFrame with models as rows and metrics as columns
            save_path: Path to save figure (optional)
        """
        # Select key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        plot_data = metrics_df[[col for col in key_metrics if col in metrics_df.columns]]
        
        plt.figure(figsize=self.figsize)
        plot_data.plot(kind='bar', ax=plt.gca())
        
        plt.title('Model Metrics Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim([0, 1.05])
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, importances: dict, top_n: int = 20, 
                               save_path: str = None) -> None:
        """
        Plot feature importance
        
        Args:
            importances: Dictionary of feature importance scores
            top_n: Number of top features to plot
            save_path: Path to save figure (optional)
        """
        # Sort and get top N
        sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        plt.figure(figsize=(12, 6))
        plt.barh(list(sorted_imp.keys()), list(sorted_imp.values()))
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Important Features')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance saved to {save_path}")
        
        plt.close()
    
    def plot_class_distribution(self, y: np.ndarray, save_path: str = None) -> None:
        """
        Plot class distribution
        
        Args:
            y: Target labels
            save_path: Path to save figure (optional)
        """
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Normal', 'Attack'], counts)
        plt.ylabel('Count')
        plt.title('Class Distribution')
        
        # Add percentages
        for i, (label, count) in enumerate(zip(['Normal', 'Attack'], counts)):
            percentage = count / len(y) * 100
            plt.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution saved to {save_path}")
        
        plt.close()
