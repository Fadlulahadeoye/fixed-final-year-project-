"""
Ensemble voting classifier for NIDS
"""
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from models.base_model import BaseNIDSModel
from utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleVotingClassifier(BaseNIDSModel):
    """Ensemble voting classifier combining multiple NIDS models"""
    
    def __init__(self, models: List[BaseNIDSModel], voting: str = "soft", 
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble voting classifier
        
        Args:
            models: List of BaseNIDSModel instances
            voting: "hard" or "soft" voting
            weights: Optional weights for each model (defaults to equal weights)
        """
        self.models = models
        self.voting = voting
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.is_trained = False
        
        # Normalize weights
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()
        
        config = {
            "voting": voting,
            "n_models": len(models),
            "weights": self.weights.tolist()
        }
        
        super().__init__("EnsembleVotingClassifier", None, config)
    
    def train(self, X_train, y_train):
        """
        Train all models in the ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training ensemble with {len(self.models)} models using {self.voting} voting...")
        
        for i, model in enumerate(self.models):
            if not getattr(model, "is_trained", False):
                logger.info(f"  Training model {i+1}/{len(self.models)}: {model.name}")
                model.train(X_train, y_train)
            else:
                logger.info(f"  Reusing already-trained model {i+1}/{len(self.models)}: {model.name}")
        
        self.is_trained = True
        logger.info("Ensemble training complete")
    
    def predict(self, X):
        """
        Make ensemble predictions
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        if self.voting == "hard":
            return self._predict_hard_voting(X)
        else:
            return self._predict_soft_voting(X)
    
    def _predict_hard_voting(self, X) -> np.ndarray:
        """Hard voting - majority vote"""
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Weighted majority vote
        votes = np.zeros((X.shape[0], 2))
        for i, model_pred in enumerate(predictions):
            for class_label in [0, 1]:
                votes[:, class_label] += (model_pred == class_label) * self.weights[i]
        
        return np.argmax(votes, axis=1)
    
    def _predict_soft_voting(self, X) -> np.ndarray:
        """Soft voting - probability averaging"""
        proba_list = []
        
        for model in self.models:
            proba = model.predict_proba(X)
            # Handle models that return 1D array (e.g., SVM decision_function)
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            proba_list.append(proba)
        
        # Weighted average of probabilities
        weighted_proba = np.average(proba_list, axis=0, weights=self.weights)
        
        return np.argmax(weighted_proba, axis=1)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Get ensemble prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        proba_list = []
        
        for model in self.models:
            proba = model.predict_proba(X)
            # Handle models that return 1D array
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            proba_list.append(proba)
        
        # Weighted average of probabilities
        weighted_proba = np.average(proba_list, axis=0, weights=self.weights)
        
        return weighted_proba
    
    def get_model_predictions(self, X) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model
        
        Args:
            X: Features
            
        Returns:
            Dictionary with predictions from each model
        """
        model_predictions = {}
        
        for model in self.models:
            model_predictions[model.name] = model.predict(X)
        
        return model_predictions
    
    def get_model_probabilities(self, X) -> Dict[str, np.ndarray]:
        """
        Get probabilities from each individual model
        
        Args:
            X: Features
            
        Returns:
            Dictionary with probabilities from each model
        """
        model_probas = {}
        
        for model in self.models:
            proba = model.predict_proba(X)
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            model_probas[model.name] = proba
        
        return model_probas
    
    def get_ensemble_summary(self) -> str:
        """Get summary of ensemble configuration"""
        summary = f"""
EnsembleVotingClassifier
==========================
Voting Method: {self.voting}
Number of Models: {len(self.models)}
Model Weights:
"""
        for model, weight in zip(self.models, self.weights):
            summary += f"  {model.name}: {weight:.3f}\n"
        
        return summary
