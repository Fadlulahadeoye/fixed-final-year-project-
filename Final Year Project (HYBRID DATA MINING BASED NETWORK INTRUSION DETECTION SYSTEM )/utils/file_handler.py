"""
File handler for NIDS models and pipelines
Saving and loading trained models
"""
import pickle
import joblib
from pathlib import Path
from typing import Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelFileHandler:
    """Handle saving and loading NIDS models and pipelines"""
    
    @staticmethod
    def save_model(model: Any, save_path: Path, filename: str, 
                  format: str = "joblib") -> Path:
        """
        Save model to disk
        
        Args:
            model: Model object to save
            save_path: Directory to save to
            filename: Filename (without extension)
            format: Save format ("joblib" or "pickle")
            
        Returns:
            Path to saved file
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        extension = ".joblib" if format == "joblib" else ".pkl"
        filepath = save_path / f"{filename}{extension}"
        
        try:
            if format == "joblib":
                joblib.dump(model, filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"Model saved to {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    @staticmethod
    def load_model(filepath: Path, format: str = "joblib") -> Any:
        """
        Load model from disk
        
        Args:
            filepath: Path to model file
            format: Load format ("joblib" or "pickle")
            
        Returns:
            Loaded model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            if format == "joblib":
                model = joblib.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"Model loaded from {filepath}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    def save_ensemble(ensemble, save_path: Path, name: str = "ensemble") -> Path:
        """
        Save ensemble model
        
        Args:
            ensemble: Ensemble classifier
            save_path: Directory to save to
            name: Model name
            
        Returns:
            Path to saved file
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        filepath = save_path / f"{name}.joblib"
        
        try:
            joblib.dump(ensemble, filepath)
            logger.info(f"Ensemble saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
            raise
    
    @staticmethod
    def load_ensemble(filepath: Path):
        """Load ensemble model"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Ensemble file not found: {filepath}")
        
        try:
            ensemble = joblib.load(filepath)
            logger.info(f"Ensemble loaded from {filepath}")
            return ensemble
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            raise
    
    @staticmethod
    def save_pipeline(pipeline, save_path: Path, name: str = "pipeline") -> Path:
        """
        Save preprocessing/feature engineering pipeline
        
        Args:
            pipeline: Pipeline object
            save_path: Directory to save to
            name: Pipeline name
            
        Returns:
            Path to saved file
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        filepath = save_path / f"{name}.joblib"
        
        try:
            joblib.dump(pipeline, filepath)
            logger.info(f"Pipeline saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise
    
    @staticmethod
    def load_pipeline(filepath: Path):
        """Load pipeline"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        try:
            pipeline = joblib.load(filepath)
            logger.info(f"Pipeline loaded from {filepath}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    @staticmethod
    def list_models(directory: Path) -> list:
        """
        List all saved models in directory
        
        Args:
            directory: Directory to search
            
        Returns:
            List of model files
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        models = list(directory.glob("*.joblib")) + list(directory.glob("*.pkl"))
        logger.info(f"Found {len(models)} models in {directory}")
        
        return models
