"""
Data loading and exploration module for NIDS
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and explore network intrusion detection datasets"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Path to directory containing datasets
        """
        self.data_dir = Path(data_dir)
        self.data = None
        self.file_path = None
        
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            filename: Name of CSV file to load
            
        Returns:
            Loaded DataFrame
        """
        self.file_path = self.data_dir / filename
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")
        
        logger.info(f"Loading dataset from {self.file_path}")
        self.data = pd.read_csv(self.file_path)
        logger.info(f"Dataset loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        return self.data
    
    def get_dataset_info(self) -> dict:
        """
        Get comprehensive dataset information
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.data is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        info = {
            "shape": self.data.shape,
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": self.data.dtypes.value_counts().to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicate_rows": self.data.duplicated().sum(),
            "columns": list(self.data.columns),
        }
        
        logger.info(f"Dataset Info - Shape: {info['shape']}, Memory: {info['memory_usage_mb']:.2f} MB")
        return info
    
    def get_class_distribution(self, target_col: str) -> dict:
        """
        Get class distribution in target column
        
        Args:
            target_col: Name of target column
            
        Returns:
            Dictionary with class counts and percentages
        """
        if self.data is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        value_counts = self.data[target_col].value_counts()
        percentages = (value_counts / len(self.data) * 100).round(2)
        
        distribution = {
            "counts": value_counts.to_dict(),
            "percentages": percentages.to_dict(),
            "class_balance_ratio": value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1.0
        }
        
        logger.info(f"Class Distribution:\n{value_counts}\n{percentages}%")
        return distribution
    
    def get_feature_statistics(self) -> pd.DataFrame:
        """
        Get statistical summary of numerical features
        
        Returns:
            DataFrame with statistics
        """
        if self.data is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        stats = self.data.describe().T
        logger.info(f"Feature Statistics:\n{stats}")
        return stats
    
    def get_missing_values_summary(self) -> pd.DataFrame:
        """
        Get summary of missing values
        
        Returns:
            DataFrame with missing value information
        """
        if self.data is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        missing = pd.DataFrame({
            "column": self.data.columns,
            "missing_count": self.data.isnull().sum().values,
            "missing_percent": (self.data.isnull().sum().values / len(self.data) * 100).round(2)
        })
        
        missing = missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=False)
        
        if len(missing) > 0:
            logger.info(f"Missing Values:\n{missing}")
        else:
            logger.info("No missing values found")
        
        return missing
    
    def get_numerical_columns(self) -> list:
        """Get list of numerical columns"""
        if self.data is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        return self.data.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self) -> list:
        """Get list of categorical columns"""
        if self.data is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        return self.data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def explore_dataset(self, target_col: Optional[str] = None) -> dict:
        """
        Perform comprehensive exploratory data analysis
        
        Args:
            target_col: Optional target column name
            
        Returns:
            Dictionary with all exploration results
        """
        logger.info("Starting exploratory data analysis...")
        
        exploration = {
            "info": self.get_dataset_info(),
            "feature_stats": self.get_feature_statistics(),
            "missing_values": self.get_missing_values_summary(),
            "numerical_cols": self.get_numerical_columns(),
            "categorical_cols": self.get_categorical_columns(),
        }
        
        if target_col:
            exploration["class_distribution"] = self.get_class_distribution(target_col)
        
        logger.info("Exploratory data analysis completed")
        return exploration
    
    def get_data(self) -> pd.DataFrame:
        """Get currently loaded dataset"""
        return self.data.copy()
    
    def save_dataset(self, output_path: Path, filename: str = "processed_data.csv") -> Path:
        """
        Save processed dataset
        
        Args:
            output_path: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if self.data is None:
            raise ValueError("No dataset to save. Load a dataset first.")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_file = output_path / filename
        self.data.to_csv(save_file, index=False)
        
        logger.info(f"Dataset saved to {save_file}")
        return save_file
