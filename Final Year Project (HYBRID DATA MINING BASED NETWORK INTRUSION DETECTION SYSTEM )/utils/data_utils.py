"""
Data utilities for NIDS
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


# NSL-KDD column names
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]


def load_nsl_kdd(filepath: Path) -> pd.DataFrame:
    """
    Load NSL-KDD dataset from TXT file
    
    Args:
        filepath: Path to NSL-KDD TXT file
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading NSL-KDD dataset from {filepath}")
    df = pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS)
    logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def convert_to_binary_classification(df: pd.DataFrame, normal_label: str = "normal") -> pd.DataFrame:
    """
    Convert multi-class labels to binary (normal vs. attack)
    
    Args:
        df: Input dataframe with 'label' column
        normal_label: Label for normal traffic
        
    Returns:
        DataFrame with binary labels (0: normal, 1: attack)
    """
    logger.info(f"Converting to binary classification (normal vs. attack)")
    
    df = df.copy()
    df['label'] = (df['label'] != normal_label).astype(int)
    
    class_counts = df['label'].value_counts()
    logger.info(f"Binary distribution - Normal: {class_counts.get(0, 0)}, Attack: {class_counts.get(1, 0)}")
    
    return df


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    random_state: int = 42, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets
    
    Args:
        X: Features dataframe
        y: Target series
        test_size: Test set proportion
        random_state: Random seed
        stratify: Whether to stratify by target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_arg = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )
    
    logger.info(f"Train-test split: {len(X_train)} train, {len(X_test)} test ({test_size*100:.0f}% test)")
    
    return X_train, X_test, y_train, y_test


def save_processed_dataset(df: pd.DataFrame, output_path: Path, filename: str = "processed.csv") -> Path:
    """
    Save processed dataset to CSV
    
    Args:
        df: DataFrame to save
        output_path: Output directory
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    
    logger.info(f"Dataset saved to {filepath}")
    return filepath


def prepare_nsl_kdd_dataset(raw_path: Path, processed_path: Path, 
                           convert_binary: bool = True, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline to prepare NSL-KDD dataset
    
    Args:
        raw_path: Path to raw NSL-KDD files
        processed_path: Path to save processed files
        convert_binary: Whether to convert to binary classification
        test_size: Test set proportion
        
    Returns:
        Tuple of (X_train, X_test) DataFrames (y labels are in separate columns)
    """
    logger.info("Starting NSL-KDD dataset preparation...")
    
    # Load training data
    train_file = raw_path / "KDDTrain+_20Percent.txt"
    df_train = load_nsl_kdd(train_file)
    
    # Load test data
    test_file = raw_path / "KDDTest+.txt"
    df_test = load_nsl_kdd(test_file)
    
    # Convert to binary if requested
    if convert_binary:
        df_train = convert_to_binary_classification(df_train)
        df_test = convert_to_binary_classification(df_test)
    
    # Drop 'difficulty' column (not needed)
    df_train = df_train.drop('difficulty', axis=1)
    df_test = df_test.drop('difficulty', axis=1)
    
    # Save processed datasets
    train_output = save_processed_dataset(df_train, processed_path, "nsl_kdd_train.csv")
    test_output = save_processed_dataset(df_test, processed_path, "nsl_kdd_test.csv")
    
    logger.info(f"NSL-KDD dataset preparation complete")
    logger.info(f"Training set: {df_train.shape[0]} samples, {df_train.shape[1]} features")
    logger.info(f"Test set: {df_test.shape[0]} samples, {df_test.shape[1]} features")
    
    return df_train, df_test
