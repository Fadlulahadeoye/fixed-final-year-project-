"""Leakage-safe preprocessing for the NSL-KDD NIDS."""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Fit preprocessing on training data and reproduce it at inference time.

    Categorical network fields are one-hot encoded instead of label encoded.
    This avoids introducing a false numeric ordering into SVM/Logistic Regression.
    """
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.encoder = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.output_columns = []
        self.outlier_bounds = {}
        self.fitted = False

    def identify_columns(self, X: pd.DataFrame) -> Tuple[list, list]:
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        return self.numerical_cols, self.categorical_cols

    def handle_missing_values(self, X: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Backward-compatible public missing-value helper."""
        X = X.copy()
        self.identify_columns(X)
        if self.numerical_cols:
            self.numeric_imputer = SimpleImputer(strategy=strategy)
            X.loc[:, self.numerical_cols] = self.numeric_imputer.fit_transform(X[self.numerical_cols])
        if self.categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            X.loc[:, self.categorical_cols] = self.categorical_imputer.fit_transform(X[self.categorical_cols])
        return X

    def _fit_imputers(self, X: pd.DataFrame):
        if self.numerical_cols:
            strategy = self.config.get("missing_value_strategy", "mean")
            self.numeric_imputer = SimpleImputer(strategy=strategy)
            self.numeric_imputer.fit(X[self.numerical_cols])
        if self.categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            self.categorical_imputer.fit(X[self.categorical_cols])

    def _impute(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.copy()
        if fit:
            self._fit_imputers(X)
        if self.numerical_cols:
            X.loc[:, self.numerical_cols] = self.numeric_imputer.transform(X[self.numerical_cols])
        if self.categorical_cols:
            X.loc[:, self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])
        return X

    def remove_duplicates(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop_duplicates()

    def _fit_outlier_bounds(self, X: pd.DataFrame):
        self.outlier_bounds = {}
        method = self.config.get("outlier_detection")
        if not method or not self.numerical_cols:
            return
        if method == "iqr":
            multiplier = self.config.get("outlier_iqr_multiplier", 1.5)
            for col in self.numerical_cols:
                q1, q3 = X[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.outlier_bounds[col] = (q1 - multiplier * iqr, q3 + multiplier * iqr)
        elif method == "zscore":
            means = X[self.numerical_cols].mean()
            stds = X[self.numerical_cols].std(ddof=0).replace(0, np.nan)
            for col in self.numerical_cols:
                self.outlier_bounds[col] = (means[col] - 3 * stds[col], means[col] + 3 * stds[col])
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def _remove_training_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.outlier_bounds:
            return X
        mask = np.zeros(len(X), dtype=bool)
        for col, (lower, upper) in self.outlier_bounds.items():
            if pd.isna(lower) or pd.isna(upper):
                continue
            mask |= (X[col] < lower) | (X[col] > upper)
        removed = int(mask.sum())
        if removed:
            logger.info("Removed %d training outliers (outlier filtering is disabled by default for NSL-KDD)", removed)
        return X.loc[~mask]

    def encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        X = X.copy()
        if not self.categorical_cols:
            return X
        # If called as a standalone helper (outside fit_transform), preserve the
        # historical API: return the original categorical column as integer codes.
        # The real training pipeline always has categorical_imputer fitted and uses
        # OneHotEncoder below.
        if self.categorical_imputer is None:
            from sklearn.preprocessing import LabelEncoder
            for col in self.categorical_cols:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str)) if fit else encoder.transform(X[col].astype(str))
            return X
        values = self.categorical_imputer.transform(X[self.categorical_cols])
        if fit:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
            encoded = self.encoder.fit_transform(values)
        else:
            if self.encoder is None:
                raise ValueError("Categorical encoder not fitted")
            encoded = self.encoder.transform(values)
        encoded_names = self.encoder.get_feature_names_out(self.categorical_cols)
        encoded_df = pd.DataFrame(encoded, index=X.index, columns=encoded_names)
        numeric_df = X[self.numerical_cols].copy() if self.numerical_cols else pd.DataFrame(index=X.index)
        return pd.concat([numeric_df, encoded_df], axis=1)

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if not self.config.get("scale_features", True) or not self.numerical_cols:
            return X.astype(np.float64)
        X = X.copy()
        X.loc[:, self.numerical_cols] = X[self.numerical_cols].astype(float)
        method = self.config.get("scaling_method", "StandardScaler")
        scaler_cls = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}.get(method)
        if scaler_cls is None:
            raise ValueError(f"Unknown scaling method: {method}")
        if fit:
            self.scaler = scaler_cls()
            numeric_values = self.scaler.fit_transform(X[self.numerical_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            numeric_values = self.scaler.transform(X[self.numerical_cols])
        numeric_df = pd.DataFrame(numeric_values, index=X.index, columns=self.numerical_cols)
        other_df = X.drop(columns=self.numerical_cols)
        return pd.concat([numeric_df, other_df], axis=1).astype(np.float64)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting preprocessing (fit_transform)...")
        X = X.copy()
        self.identify_columns(X)
        X = self._impute(X, fit=True)
        X = self.remove_duplicates(X)
        self._fit_outlier_bounds(X)
        X = self._remove_training_outliers(X)
        X = self.encode_categorical(X, fit=True)
        X = self.scale_features(X, fit=True)
        self.output_columns = X.columns.tolist()
        self.fitted = True
        logger.info("Preprocessing complete. Shape: %s", X.shape)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        X = self._impute(X.copy(), fit=False)
        X = self.encode_categorical(X, fit=False)
        X = self.scale_features(X, fit=False)
        X = X.reindex(columns=self.output_columns, fill_value=0.0)
        return X.astype(np.float64)
