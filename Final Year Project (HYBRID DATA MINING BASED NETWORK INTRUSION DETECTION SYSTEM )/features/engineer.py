"""Leakage-safe feature engineering for NIDS."""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from utils.logger import get_logger

logger = get_logger(__name__)


def safe_f_classif(X, y):
    """f_classif variant that assigns zero score to constant columns."""
    X = np.asarray(X, dtype=float)
    scores = np.zeros(X.shape[1], dtype=float)
    pvalues = np.ones(X.shape[1], dtype=float)
    variable = np.nanvar(X, axis=0) > 0
    if variable.any():
        s, p = f_classif(X[:, variable], y)
        scores[variable] = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        pvalues[variable] = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    return scores, pvalues


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.selector = None
        self.selected_features = None
        self.kept_after_variance = None
        self.kept_after_correlation = None
        self.interaction_pairs = []
        self.polynomial_specs = []

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series, method: str = "f_classif") -> dict:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        if method == "f_classif":
            scores = np.zeros(X.shape[1], dtype=float)
            variable = X.nunique(dropna=False).to_numpy() > 1
            if variable.any():
                scores[variable], _ = safe_f_classif(X.loc[:, variable], y)
        elif method == "mutual_info":
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return dict(sorted(zip(X.columns, scores), key=lambda item: item[1], reverse=True))

    def _remove_low_variance_fit(self, X, threshold=0.0):
        variance = X.var(ddof=0)
        kept = variance[variance > threshold].index.tolist()
        if not kept:
            raise ValueError("All features were removed by variance filtering")
        self.kept_after_variance = kept
        return X[kept]

    def _remove_correlation_fit(self, X, threshold=0.95):
        if X.shape[1] < 2:
            self.kept_after_correlation = X.columns.tolist()
            return X
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if (upper[c] > threshold).any()]
        kept = [c for c in X.columns if c not in drop]
        self.kept_after_correlation = kept
        return X[kept]

    def _add_derived_features_fit(self, X):
        X = X.copy()
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        top_n = min(5, len(numeric))
        top = numeric[:top_n]
        self.interaction_pairs = []
        if self.config.get("create_interaction_features", False):
            for i, a in enumerate(top):
                for b in top[i + 1:]:
                    name = f"{a}_x_{b}"
                    X[name] = X[a] * X[b]
                    self.interaction_pairs.append((a, b, name))
        self.polynomial_specs = []
        if self.config.get("polynomial_features", False):
            for col in top:
                name = f"{col}_pow_2"
                X[name] = X[col] ** 2
                self.polynomial_specs.append((col, name))
        return X

    def _add_derived_features(self, X):
        X = X.copy()
        for a, b, name in self.interaction_pairs:
            if a in X and b in X:
                X[name] = X[a] * X[b]
        for col, name in self.polynomial_specs:
            if col in X:
                X[name] = X[col] ** 2
        return X

    def remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Public low-variance filter retained for compatibility and experimentation."""
        variance = X.var(ddof=0)
        return X.loc[:, variance >= threshold].copy()

    def select_features(self, X: pd.DataFrame, y: pd.Series, k="all", fit: bool = True) -> pd.DataFrame:
        """Public feature-selection helper."""
        if k == "all":
            self.selected_features = X.columns.tolist()
            return X.copy()
        k = min(int(k), X.shape[1])
        if fit:
            self.selector = SelectKBest(score_func=safe_f_classif, k=k)
            arr = self.selector.fit_transform(X, y)
            self.selected_features = X.columns[self.selector.get_support()].tolist()
        else:
            if self.selector is None:
                raise ValueError("Selector not fitted")
            arr = self.selector.transform(X)
        return pd.DataFrame(arr, index=X.index, columns=self.selected_features)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        logger.info("Starting feature engineering (fit_transform)...")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        X = self._remove_low_variance_fit(X, threshold=0.0)
        X = self._remove_correlation_fit(X, threshold=0.95)
        X = self._add_derived_features_fit(X)
        k = self.config.get("k_features", "all")
        if k != "all":
            k = min(int(k), X.shape[1])
            self.selector = SelectKBest(score_func=safe_f_classif, k=k)
            arr = self.selector.fit_transform(X, y)
            self.selected_features = X.columns[self.selector.get_support()].tolist()
            X = pd.DataFrame(arr, index=X.index, columns=self.selected_features)
        else:
            self.selected_features = X.columns.tolist()
        return X.astype(np.float64)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            raise ValueError("Feature engineer not fitted. Call fit_transform first.")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        X = X.reindex(columns=self.kept_after_variance, fill_value=0.0)
        X = X.reindex(columns=self.kept_after_correlation, fill_value=0.0)
        X = self._add_derived_features(X)
        if self.selector is not None:
            X = X.reindex(columns=self.selector.feature_names_in_, fill_value=0.0)
            X = pd.DataFrame(self.selector.transform(X), index=X.index, columns=self.selected_features)
        else:
            X = X.reindex(columns=self.selected_features, fill_value=0.0)
        return X.astype(np.float64)
