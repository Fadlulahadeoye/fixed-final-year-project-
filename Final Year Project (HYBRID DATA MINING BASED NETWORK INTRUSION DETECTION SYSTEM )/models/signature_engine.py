"""
Signature-based (misuse) detection engine for NIDS.

Unlike the ML ensemble (which generalizes from feature patterns), this engine
mines explicit, human-inspectable "signatures" directly from the training
data: value combinations that are near-perfectly associated with attacks
(the same idea as Apriori-style association rule mining referenced in the
literature review). At inference time it flags any record that exactly
matches a learned signature as a known attack, with very high precision by
construction (min_confidence), independent of the ML models.

This is deliberately simple and transparent — that's the point of a
signature engine: fast, explainable, zero false positives on the patterns
it knows, but blind to anything it hasn't seen (which is why it's paired
with the anomaly/ML ensemble in fusion.py rather than used alone).
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignatureRule:
    """A single mined signature: a set of column==value conditions -> Attack"""
    conditions: Dict[str, Any]
    support: float          # fraction of training data matching the conditions
    confidence: float       # P(attack | conditions) in training data
    match_count: int        # raw count of matching rows in training data

    def matches(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean mask of rows in X satisfying every condition in this rule"""
        mask = np.ones(len(X), dtype=bool)
        for col, val in self.conditions.items():
            if col not in X.columns:
                return np.zeros(len(X), dtype=bool)
            mask &= (X[col].values == val)
        return mask

    def __str__(self):
        cond_str = " AND ".join(f"{k}={v!r}" for k, v in self.conditions.items())
        return (f"IF {cond_str} THEN Attack "
                f"(support={self.support:.4f}, confidence={self.confidence:.4f}, "
                f"n={self.match_count})")


class SignatureEngine:
    """
    Misuse-detection engine that mines high-confidence attack signatures
    from labeled training data instead of relying on a hand-curated
    signature database.

    Config:
        categorical_columns: columns to mine single/combo signatures from
                              (default targets NSL-KDD-style categorical fields)
        binary_columns: 0/1 flag columns to include as single-value signatures
                         (e.g. 'land', 'root_shell', 'su_attempted')
        max_combo_size: how many columns a single signature may combine (1 or 2)
        min_support: minimum fraction of training rows a signature must cover
        min_confidence: minimum P(attack | conditions) required to keep a rule
        max_rules: cap on number of signatures kept (highest-confidence first)
    """

    DEFAULT_CATEGORICAL = ["protocol_type", "service", "flag"]
    DEFAULT_BINARY = ["land", "root_shell", "su_attempted", "is_guest_login"]

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.categorical_columns = config.get("categorical_columns", self.DEFAULT_CATEGORICAL)
        self.binary_columns = config.get("binary_columns", self.DEFAULT_BINARY)
        self.max_combo_size = config.get("max_combo_size", 2)
        self.min_support = config.get("min_support", 0.002)
        self.min_confidence = config.get("min_confidence", 0.98)
        self.max_rules = config.get("max_rules", 50)

        self.rules: List[SignatureRule] = []
        self.is_fitted = False

    # ------------------------------------------------------------------ #
    # Mining
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SignatureEngine":
        """
        Mine signatures from labeled training data.

        Args:
            X_train: training features (must include the configured
                     categorical/binary columns, pre- or post-preprocessing
                     as long as the raw category values are still present)
            y_train: binary labels (0 = normal, 1 = attack)
        """
        logger.info("Mining attack signatures from training data...")
        y = pd.Series(np.asarray(y_train)).reset_index(drop=True)
        X = X_train.reset_index(drop=True)
        n_total = len(X)

        candidate_rules: List[SignatureRule] = []

        # --- Single-column categorical signatures ---
        available_cat = [c for c in self.categorical_columns if c in X.columns]
        for col in available_cat:
            candidate_rules += self._mine_column_group(X, y, [col], n_total)

        # --- Single-column binary flag signatures (flag == 1 only) ---
        available_bin = [c for c in self.binary_columns if c in X.columns]
        for col in available_bin:
            candidate_rules += self._mine_binary_flag(X, y, col, n_total)

        # --- Multi-column combos (e.g. service + flag) ---
        if self.max_combo_size >= 2 and len(available_cat) >= 2:
            for col_pair in combinations(available_cat, 2):
                candidate_rules += self._mine_column_group(X, y, list(col_pair), n_total)

        # Keep the highest-confidence, highest-support rules, capped at max_rules
        candidate_rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        self.rules = self._deduplicate(candidate_rules)[: self.max_rules]
        self.is_fitted = True

        logger.info(f"Signature mining complete: kept {len(self.rules)} signatures "
                    f"(from {len(candidate_rules)} candidates)")
        for rule in self.rules[:10]:
            logger.info(f"  {rule}")
        if len(self.rules) > 10:
            logger.info(f"  ... and {len(self.rules) - 10} more")

        return self

    def _mine_column_group(self, X: pd.DataFrame, y: pd.Series,
                            cols: List[str], n_total: int) -> List[SignatureRule]:
        """Mine signatures over every observed value-combination of `cols`"""
        rules = []
        grouped = X.groupby(cols, observed=True)
        for group_key, group_df in grouped:
            idx = group_df.index
            n_match = len(idx)
            support = n_match / n_total
            if support < self.min_support:
                continue

            attack_rate = y.loc[idx].mean()
            if attack_rate < self.min_confidence:
                continue

            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            conditions = dict(zip(cols, group_key))

            rules.append(SignatureRule(
                conditions=conditions,
                support=support,
                confidence=attack_rate,
                match_count=n_match,
            ))
        return rules

    def _mine_binary_flag(self, X: pd.DataFrame, y: pd.Series,
                           col: str, n_total: int) -> List[SignatureRule]:
        """Mine a signature for a binary/rare flag column being set to 1"""
        idx = X.index[X[col] == 1]
        n_match = len(idx)
        if n_match == 0:
            return []
        support = n_match / n_total
        if support < self.min_support:
            return []
        attack_rate = y.loc[idx].mean()
        if attack_rate < self.min_confidence:
            return []
        return [SignatureRule(
            conditions={col: 1},
            support=support,
            confidence=attack_rate,
            match_count=n_match,
        )]

    @staticmethod
    def _deduplicate(rules: List[SignatureRule]) -> List[SignatureRule]:
        """Drop rules whose conditions are a superset of an already-kept,
        equally-confident rule (prefer the simpler/more general signature)."""
        kept: List[SignatureRule] = []
        for rule in rules:
            redundant = any(
                set(existing.conditions.items()) <= set(rule.conditions.items())
                for existing in kept
            )
            if not redundant:
                kept.append(rule)
        return kept

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def match_mask(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean array: True where any learned signature fires"""
        if not self.is_fitted:
            raise ValueError("SignatureEngine not fitted. Call fit() first.")
        if not self.rules:
            return np.zeros(len(X), dtype=bool)

        mask = np.zeros(len(X), dtype=bool)
        for rule in self.rules:
            mask |= rule.matches(X)
        return mask

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Standalone prediction: 1 where a signature matches, else 0.
        (For fusion use, prefer match_mask() so 'no match' can be treated as
        'defer to the anomaly engine' rather than a hard 'normal' verdict.)"""
        return self.match_mask(X).astype(int)

    def get_rules_summary(self) -> pd.DataFrame:
        """Rules as a DataFrame for reporting/appendix tables"""
        if not self.rules:
            return pd.DataFrame(columns=["conditions", "support", "confidence", "match_count"])
        return pd.DataFrame([
            {
                "conditions": rule.conditions,
                "support": rule.support,
                "confidence": rule.confidence,
                "match_count": rule.match_count,
            }
            for rule in self.rules
        ])
