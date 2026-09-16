"""
Individual classifiers for NIDS
"""
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from models.base_model import BaseNIDSModel
from utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostNIDSModel(BaseNIDSModel):
    """XGBoost classifier for NIDS"""
    
    def __init__(self, config: dict):
        """Initialize XGBoost model"""
        model = XGBClassifier(**config)
        super().__init__("XGBoost", model, config)
    
    def train(self, X_train, y_train):
        """Train XGBoost"""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} trained successfully")
    
    def predict(self, X):
        """Predict labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)


class LightGBMNIDSModel(BaseNIDSModel):
    """LightGBM classifier for NIDS"""
    
    def __init__(self, config: dict):
        """Initialize LightGBM model"""
        model = LGBMClassifier(**config)
        super().__init__("LightGBM", model, config)
    
    def train(self, X_train, y_train):
        """Train LightGBM"""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} trained successfully")
    
    def predict(self, X):
        """Predict labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)


class RandomForestNIDSModel(BaseNIDSModel):
    """Random Forest classifier for NIDS"""
    
    def __init__(self, config: dict):
        """Initialize Random Forest model"""
        model = RandomForestClassifier(**config)
        super().__init__("RandomForest", model, config)
    
    def train(self, X_train, y_train):
        """Train Random Forest"""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} trained successfully")
    
    def predict(self, X):
        """Predict labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)


class SVMNIDSModel(BaseNIDSModel):
    """SVM classifier for NIDS"""
    
    def __init__(self, config: dict):
        """Initialize SVM model"""
        model = SVC(**config)
        super().__init__("SVM", model, config)
    
    def train(self, X_train, y_train):
        """Train SVM"""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} trained successfully")
    
    def predict(self, X):
        """Predict labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities

        Note: requires the SVC to be constructed with probability=True
        (see config/settings.py SVM_CONFIG). Using decision_function()
        here would return an unbounded score rather than a [0,1]
        probability, which corrupts soft-voting ensemble averages.
        """
        return self.model.predict_proba(X)


class LogisticRegressionNIDSModel(BaseNIDSModel):
    """Logistic Regression classifier for NIDS"""
    
    def __init__(self, config: dict):
        """Initialize Logistic Regression model"""
        model = LogisticRegression(**config)
        super().__init__("LogisticRegression", model, config)
    
    def train(self, X_train, y_train):
        """Train Logistic Regression"""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} trained successfully")
    
    def predict(self, X):
        """Predict labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
