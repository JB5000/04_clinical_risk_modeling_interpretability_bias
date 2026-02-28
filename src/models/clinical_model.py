"""Clinical risk model implementations."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from typing import Optional
import joblib


class ClinicalRiskModel:
    """Base clinical risk prediction model."""
    
    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.version = "v1.0.0"
        
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """Train the model."""
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """Predict classes."""
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'version': self.version,
            'model_type': self.model_type
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.version = data['version']
        return instance
