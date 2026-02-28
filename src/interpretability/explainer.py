"""SHAP and LIME explainers for clinical models."""
import shap
import numpy as np
from typing import Dict, List


class SHAPExplainer:
    """SHAP-based model explainer."""
    
    def __init__(self, model, X_background):
        """Initialize with trained model and background data."""
        self.model = model
        self.explainer = shap.Explainer(model.model, X_background)
    
    def explain_instance(self, X_instance) -> Dict[str, float]:
        """Get SHAP values for single instance."""
        shap_values = self.explainer(X_instance)
        
        # Get feature names
        if self.model.feature_names:
            feature_names = self.model.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(X_instance.shape[1])]
        
        # Create dictionary of feature: shap_value
        shap_dict = {}
        for i, fname in enumerate(feature_names):
            shap_dict[fname] = float(shap_values.values[0][i])
        
        return shap_dict
    
    def explain_batch(self, X_batch) -> List[Dict[str, float]]:
        """Get SHAP values for batch of instances."""
        shap_values = self.explainer(X_batch)
        
        feature_names = self.model.feature_names or [
            f"feature_{i}" for i in range(X_batch.shape[1])
        ]
        
        results = []
        for i in range(len(X_batch)):
            shap_dict = {
                fname: float(shap_values.values[i][j])
                for j, fname in enumerate(feature_names)
            }
            results.append(shap_dict)
        
        return results
    
    def global_importance(self, X_sample) -> Dict[str, float]:
        """Compute global feature importance."""
        shap_values = self.explainer(X_sample)
        
        feature_names = self.model.feature_names or [
            f"feature_{i}" for i in range(X_sample.shape[1])
        ]
        
        # Mean absolute SHAP values
        importance = {}
        for i, fname in enumerate(feature_names):
            importance[fname] = float(np.abs(shap_values.values[:, i]).mean())
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance


def rank_features(mean_abs_contrib: dict[str, float]) -> list[str]:
    """Rank features by contribution (compatibility with existing code)."""
    return [k for k, _ in sorted(mean_abs_contrib.items(), key=lambda item: item[1], reverse=True)]
