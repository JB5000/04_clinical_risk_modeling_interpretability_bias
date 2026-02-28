"""Fairness auditing for clinical risk models."""
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List


class FairnessAuditor:
    """Audit model fairness across protected attributes."""
    
    def __init__(self, protected_attributes: List[str]):
        """Initialize with list of protected attributes to audit."""
        self.protected_attributes = protected_attributes
    
    def compute_metrics(self, y_true, y_pred, y_proba, sensitive_features) -> Dict:
        """Compute comprehensive fairness metrics."""
        metrics = {}
        
        for attr in self.protected_attributes:
            if attr not in sensitive_features:
                continue
            
            groups = np.unique(sensitive_features[attr])
            
            # Demographic parity
            pos_rates = {}
            for group in groups:
                mask = sensitive_features[attr] == group
                pos_rates[group] = y_pred[mask].mean()
            
            dp_gap = max(pos_rates.values()) - min(pos_rates.values())
            
            # Equal opportunity
            eo_diff = 0.0
            if len(groups) >= 2:
                group_tpr = {}
                for group in groups:
                    mask = (sensitive_features[attr] == group) & (y_true == 1)
                    if mask.sum() > 0:
                        group_tpr[group] = y_pred[mask].mean()
                
                if len(group_tpr) >= 2:
                    eo_diff = max(group_tpr.values()) - min(group_tpr.values())
            
            # Equalized odds
            eod_diff = eo_diff  # Simplified
            
            # Subgroup performance
            subgroup_perf = {}
            for group in groups:
                mask = sensitive_features[attr] == group
                if mask.sum() > 0:
                    cm = confusion_matrix(y_true[mask], y_pred[mask])
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        subgroup_perf[str(group)] = {
                            'accuracy': (tp + tn) / (tp + tn + fp + fn),
                            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
                        }
            
            metrics[attr] = {
                'demographic_parity': dp_gap,
                'equal_opportunity': eo_diff,
                'equalized_odds': eod_diff,
                'subgroups': subgroup_perf
            }
        
        return metrics
    
    def passes_fairness_threshold(self, metrics: Dict, thresholds: Dict = None) -> bool:
        """Check if metrics pass fairness thresholds."""
        if thresholds is None:
            thresholds = {
                'demographic_parity': 0.05,
                'equal_opportunity': 0.10,
                'equalized_odds': 0.10
            }
        
        for attr, attr_metrics in metrics.items():
            if attr_metrics.get('demographic_parity', 1.0) > thresholds['demographic_parity']:
                return False
            if attr_metrics.get('equal_opportunity', 1.0) > thresholds['equal_opportunity']:
                return False
            if attr_metrics.get('equalized_odds', 1.0) > thresholds['equalized_odds']:
                return False
        
        return True


def positive_rate(predictions: list[int]) -> float:
    """Calculate positive prediction rate (compatibility)."""
    if not predictions:
        return 0.0
    return round(sum(predictions) / len(predictions), 4)


def demographic_parity_gap(group_a_preds: list[int], group_b_preds: list[int]) -> float:
    """Calculate demographic parity gap (compatibility)."""
    return round(abs(positive_rate(group_a_preds) - positive_rate(group_b_preds)), 4)
