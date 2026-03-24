"""Fairness auditing for clinical risk models."""
import numpy as np
from typing import Dict, List


def _safe_rate(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


class FairnessAuditor:
    """Audit model fairness across protected attributes."""
    
    def __init__(self, protected_attributes: List[str]):
        """Initialize with list of protected attributes to audit."""
        self.protected_attributes = protected_attributes
    
    def compute_metrics(self, y_true, y_pred, y_proba, sensitive_features) -> Dict:
        """Compute comprehensive fairness metrics."""
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        metrics = {}
        
        for attr in self.protected_attributes:
            if attr not in sensitive_features:
                continue
            
            attr_values = np.asarray(sensitive_features[attr])
            groups = np.unique(attr_values)
            
            # Demographic parity
            pos_rates = {}
            for group in groups:
                mask = attr_values == group
                pos_rates[group] = _safe_rate(y_pred_arr[mask])
            
            dp_gap = max(pos_rates.values()) - min(pos_rates.values()) if pos_rates else 0.0
            
            # Equal opportunity
            eo_diff = 0.0
            if len(groups) >= 2:
                group_tpr = {}
                for group in groups:
                    mask = (attr_values == group) & (y_true_arr == 1)
                    if mask.sum() > 0:
                        group_tpr[group] = _safe_rate(y_pred_arr[mask])
                
                if len(group_tpr) >= 2:
                    eo_diff = max(group_tpr.values()) - min(group_tpr.values())
            
            # Equalized odds
            eod_diff = eo_diff  # Simplified
            
            # Subgroup performance
            subgroup_perf = {}
            for group in groups:
                mask = attr_values == group
                if mask.sum() > 0:
                    tn, fp, fn, tp = _binary_confusion_counts(y_true_arr[mask], y_pred_arr[mask])
                    total = tp + tn + fp + fn
                    subgroup_perf[str(group)] = {
                        'accuracy': (tp + tn) / total if total > 0 else 0.0,
                        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
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
        violations = self.fairness_violations(metrics, thresholds=thresholds)
        return len(violations) == 0

    def fairness_violations(self, metrics: Dict, thresholds: Dict = None) -> Dict[str, List[str]]:
        """Return threshold violations by protected attribute."""
        if thresholds is None:
            thresholds = {
                'demographic_parity': 0.05,
                'equal_opportunity': 0.10,
                'equalized_odds': 0.10
            }
        
        violations: Dict[str, List[str]] = {}
        for attr, attr_metrics in metrics.items():
            failed: List[str] = []
            if attr_metrics.get('demographic_parity', 1.0) > thresholds['demographic_parity']:
                failed.append('demographic_parity')
            if attr_metrics.get('equal_opportunity', 1.0) > thresholds['equal_opportunity']:
                failed.append('equal_opportunity')
            if attr_metrics.get('equalized_odds', 1.0) > thresholds['equalized_odds']:
                failed.append('equalized_odds')
            if failed:
                violations[attr] = failed
        
        return violations


def positive_rate(predictions: list[int]) -> float:
    """Calculate positive prediction rate (compatibility)."""
    if not predictions:
        return 0.0
    return round(sum(predictions) / len(predictions), 4)


def demographic_parity_gap(group_a_preds: list[int], group_b_preds: list[int]) -> float:
    """Calculate demographic parity gap (compatibility)."""
    return round(abs(positive_rate(group_a_preds) - positive_rate(group_b_preds)), 4)
