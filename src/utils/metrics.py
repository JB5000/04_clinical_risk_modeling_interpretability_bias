"""Metrics computation utilities."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score
)
from typing import Dict


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive set of classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metric name to value
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'specificity': compute_specificity(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Additional rates
    metrics['positive_rate'] = (tp + fp) / len(y_true)
    metrics['negative_rate'] = (tn + fn) / len(y_true)
    
    return metrics


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0
