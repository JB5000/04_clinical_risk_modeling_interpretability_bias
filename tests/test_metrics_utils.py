import math

import numpy as np

from src.utils.metrics import compute_all_metrics, compute_specificity


def test_compute_all_metrics_handles_single_class_truth() -> None:
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    y_proba = np.array([0.01, 0.05, 0.6, 0.2, 0.15])

    metrics = compute_all_metrics(y_true, y_pred, y_proba)

    assert metrics["true_negatives"] == 4
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 0
    assert math.isnan(float(metrics["auc"]))
    assert math.isnan(float(metrics["average_precision"]))


def test_compute_all_metrics_thresholds_probability_like_predictions() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred_proba_like = np.array([0.2, 0.8, 0.6, 0.3, 0.4, 0.1])

    metrics = compute_all_metrics(y_true, y_pred_proba_like)

    assert metrics["accuracy"] == 5 / 6
    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1


def test_compute_specificity_returns_zero_when_no_negatives() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    assert compute_specificity(y_true, y_pred) == 0.0
