"""Metrics computation utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _coerce_binary(values: np.ndarray, *, allow_threshold: bool, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values.")

    unique = np.unique(arr)
    if unique.size <= 2:
        if unique.size == 2 and not np.array_equal(unique, np.array([0.0, 1.0])):
            low, high = unique[0], unique[1]
            return np.where(arr == high, 1, 0).astype(int)
        if unique.size == 1:
            return (arr > 0).astype(int)
        return arr.astype(int)

    if allow_threshold:
        return (arr >= 0.5).astype(int)

    raise ValueError(f"{name} must be binary or contain exactly 2 classes.")


def _binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def _matthews_corrcoef(tn: int, fp: int, fn: int, tp: int) -> float:
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return _safe_div(numerator, denominator)


def _cohen_kappa(tn: int, fp: int, fn: int, tp: int) -> float:
    total = tn + fp + fn + tp
    if total == 0:
        return 0.0

    observed = _safe_div(tp + tn, total)
    true_pos_rate = _safe_div(tp + fn, total)
    true_neg_rate = _safe_div(tn + fp, total)
    pred_pos_rate = _safe_div(tp + fp, total)
    pred_neg_rate = _safe_div(tn + fn, total)
    expected = (true_pos_rate * pred_pos_rate) + (true_neg_rate * pred_neg_rate)
    return _safe_div(observed - expected, 1.0 - expected) if expected < 1.0 else 0.0


def _roc_auc_score_binary(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    positive_scores = y_scores[y_true == 1]
    negative_scores = y_scores[y_true == 0]
    n_pos = positive_scores.size
    n_neg = negative_scores.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    wins = (positive_scores[:, None] > negative_scores[None, :]).sum()
    ties = (positive_scores[:, None] == negative_scores[None, :]).sum()
    return float((wins + 0.5 * ties) / (n_pos * n_neg))


def _average_precision_binary(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_scores, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision[sorted_true == 1]) / positives)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float | int]:
    """Compute a robust set of binary classification metrics."""
    y_true_binary = _coerce_binary(y_true, allow_threshold=False, name="y_true")
    y_pred_binary = _coerce_binary(y_pred, allow_threshold=True, name="y_pred")
    if y_true_binary.shape[0] != y_pred_binary.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    tn, fp, fn, tp = _binary_confusion_counts(y_true_binary, y_pred_binary)
    total = y_true_binary.shape[0]

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    metrics: Dict[str, float | int] = {
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1_score": _safe_div(2 * precision * recall, precision + recall),
        "specificity": specificity,
        "mcc": _matthews_corrcoef(tn, fp, fn, tp),
        "kappa": _cohen_kappa(tn, fp, fn, tp),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "positive_rate": _safe_div(tp + fp, total),
        "negative_rate": _safe_div(tn + fn, total),
    }

    if y_proba is not None:
        y_scores = np.asarray(y_proba, dtype=float).ravel()
        if y_scores.shape[0] != y_true_binary.shape[0]:
            raise ValueError("y_true and y_proba must have the same length.")
        metrics["auc"] = _roc_auc_score_binary(y_true_binary, y_scores)
        metrics["average_precision"] = _average_precision_binary(y_true_binary, y_scores)

    return metrics


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    y_true_binary = _coerce_binary(y_true, allow_threshold=False, name="y_true")
    y_pred_binary = _coerce_binary(y_pred, allow_threshold=True, name="y_pred")
    tn, fp, _, _ = _binary_confusion_counts(y_true_binary, y_pred_binary)
    return _safe_div(tn, tn + fp)
