import numpy as np

from src.fairness.auditor import FairnessAuditor
from src.fairness.metrics import demographic_parity_gap
from src.interpretability.shap_stub import rank_features


def test_demographic_parity_gap_nonzero() -> None:
    gap = demographic_parity_gap([1, 1, 0, 1], [0, 0, 1, 0])
    assert gap == 0.5


def test_rank_features_descending_importance() -> None:
    ranked = rank_features({"age": 0.14, "hr": 0.22, "bp": 0.08})
    assert ranked == ["hr", "age", "bp"]


def test_fairness_violations_identifies_failing_attribute() -> None:
    auditor = FairnessAuditor(protected_attributes=["gender"])
    metrics = {
        "gender": {
            "demographic_parity": 0.12,
            "equal_opportunity": 0.06,
            "equalized_odds": 0.05,
            "subgroups": {},
        }
    }
    violations = auditor.fairness_violations(metrics)
    assert violations == {"gender": ["demographic_parity"]}
    assert auditor.passes_fairness_threshold(metrics) is False


def test_auditor_handles_subgroup_with_single_outcome_class() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    sensitive_features = {
        "race": np.array(["A", "A", "A", "B", "B", "B"])
    }
    auditor = FairnessAuditor(protected_attributes=["race"])
    metrics = auditor.compute_metrics(y_true, y_pred, y_proba=None, sensitive_features=sensitive_features)
    assert "race" in metrics
    assert set(metrics["race"]["subgroups"]) == {"A", "B"}
