from src.fairness.metrics import demographic_parity_gap
from src.interpretability.shap_stub import rank_features


def test_demographic_parity_gap_nonzero() -> None:
    gap = demographic_parity_gap([1, 1, 0, 1], [0, 0, 1, 0])
    assert gap == 0.5


def test_rank_features_descending_importance() -> None:
    ranked = rank_features({"age": 0.14, "hr": 0.22, "bp": 0.08})
    assert ranked == ["hr", "age", "bp"]
