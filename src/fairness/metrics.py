"""Fairness metrics for binary clinical risk predictions."""


def positive_rate(predictions: list[int]) -> float:
    if not predictions:
        return 0.0
    return round(sum(predictions) / len(predictions), 4)


def demographic_parity_gap(group_a_preds: list[int], group_b_preds: list[int]) -> float:
    return round(abs(positive_rate(group_a_preds) - positive_rate(group_b_preds)), 4)
