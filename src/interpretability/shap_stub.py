"""Minimal SHAP-like feature ranking placeholder."""


def rank_features(mean_abs_contrib: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(mean_abs_contrib.items(), key=lambda item: item[1], reverse=True)]
