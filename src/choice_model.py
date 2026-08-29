from __future__ import annotations

import numpy as np


def position_effect(top_k: int, scale: float = 0.8) -> np.ndarray:
    ranks = np.arange(1, top_k + 1, dtype=np.float64)
    return scale / np.log2(ranks + 1.0)


def softmax_with_outside(utilities: np.ndarray, outside_utility: float = 0.5) -> np.ndarray:
    # Preserve the original executable category order: outside first, then ranks 1..K.
    values = np.insert(np.asarray(utilities, dtype=np.float64), 0, outside_utility)
    values -= values.max()
    probabilities = np.exp(values)
    return probabilities / probabilities.sum()


def choice_from_uniform(probabilities: np.ndarray, uniform: float) -> int:
    """Return -1 for no click, otherwise the displayed zero-based position."""
    index = int(np.searchsorted(np.cumsum(probabilities), uniform, side="right"))
    return -1 if index == 0 or index >= len(probabilities) else index - 1


def oracle_importance_weight(
    anchored_scores: np.ndarray,
    position: np.ndarray,
    true_effects: np.ndarray,
    clicked_position: int,
    misspecification: float,
    outside_utility: float = 0.5,
    lower: float = 0.25,
    upper: float = 4.0,
) -> float:
    """Oracle ratio P(effect=(1-c)delta)/P(effect=delta) for the observed click."""
    if clicked_position < 0:
        return 1.0
    true_p = softmax_with_outside(
        anchored_scores + position + true_effects, outside_utility
    )[clicked_position + 1]
    residual_p = softmax_with_outside(
        anchored_scores + position + (1.0 - misspecification) * true_effects,
        outside_utility,
    )[clicked_position + 1]
    return float(np.clip(residual_p / max(true_p, 1e-12), lower, upper))
