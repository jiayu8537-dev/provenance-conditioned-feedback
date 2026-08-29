from __future__ import annotations

import numpy as np

from .choice_model import oracle_importance_weight
from .quota_reranker import hard_quota_rerank


def select_display(
    intervention: str,
    candidates: np.ndarray,
    raw_scores: np.ndarray,
    groups: np.ndarray,
    round_index: int,
    top_k: int,
) -> np.ndarray:
    if intervention in {"quota_reranking", "combined"}:
        return hard_quota_rerank(candidates, raw_scores, groups, round_index, top_k)
    order = np.argsort(-raw_scores, kind="stable")[:top_k]
    return candidates[order].astype(np.int32)


def feedback_weight(
    intervention: str,
    anchored_scores: np.ndarray,
    position: np.ndarray,
    true_effects: np.ndarray,
    clicked_position: int,
    misspecification: float,
    outside_utility: float,
) -> float:
    if intervention not in {"oracle_feedback_correction", "combined"}:
        return 1.0
    return oracle_importance_weight(
        anchored_scores,
        position,
        true_effects,
        clicked_position,
        misspecification,
        outside_utility,
    )
