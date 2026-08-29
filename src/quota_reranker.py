from __future__ import annotations

import numpy as np


QUOTA_CYCLE = np.array([[7, 7, 6], [7, 6, 7], [6, 7, 7]], dtype=np.int16)


def quota_for_round(round_index: int) -> np.ndarray:
    """Return quotas in fixed order: AI, Human-AI, Human."""
    return QUOTA_CYCLE[round_index % 3].copy()


def hard_quota_rerank(
    candidates: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    round_index: int,
    top_k: int = 20,
) -> np.ndarray:
    """Select highest scores within quota, then fill shortages globally by score."""
    if top_k != 20:
        raise ValueError("The confirmatory rotating quota is defined for Top-20")
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    selected_set: set[int] = set()
    for group, quota in enumerate(quota_for_round(round_index)):
        for index in order:
            item = int(candidates[index])
            if groups[item] == group and item not in selected_set:
                selected.append(item)
                selected_set.add(item)
                if sum(groups[value] == group for value in selected) == quota:
                    break
    # Documented fallback: highest-scoring remaining candidate irrespective of group.
    for index in order:
        item = int(candidates[index])
        if item not in selected_set:
            selected.append(item)
            selected_set.add(item)
        if len(selected) == top_k:
            break
    if len(selected) != top_k:
        raise ValueError("Fewer unique candidates than requested Top-20")
    score_lookup = {int(item): float(score) for item, score in zip(candidates, scores)}
    return np.array(
        sorted(selected, key=lambda item: score_lookup[item], reverse=True), dtype=np.int32
    )
