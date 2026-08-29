from __future__ import annotations

import numpy as np


def discounted_exposure(items: np.ndarray, item_groups: np.ndarray) -> np.ndarray:
    weights = 1.0 / np.log2(np.arange(2, len(items) + 2, dtype=np.float64))
    return np.bincount(item_groups[items], weights=weights, minlength=3).astype(float)


def relative_exposure(exposure: np.ndarray, catalog_shares: np.ndarray) -> np.ndarray:
    shares = exposure / max(float(exposure.sum()), 1e-12)
    return shares / np.maximum(catalog_shares, 1e-12)


def human_ai_log_gap(exposure: np.ndarray, catalog_shares: np.ndarray) -> float:
    relative = relative_exposure(exposure, catalog_shares)
    return float(np.log(max(relative[2], 1e-12) / max(relative[0], 1e-12)))


def total_variation_exposure(exposure: np.ndarray, catalog_shares: np.ndarray) -> float:
    shares = exposure / max(float(exposure.sum()), 1e-12)
    return float(0.5 * np.abs(shares - catalog_shares).sum())


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    ordered = np.sort(values)
    n = len(ordered)
    return float((2 * np.dot(np.arange(1, n + 1), ordered) / ordered.sum() - n - 1) / n)


def cumulative_metrics(
    exposure_by_item: np.ndarray,
    clicks: int,
    impressions: int,
    clicked_utility_sum: float,
    item_groups: np.ndarray,
) -> dict[str, float]:
    group_exposure = np.bincount(
        item_groups, weights=exposure_by_item, minlength=3
    ).astype(float)
    catalog_shares = np.bincount(item_groups, minlength=3).astype(float)
    catalog_shares /= catalog_shares.sum()
    exposure_shares = group_exposure / max(float(group_exposure.sum()), 1e-12)
    return {
        "D": human_ai_log_gap(group_exposure, catalog_shares),
        "abs_D": abs(human_ai_log_gap(group_exposure, catalog_shares)),
        "exposure_tv": total_variation_exposure(group_exposure, catalog_shares),
        "exposure_share_ai": float(exposure_shares[0]),
        "exposure_share_human_ai": float(exposure_shares[1]),
        "exposure_share_human": float(exposure_shares[2]),
        "human_ai_share_gap_pp": float(
            100.0 * (exposure_shares[2] - exposure_shares[0])
        ),
        "ctr": clicks / max(impressions, 1),
        "candidate_anchored_utility": clicked_utility_sum / max(clicks, 1),
        "coverage": float(np.count_nonzero(exposure_by_item) / len(exposure_by_item)),
        "gini": gini(exposure_by_item),
    }
