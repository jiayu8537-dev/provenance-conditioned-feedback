from __future__ import annotations

import numpy as np
import pandas as pd


def add_amplification(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["panel", "assignment", "response_seed", "scenario", "intervention", "round"]
    wide = frame.pivot(index=keys, columns="branch", values="D").reset_index()
    wide["closed_minus_frozen"] = wide["closed"] - wide["frozen"]
    neutral = wide[wide.scenario == "zero"][
        ["panel", "assignment", "response_seed", "intervention", "round", "closed_minus_frozen"]
    ].rename(columns={"closed_minus_frozen": "neutral_drift"})
    merged = wide.merge(
        neutral,
        on=["panel", "assignment", "response_seed", "intervention", "round"],
        how="left",
        validate="many_to_one",
    )
    merged["AA"] = merged.closed_minus_frozen - merged.neutral_drift
    return merged


def hierarchical_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    """Independently resample panels, assignments, and responses with replacement."""
    rng = np.random.default_rng(seed)
    dimensions = ["panel", "assignment", "response_seed"]
    levels = {dimension: np.sort(frame[dimension].unique()) for dimension in dimensions}
    values = frame[value].to_numpy(dtype=float)
    estimates = np.empty(repetitions, dtype=float)
    # Vectorized frequency-weight bootstrap. Processing in chunks bounds memory
    # while preserving the same independent crossed resampling distribution.
    chunk_size = min(1000, repetitions)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        size = stop - start
        weights = np.ones((size, len(frame)), dtype=np.int32)
        for dimension in dimensions:
            dimension_levels = levels[dimension]
            level_lookup = {
                level: index for index, level in enumerate(dimension_levels)
            }
            row_indices = frame[dimension].map(level_lookup).to_numpy(dtype=int)
            counts = rng.multinomial(
                len(dimension_levels),
                np.full(len(dimension_levels), 1.0 / len(dimension_levels)),
                size=size,
            )
            weights *= counts[:, row_indices]
        denominators = weights.sum(axis=1)
        estimates[start:stop] = (weights @ values) / denominators
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def paired_difference(
    frame: pd.DataFrame,
    metric: str,
    treatment: str,
    control: str = "none",
) -> pd.DataFrame:
    keys = ["panel", "assignment", "response_seed", "scenario", "round", "branch"]
    subset = frame[frame.intervention.isin([treatment, control])]
    wide = subset.pivot(index=keys, columns="intervention", values=metric).reset_index()
    wide[f"delta_{metric}"] = wide[treatment] - wide[control]
    return wide
