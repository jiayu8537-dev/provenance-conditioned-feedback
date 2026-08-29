from __future__ import annotations

import numpy as np


def candidate_anchor(scores: np.ndarray, minimum_sd: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    finite = np.where(np.isfinite(scores), scores, np.nan)
    mean = np.nanmean(finite, axis=1, keepdims=True)
    sd = np.nanstd(finite, axis=1, keepdims=True)
    sd = np.where((~np.isfinite(sd)) | (sd < minimum_sd), 1.0, sd)
    return mean.astype(np.float32), sd.astype(np.float32)


def apply_anchor(scores: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((scores - mean) / sd).astype(np.float32)

