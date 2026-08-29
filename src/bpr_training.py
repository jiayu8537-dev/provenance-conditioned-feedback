from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .data_loading import DataBundle


@njit(fastmath=True)
def _bpr_epoch(U, V, users, positives, negatives, order, learning_rate, regularization):
    loss = 0.0
    dimension = U.shape[1]
    for index in range(order.shape[0]):
        row = order[index]
        user = users[row]
        positive = positives[row]
        negative = negatives[row]
        score = 0.0
        for factor in range(dimension):
            score += U[user, factor] * (V[positive, factor] - V[negative, factor])
        derivative = (
            0.0
            if score > 20
            else (1.0 if score < -20 else 1.0 / (1.0 + np.exp(score)))
        )
        loss += np.log1p(np.exp(-min(max(score, -20.0), 20.0)))
        for factor in range(dimension):
            user_value = U[user, factor]
            positive_value = V[positive, factor]
            negative_value = V[negative, factor]
            U[user, factor] = user_value + learning_rate * (
                derivative * (positive_value - negative_value)
                - regularization * user_value
            )
            V[positive, factor] = positive_value + learning_rate * (
                derivative * user_value - regularization * positive_value
            )
            V[negative, factor] = negative_value + learning_rate * (
                -derivative * user_value - regularization * negative_value
            )
    return loss / order.shape[0]


def sample_negatives(
    users: np.ndarray,
    full_keys: np.ndarray,
    n_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, n_items, size=len(users), dtype=np.int32)
    while True:
        keys = users.astype(np.int64) * n_items + negatives.astype(np.int64)
        positions = np.searchsorted(full_keys, keys)
        clamped = np.minimum(positions, len(full_keys) - 1)
        bad = (positions < len(full_keys)) & (full_keys[clamped] == keys)
        if not bad.any():
            return negatives
        negatives[bad] = rng.integers(0, n_items, size=bad.sum(), dtype=np.int32)


def train_full_history_bpr(
    data: DataBundle,
    dimension: int,
    learning_rate: float,
    regularization: float,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (data.n_users, dimension)).astype(np.float32)
    V = rng.normal(0, 0.01, (data.n_items, dimension)).astype(np.float32)
    logs = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(data.all_users)).astype(np.int32)
        negatives = sample_negatives(
            data.all_users, data.full_keys, data.n_items, rng
        )
        loss = _bpr_epoch(
            U,
            V,
            data.all_users,
            data.all_items,
            negatives,
            order,
            learning_rate,
            regularization,
        )
        logs.append({"epoch": epoch, "loss": float(loss)})
    return U, V, pd.DataFrame(logs)


def load_or_train(
    data: DataBundle,
    model_path: Path,
    log_path: Path,
    manifest_path: Path,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    expected = {
        "n_users": data.n_users,
        "n_items": data.n_items,
        "dimension": int(config["dimension"]),
        "learning_rate": float(config["learning_rate"]),
        "regularization": float(config["regularization"]),
        "epochs": int(config["epochs"]),
        "seed": int(config["seed"]),
        "history": "all rows of leave-last-two split; simulation begins after full observed history",
    }
    force_retrain = os.environ.get("JIIS_FORCE_RETRAIN", "0") == "1"
    if not force_retrain and model_path.exists() and manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current == expected:
            model = np.load(model_path)
            return model["U"].astype(np.float32), model["V"].astype(np.float32)
    U, V, logs = train_full_history_bpr(
        data=data,
        dimension=expected["dimension"],
        learning_rate=expected["learning_rate"],
        regularization=expected["regularization"],
        epochs=expected["epochs"],
        seed=expected["seed"],
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, U=U, V=V)
    logs.to_csv(log_path, index=False)
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return U, V
