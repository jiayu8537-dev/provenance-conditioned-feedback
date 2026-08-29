from __future__ import annotations

import math

import numpy as np


HISTORICAL_REPLAY = False


def update_pair(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user: int,
    positive: int,
    negative: int,
    learning_rate: float,
    regularization: float,
    weight: float = 1.0,
    steps: int = 3,
) -> int:
    for _ in range(steps):
        user_value = user_embeddings[user].copy()
        positive_value = item_embeddings[positive].copy()
        negative_value = item_embeddings[negative].copy()
        score = float(user_value @ (positive_value - negative_value))
        derivative = 1.0 / (1.0 + math.exp(max(min(score, 20.0), -20.0)))
        rate = learning_rate * weight
        user_embeddings[user] = user_value + rate * (
            derivative * (positive_value - negative_value)
            - regularization * user_value
        )
        item_embeddings[positive] = positive_value + rate * (
            derivative * user_value - regularization * positive_value
        )
        item_embeddings[negative] = negative_value + rate * (
            -derivative * user_value - regularization * negative_value
        )
    return steps


def update_batch_synchronous(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    users: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    learning_rate: float,
    regularization: float,
    weights: np.ndarray | None = None,
    steps: int = 3,
) -> tuple[int, int]:
    """Apply simultaneous BPR gradients for all accepted events in a round.

    Each pass evaluates one gradient contribution per accepted event against
    the same pre-update parameter state. Contributions are summed by embedding
    index and applied synchronously, so event order cannot affect the result.
    The return values are event-gradient evaluations and parameter-update
    operations, respectively.
    """
    users = np.asarray(users, dtype=np.int32)
    positives = np.asarray(positives, dtype=np.int32)
    negatives = np.asarray(negatives, dtype=np.int32)
    if not (len(users) == len(positives) == len(negatives)):
        raise ValueError("users, positives, and negatives must have equal length")
    if weights is None:
        weights = np.ones(len(users), dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if len(weights) != len(users):
            raise ValueError("weights must match the number of accepted events")
    if len(users) == 0 or steps <= 0:
        return 0, 0

    for _ in range(steps):
        user_values = user_embeddings[users].copy()
        positive_values = item_embeddings[positives].copy()
        negative_values = item_embeddings[negatives].copy()
        scores = np.einsum(
            "nd,nd->n",
            user_values,
            positive_values - negative_values,
            optimize=True,
        )
        derivatives = 1.0 / (
            1.0 + np.exp(np.clip(scores, -20.0, 20.0))
        )
        rates = (float(learning_rate) * weights).reshape(-1, 1)
        user_deltas = rates * (
            derivatives.reshape(-1, 1) * (positive_values - negative_values)
            - float(regularization) * user_values
        )
        positive_deltas = rates * (
            derivatives.reshape(-1, 1) * user_values
            - float(regularization) * positive_values
        )
        negative_deltas = rates * (
            -derivatives.reshape(-1, 1) * user_values
            - float(regularization) * negative_values
        )
        np.add.at(user_embeddings, users, user_deltas)
        np.add.at(item_embeddings, positives, positive_deltas)
        np.add.at(item_embeddings, negatives, negative_deltas)

    return int(steps * len(users)), int(steps)
