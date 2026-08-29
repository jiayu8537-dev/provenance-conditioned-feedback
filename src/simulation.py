from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .choice_model import choice_from_uniform, position_effect, softmax_with_outside
from .exposure_metrics import cumulative_metrics, discounted_exposure
from .interventions import feedback_weight, select_display
from .online_bpr import update_batch_synchronous, update_pair
from .score_normalization import apply_anchor, candidate_anchor


@dataclass(frozen=True)
class SimulationConfig:
    rounds: int = 6
    top_k: int = 20
    online_steps: int = 3
    online_learning_rate: float = 0.01
    online_regularization: float = 0.0001
    outside_utility: float = 0.5
    position_scale: float = 0.8
    minimum_score_sd: float = 1e-6
    update_regime: str = "event_online"
    clicked_steps: int = 3
    replay_steps: int = 0


def _display_positions(candidates: np.ndarray, displayed: np.ndarray) -> np.ndarray:
    lookup = {int(item): idx for idx, item in enumerate(candidates)}
    return np.array([lookup[int(item)] for item in displayed], dtype=np.int32)


def simulate_paired(
    initial_user_embeddings: np.ndarray,
    initial_item_embeddings: np.ndarray,
    panel: np.ndarray,
    candidate_pools: np.ndarray,
    item_groups: np.ndarray,
    scenario_effects: np.ndarray,
    response_uniforms: np.ndarray,
    negative_uniforms: np.ndarray,
    user_orders: np.ndarray,
    config: SimulationConfig,
    intervention: str = "none",
    oracle_misspecification: float = 1.0,
    heterogeneous_multipliers: np.ndarray | None = None,
    panel_histories: list[np.ndarray] | None = None,
    replay_uniforms: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run matched branches with round-batched recommendations and explicit updates."""
    if config.update_regime not in {
        "event_online",
        "fixed_budget_history_replay",
        "round_synchronous",
    }:
        raise ValueError(f"Unknown update regime: {config.update_regime}")
    if config.update_regime == "fixed_budget_history_replay":
        if panel_histories is None or replay_uniforms is None:
            raise ValueError("History replay requires panel_histories and replay_uniforms")
        if replay_uniforms.shape != (config.rounds, len(panel)):
            raise ValueError("replay_uniforms has the wrong shape")
    rows: list[dict[str, float | int | str]] = []
    positions = position_effect(config.top_k, config.position_scale)
    n_items = len(item_groups)
    union_parts = [candidate_pools.ravel()]
    if panel_histories is not None:
        union_parts.extend(np.asarray(values, np.int32) for values in panel_histories)
    union_items = np.unique(np.concatenate(union_parts))
    global_to_local = np.full(n_items, -1, dtype=np.int32)
    global_to_local[union_items] = np.arange(len(union_items), dtype=np.int32)
    candidate_local = global_to_local[candidate_pools]
    for branch in ("frozen", "closed"):
        U = initial_user_embeddings[panel].copy()
        V = initial_item_embeddings[union_items].copy()
        exposure = np.zeros(n_items, dtype=np.float64)
        clicks = 0
        impressions = 0
        clicked_utility_sum = 0.0
        gradient_evaluations = 0
        parameter_updates = 0
        seen = np.zeros(candidate_pools.shape, dtype=bool)
        for round_index in range(config.rounds):
            score_matrix = np.einsum(
                "ud,umd->um", U, V[candidate_local], optimize=True
            ).astype(np.float32)
            score_matrix[seen] = -np.inf
            means, sds = candidate_anchor(score_matrix, config.minimum_score_sd)
            anchored_matrix = apply_anchor(score_matrix, means, sds)
            accepted: list[tuple[int, int, int, float]] = []
            for panel_row in range(len(panel)):
                panel_row = int(panel_row)
                candidates = candidate_pools[panel_row]
                raw = score_matrix[panel_row]
                anchored_pool = anchored_matrix[panel_row]
                displayed = select_display(
                    intervention,
                    candidates,
                    raw,
                    item_groups,
                    round_index,
                    config.top_k,
                )
                candidate_positions = _display_positions(candidates, displayed)
                anchored = anchored_pool[candidate_positions]
                effect_row = (
                    scenario_effects[panel_row]
                    if scenario_effects.ndim == 2
                    else scenario_effects
                )
                effects = effect_row[item_groups[displayed]].astype(float)
                if heterogeneous_multipliers is not None:
                    effects *= heterogeneous_multipliers[panel_row]
                probabilities = softmax_with_outside(
                    anchored + positions + effects, config.outside_utility
                )
                clicked_position = choice_from_uniform(
                    probabilities, float(response_uniforms[round_index, panel_row])
                )
                # Per-item discounted exposure; written explicitly to avoid group duplication.
                exposure[displayed] += 1.0 / np.log2(
                    np.arange(2, config.top_k + 2, dtype=float)
                )
                impressions += 1
                if clicked_position >= 0:
                    clicks += 1
                    clicked_utility_sum += float(anchored[clicked_position])
                    seen[panel_row, candidate_positions[clicked_position]] = True
                    if branch == "closed":
                        positive = int(displayed[clicked_position])
                        negative_index = min(
                            int(negative_uniforms[round_index, panel_row] * (config.top_k - 1)),
                            config.top_k - 2,
                        )
                        if negative_index >= clicked_position:
                            negative_index += 1
                        negative = int(displayed[negative_index])
                        weight = feedback_weight(
                            intervention,
                            anchored,
                            positions,
                            effects,
                            clicked_position,
                            oracle_misspecification,
                            config.outside_utility,
                        )
                        accepted.append((panel_row, positive, negative, weight))
            if branch == "closed" and accepted:
                by_user = {entry[0]: entry for entry in accepted}
                ordered = [
                    by_user[int(panel_row)]
                    for panel_row in user_orders[round_index]
                    if int(panel_row) in by_user
                ]
                if config.update_regime == "round_synchronous":
                    round_users = np.asarray(
                        [entry[0] for entry in ordered],
                        dtype=np.int32,
                    )
                    round_positives = np.asarray(
                        [global_to_local[entry[1]] for entry in ordered],
                        dtype=np.int32,
                    )
                    round_negatives = np.asarray(
                        [global_to_local[entry[2]] for entry in ordered],
                        dtype=np.int32,
                    )
                    round_weights = np.asarray(
                        [entry[3] for entry in ordered],
                        dtype=np.float32,
                    )
                    evaluations, operations = update_batch_synchronous(
                        U,
                        V,
                        round_users,
                        round_positives,
                        round_negatives,
                        config.online_learning_rate,
                        config.online_regularization,
                        weights=round_weights,
                        steps=config.online_steps,
                    )
                    gradient_evaluations += evaluations
                    parameter_updates += operations
                else:
                    for user_local, positive, negative, weight in ordered:
                        if config.update_regime == "event_online":
                            count = update_pair(
                                U, V, user_local,
                                int(global_to_local[positive]),
                                int(global_to_local[negative]),
                                config.online_learning_rate,
                                config.online_regularization,
                                weight=weight,
                                steps=config.online_steps,
                            )
                            gradient_evaluations += count
                            parameter_updates += count
                        else:
                            count = update_pair(
                                U, V, user_local,
                                int(global_to_local[positive]),
                                int(global_to_local[negative]),
                                config.online_learning_rate,
                                config.online_regularization,
                                weight=weight,
                                steps=config.clicked_steps,
                            )
                            gradient_evaluations += count
                            parameter_updates += count
                            history = panel_histories[user_local]
                            if config.replay_steps and len(history):
                                replay_index = min(
                                    int(
                                        replay_uniforms[round_index, user_local]
                                        * len(history)
                                    ),
                                    len(history) - 1,
                                )
                                replay_positive = int(history[replay_index])
                                count = update_pair(
                                    U,
                                    V,
                                    user_local,
                                    int(global_to_local[replay_positive]),
                                    int(global_to_local[negative]),
                                    config.online_learning_rate,
                                    config.online_regularization,
                                    weight=1.0,
                                    steps=config.replay_steps,
                                )
                                gradient_evaluations += count
                                parameter_updates += count
            metrics = cumulative_metrics(
                exposure, clicks, impressions, clicked_utility_sum, item_groups
            )
            rows.append(
                {
                    "branch": branch,
                    "round": round_index + 1,
                    "accepted_interactions": clicks,
                    "online_sgd_updates": gradient_evaluations,
                    "event_gradient_evaluations": gradient_evaluations,
                    "parameter_update_operations": parameter_updates,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def zero_effect_drift(frame: pd.DataFrame) -> float:
    final = frame[frame["round"] == frame["round"].max()].set_index("branch")
    return float(final.loc["closed", "D"] - final.loc["frozen", "D"])
