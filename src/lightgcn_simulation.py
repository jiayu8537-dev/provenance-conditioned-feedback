from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .choice_model import choice_from_uniform, position_effect, softmax_with_outside
from .data_loading import DataBundle
from .exposure_metrics import cumulative_metrics
from .interventions import feedback_weight, select_display
from .lightgcn_cpu import (
    FullGraphLightGCN,
    normalized_bipartite_adjacency,
    weighted_bpr_loss,
)
from .score_normalization import apply_anchor, candidate_anchor


@dataclass(frozen=True)
class DynamicLightGCNConfig:
    rounds: int = 6
    top_k: int = 20
    learning_rate: float = 0.005
    regularization: float = 0.0001
    update_steps: int = 3
    outside_utility: float = 0.5
    position_scale: float = 0.8
    minimum_score_sd: float = 1e-6
    layers: int = 2


def _display_positions(candidates: np.ndarray, displayed: np.ndarray) -> np.ndarray:
    lookup = {int(item): index for index, item in enumerate(candidates)}
    return np.asarray([lookup[int(item)] for item in displayed], dtype=np.int32)


def _propagated_candidate_scores(
    model: FullGraphLightGCN,
    panel: np.ndarray,
    candidate_pools: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        users, items = model.propagate()
        user_vectors = users[torch.from_numpy(panel.astype(np.int64))]
        candidate_index = torch.from_numpy(candidate_pools.astype(np.int64))
        candidate_vectors = items[candidate_index]
        scores = torch.einsum("ud,umd->um", user_vectors, candidate_vectors)
    return scores.detach().cpu().numpy().astype(np.float32)


def simulate_paired_lightgcn(
    data: DataBundle,
    initial_raw_embeddings: np.ndarray,
    panel: np.ndarray,
    candidate_pools: np.ndarray,
    item_groups: np.ndarray,
    scenario_effects: np.ndarray,
    response_uniforms: np.ndarray,
    negative_uniforms: np.ndarray,
    config: DynamicLightGCNConfig,
    intervention: str = "none",
    oracle_misspecification: float = 1.0,
) -> pd.DataFrame:
    if response_uniforms.shape != (config.rounds, len(panel)):
        raise ValueError("response_uniforms has the wrong shape")
    if negative_uniforms.shape != (config.rounds, len(panel)):
        raise ValueError("negative_uniforms has the wrong shape")
    base_adjacency = normalized_bipartite_adjacency(
        data.n_users,
        data.n_items,
        data.all_users,
        data.all_items,
    )
    rows: list[dict[str, float | int | str]] = []
    positions = position_effect(config.top_k, config.position_scale)
    n_items = data.n_items
    initial_model = FullGraphLightGCN(
        data.n_users,
        data.n_items,
        initial_raw_embeddings.shape[1],
        config.layers,
        base_adjacency,
        raw_embeddings=initial_raw_embeddings,
    )
    initial_scores = _propagated_candidate_scores(
        initial_model,
        panel,
        candidate_pools,
    )
    del initial_model

    for branch in ("frozen", "closed"):
        model = FullGraphLightGCN(
            data.n_users,
            data.n_items,
            initial_raw_embeddings.shape[1],
            config.layers,
            base_adjacency,
            raw_embeddings=initial_raw_embeddings,
        )
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            if branch == "closed"
            else None
        )
        graph_users = data.all_users.astype(np.int32).copy()
        graph_items = data.all_items.astype(np.int32).copy()
        exposure = np.zeros(n_items, dtype=np.float64)
        clicks = 0
        impressions = 0
        clicked_utility_sum = 0.0
        updates = 0
        seen = np.zeros(candidate_pools.shape, dtype=bool)
        cumulative_new_edges = 0

        for round_index in range(config.rounds):
            score_matrix = (
                initial_scores.copy()
                if branch == "frozen"
                else _propagated_candidate_scores(model, panel, candidate_pools)
            )
            score_matrix[seen] = -np.inf
            means, standard_deviations = candidate_anchor(
                score_matrix,
                config.minimum_score_sd,
            )
            anchored_matrix = apply_anchor(
                score_matrix,
                means,
                standard_deviations,
            )
            accepted: list[tuple[int, int, int, float]] = []

            for panel_row in range(len(panel)):
                candidates = candidate_pools[panel_row]
                raw_scores = score_matrix[panel_row]
                displayed = select_display(
                    intervention,
                    candidates,
                    raw_scores,
                    item_groups,
                    round_index,
                    config.top_k,
                )
                candidate_positions = _display_positions(candidates, displayed)
                anchored = anchored_matrix[panel_row, candidate_positions]
                effects = scenario_effects[item_groups[displayed]].astype(float)
                probabilities = softmax_with_outside(
                    anchored + positions + effects,
                    config.outside_utility,
                )
                clicked_position = choice_from_uniform(
                    probabilities,
                    float(response_uniforms[round_index, panel_row]),
                )
                exposure[displayed] += 1.0 / np.log2(
                    np.arange(2, config.top_k + 2, dtype=float)
                )
                impressions += 1
                if clicked_position < 0:
                    continue
                clicks += 1
                clicked_utility_sum += float(anchored[clicked_position])
                seen[panel_row, candidate_positions[clicked_position]] = True
                if branch == "closed":
                    positive = int(displayed[clicked_position])
                    negative_position = min(
                        int(
                            negative_uniforms[round_index, panel_row]
                            * (config.top_k - 1)
                        ),
                        config.top_k - 2,
                    )
                    if negative_position >= clicked_position:
                        negative_position += 1
                    negative = int(displayed[negative_position])
                    weight = feedback_weight(
                        intervention,
                        anchored,
                        positions,
                        effects,
                        clicked_position,
                        oracle_misspecification,
                        config.outside_utility,
                    )
                    accepted.append((int(panel[panel_row]), positive, negative, weight))

            if branch == "closed" and accepted:
                new_users = np.asarray([entry[0] for entry in accepted], np.int32)
                new_items = np.asarray([entry[1] for entry in accepted], np.int32)
                negatives = np.asarray([entry[2] for entry in accepted], np.int32)
                weights = np.asarray([entry[3] for entry in accepted], np.float32)
                graph_users = np.concatenate([graph_users, new_users])
                graph_items = np.concatenate([graph_items, new_items])
                cumulative_new_edges += len(accepted)
                model.set_adjacency(
                    normalized_bipartite_adjacency(
                        data.n_users,
                        data.n_items,
                        graph_users,
                        graph_items,
                    )
                )
                assert optimizer is not None
                for _ in range(config.update_steps):
                    loss = weighted_bpr_loss(
                        model,
                        new_users,
                        new_items,
                        negatives,
                        config.regularization,
                        weights,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    updates += len(accepted)

            metrics = cumulative_metrics(
                exposure,
                clicks,
                impressions,
                clicked_utility_sum,
                item_groups,
            )
            rows.append(
                {
                    "branch": branch,
                    "round": round_index + 1,
                    "accepted_interactions": clicks,
                    "online_optimizer_events": updates,
                    "cumulative_new_graph_edges": cumulative_new_edges,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
