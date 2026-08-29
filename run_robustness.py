#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse, stats
from scipy.cluster.vq import kmeans2

from src.bootstrap import add_amplification, hierarchical_bootstrap_mean
from src.bpr_training import load_or_train, train_full_history_bpr
from src.data_loading import (
    generate_assignment,
    load_data,
    make_candidate_pool,
    qualified_assignment_seeds,
    stratified_panel,
)
from src.simulation import SimulationConfig, simulate_paired


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("JIIS_DATA_ROOT", ROOT / "data" / "derived"))
MAIN_SPLIT = DATA_ROOT / "main_10u5i_leave_last_two_split.csv.gz"
STRICT_USERS = DATA_ROOT / "strict_15u5i_user_index.csv"
STRICT_ITEMS = DATA_ROOT / "strict_15u5i_item_index.csv"
STRICT_ROWS = DATA_ROOT / "strict_15u5i_interaction_row_ids.csv.gz"
MAIN_ASSIGNMENT = DATA_ROOT / "model_aware_provenance_assignment_v2.csv"
MAIN_ASSIGNMENT_CANDIDATES = (
    ROOT / "data" / "protocol" / "model_aware_assignment_candidates.csv"
)

CONFIGS = ROOT / "configs"
RAW = ROOT / "raw" / "robustness"
TABLES = ROOT / "tables" / "robustness"
LOGS = ROOT / "logs" / "robustness"
ARTIFACTS = ROOT / "artifacts" / "robustness"
STRICT = ARTIFACTS / "strict_core"
for directory in (RAW, TABLES, LOGS, ARTIFACTS, STRICT):
    directory.mkdir(parents=True, exist_ok=True)

STRICT_SPLIT = STRICT / "strict_15u5i_leave_last_two_split.csv.gz"
STRICT_ASSIGNMENT = STRICT / "strict_model_aware_assignment.csv.gz"
STRICT_ASSIGNMENT_CANDIDATES = (
    STRICT / "strict_model_aware_assignment_candidates.csv"
)
STRICT_SELECTED_ASSIGNMENTS = STRICT / "strict_selected_assignments.csv"
STRICT_BPR = STRICT / "strict_bpr_full_history.npz"
STRICT_BPR_MANIFEST = STRICT / "strict_bpr_full_history_manifest.json"
STRICT_BPR_LOG = LOGS / "strict_bpr_full_history_training.csv"


def read_configs() -> tuple[dict, dict]:
    robustness = yaml.safe_load(
        (CONFIGS / "robustness.yaml").read_text(encoding="utf-8")
    )
    bpr = yaml.safe_load((CONFIGS / "bpr.yaml").read_text(encoding="utf-8"))
    return robustness, bpr


def recompute_leave_last_two(split: pd.DataFrame) -> pd.DataFrame:
    ordered = split.sort_values(
        ["user_id", "timestamp", "row_id"],
        kind="stable",
    ).copy()
    ordered["position_from_end"] = (
        ordered.groupby("user_id").cumcount(ascending=False)
    )
    ordered["split"] = "train"
    ordered.loc[ordered.position_from_end == 1, "split"] = "validation"
    ordered.loc[ordered.position_from_end == 0, "split"] = "test"
    warm_items = set(
        ordered.loc[ordered["split"] == "train", "item_id"].astype(str)
    )
    ordered["warm_item"] = ordered.item_id.astype(str).isin(warm_items)
    ordered = ordered.drop(columns="position_from_end")
    return ordered.sort_values("row_id", kind="stable").reset_index(drop=True)


def preliminary_strict_assignment(
    split: pd.DataFrame,
    strict_items: set[str],
) -> pd.DataFrame:
    assignment = pd.read_csv(MAIN_ASSIGNMENT)
    assignment = assignment[
        assignment.item_id.astype(str).isin(strict_items)
    ].copy()
    counts = split.groupby("item_id").size().rename("strict_interactions")
    assignment = assignment.merge(
        counts,
        left_on="item_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    assignment["sampled_interactions"] = (
        assignment.strict_interactions.fillna(0).astype(int)
    )
    rank = assignment.sampled_interactions.rank(method="first")
    assignment["strict_popularity_bin"] = (
        pd.qcut(rank, 5, labels=False, duplicates="drop").astype(int)
    )
    assignment["model_aware_block"] = (
        assignment.latent_cluster.astype(str)
        + "|sp"
        + (assignment.strict_popularity_bin + 1).astype(str)
    )
    return assignment.drop(columns="strict_interactions")


def rebuild_strict_blocks(
    assignment: pd.DataFrame,
    item_embeddings: np.ndarray,
    values: dict,
) -> pd.DataFrame:
    dimensions = int(
        values["strict_core"]["random_projection_dimensions"]
    )
    clusters = int(values["strict_core"]["latent_clusters"])
    seed = int(values["global_seed"]) + 61000
    rng = np.random.default_rng(seed)
    normalized = item_embeddings / np.maximum(
        np.linalg.norm(item_embeddings, axis=1, keepdims=True),
        1e-8,
    )
    projection = rng.normal(
        size=(normalized.shape[1], dimensions)
    ).astype(np.float32) / np.sqrt(normalized.shape[1])
    observed = np.log1p(
        assignment[
            [
                "sampled_interactions",
                "view_number",
                "thumbup_number",
                "favorite_number",
            ]
        ]
        .fillna(0)
        .to_numpy(np.float32)
    )
    features = np.concatenate([normalized @ projection, observed], axis=1)
    means = features.mean(axis=0)
    standard_deviations = features.std(axis=0)
    features = (features - means) / np.maximum(standard_deviations, 1e-8)
    _, labels = kmeans2(
        features.astype(np.float32),
        clusters,
        iter=75,
        minit="++",
        seed=seed,
    )
    rebuilt = assignment.copy()
    rebuilt["strict_latent_cluster"] = labels.astype(int)
    rebuilt["model_aware_block"] = (
        rebuilt.strict_latent_cluster.astype(str)
        + "|sp"
        + (rebuilt.strict_popularity_bin + 1).astype(str)
    )
    return rebuilt


def exact_top_k(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    users: np.ndarray,
    seen: sparse.csr_matrix,
    eligible_items: np.ndarray | None = None,
    top_k: int = 20,
    batch_size: int = 256,
) -> np.ndarray:
    eligible_mask = None
    if eligible_items is not None:
        eligible_mask = np.zeros(len(item_embeddings), dtype=bool)
        eligible_mask[np.asarray(eligible_items, dtype=np.int32)] = True
    result = np.empty((len(users), top_k), dtype=np.int32)
    item_tensor = torch.from_numpy(item_embeddings.astype(np.float32))
    for start in range(0, len(users), batch_size):
        selected_users = users[start : start + batch_size]
        scores = (
            torch.from_numpy(
                user_embeddings[selected_users].astype(np.float32)
            )
            @ item_tensor.T
        )
        if eligible_mask is not None:
            ineligible = np.where(~eligible_mask)[0]
            if len(ineligible):
                scores[:, torch.from_numpy(ineligible)] = -torch.inf
        for row, user in enumerate(selected_users):
            consumed = seen.indices[
                seen.indptr[user] : seen.indptr[user + 1]
            ]
            if len(consumed):
                scores[row, torch.from_numpy(consumed)] = -torch.inf
        result[start : start + len(selected_users)] = (
            torch.topk(scores, top_k, dim=1)
            .indices.numpy()
            .astype(np.int32)
        )
    return result


def most_popular_top_k(
    item_counts: np.ndarray,
    users: np.ndarray,
    seen: sparse.csr_matrix,
    top_k: int,
) -> np.ndarray:
    order = np.argsort(-item_counts, kind="stable")
    result = np.empty((len(users), top_k), dtype=np.int32)
    for row, user in enumerate(users):
        consumed = set(
            seen.indices[seen.indptr[user] : seen.indptr[user + 1]].tolist()
        )
        selected: list[int] = []
        for item in order:
            if int(item) not in consumed:
                selected.append(int(item))
            if len(selected) == top_k:
                break
        result[row] = selected
    return result


def exposure_vector(top: np.ndarray, catalog_size: int) -> np.ndarray:
    weights = 1.0 / np.log2(np.arange(2, top.shape[1] + 2, dtype=float))
    return np.bincount(
        top.ravel(),
        weights=np.tile(weights, len(top)),
        minlength=catalog_size,
    )


def continuous_balance_matrix(assignment: pd.DataFrame) -> np.ndarray:
    columns = [
        "sampled_interactions",
        "observed_item_age_days",
        "view_number",
        "comment_number",
        "thumbup_number",
        "share_number",
        "coin_number",
        "favorite_number",
        "barrage_number",
        "platform_engagement_score",
    ]
    transformed = []
    for column in columns:
        values = pd.to_numeric(
            assignment[column],
            errors="coerce",
        )
        values = values.fillna(values.median()).to_numpy(float)
        if column not in {
            "observed_item_age_days",
            "platform_engagement_score",
        }:
            values = np.log1p(np.maximum(values, 0))
        transformed.append(
            (values - values.mean()) / max(values.std(), 1e-12)
        )
    return np.vstack(transformed).T


def assignment_objective(
    labels: np.ndarray,
    balance_matrix: np.ndarray,
    theme_codes: np.ndarray,
    exposures: dict[str, np.ndarray],
) -> dict[str, float]:
    group_means = np.vstack(
        [balance_matrix[labels == group].mean(axis=0) for group in range(3)]
    )
    maximum_smd = max(
        np.abs(group_means[first] - group_means[second]).max()
        for first in range(3)
        for second in range(first + 1, 3)
    )
    theme_count = int(theme_codes.max()) + 1
    theme_distributions = np.zeros((theme_count, 3), dtype=float)
    for group in range(3):
        theme_distributions[:, group] = (
            np.bincount(
                theme_codes[labels == group],
                minlength=theme_count,
            )
            / (labels == group).sum()
        )
    maximum_theme_tv = max(
        0.5
        * np.abs(
            theme_distributions[:, first]
            - theme_distributions[:, second]
        ).sum()
        for first in range(3)
        for second in range(first + 1, 3)
    )
    maximum_exposure_deviation = 0.0
    gaps: dict[str, float] = {}
    catalog_share = np.bincount(labels, minlength=3) / len(labels)
    for name, exposure in exposures.items():
        group_exposure = np.bincount(
            labels,
            weights=exposure,
            minlength=3,
        )
        exposure_share = group_exposure / group_exposure.sum()
        relative_share = exposure_share / catalog_share
        maximum_exposure_deviation = max(
            maximum_exposure_deviation,
            float(np.abs(relative_share - 1.0).max()),
        )
        gaps[name] = float(
            np.log(
                (relative_share[2] + 1e-12)
                / (relative_share[0] + 1e-12)
            )
        )
    return {
        "objective": float(
            maximum_smd
            + 1.5 * maximum_theme_tv
            + 3.0 * maximum_exposure_deviation
        ),
        "max_standardized_mean_difference": float(maximum_smd),
        "max_theme_TV": float(maximum_theme_tv),
        "max_model_exposure_deviation": float(
            maximum_exposure_deviation
        ),
        "MostPop_HA_gap": gaps["MostPop"],
        "BPR_HA_gap": gaps["BPR"],
    }


def strict_panels(data, values: dict) -> list[np.ndarray]:
    return [
        stratified_panel(
            data.user_activity,
            int(values["panel_size"]),
            int(values["global_seed"]) + 51000 + panel * 1000,
        )
        for panel in range(int(values["panel_count"]))
    ]


def prepare_strict_core() -> None:
    started = time.time()
    values, bpr = read_configs()
    source = pd.read_csv(MAIN_SPLIT)
    strict_users = set(pd.read_csv(STRICT_USERS).user_id.astype(str))
    strict_items = set(pd.read_csv(STRICT_ITEMS).item_id.astype(str))
    strict_row_ids = set(pd.read_csv(STRICT_ROWS).row_id.astype(int))
    filtered = source[
        source.user_id.astype(str).isin(strict_users)
        & source.item_id.astype(str).isin(strict_items)
    ].copy()
    if set(filtered.row_id.astype(int)) != strict_row_ids:
        raise RuntimeError("Strict-core row identifiers do not match the filtered split")
    strict_split = recompute_leave_last_two(filtered)
    strict_split.to_csv(STRICT_SPLIT, index=False, compression="gzip")

    preliminary = preliminary_strict_assignment(strict_split, strict_items)
    preliminary_path = STRICT / "strict_assignment_preliminary.csv.gz"
    preliminary.to_csv(preliminary_path, index=False, compression="gzip")
    preliminary_data = load_data(STRICT_SPLIT, preliminary_path)
    users, items = load_or_train(
        preliminary_data,
        STRICT_BPR,
        STRICT_BPR_LOG,
        STRICT_BPR_MANIFEST,
        bpr,
    )
    rebuilt = rebuild_strict_blocks(preliminary, items, values)
    rebuilt.to_csv(STRICT_ASSIGNMENT, index=False, compression="gzip")
    data = load_data(STRICT_SPLIT, STRICT_ASSIGNMENT)
    panels = strict_panels(data, values)
    allocation_users = np.concatenate(panels)
    bpr_top = exact_top_k(
        users,
        items,
        allocation_users,
        data.full_csr,
        top_k=int(values["top_k"]),
    )
    pop_top = most_popular_top_k(
        data.item_counts,
        allocation_users,
        data.full_csr,
        int(values["top_k"]),
    )
    exposures = {
        "MostPop": exposure_vector(pop_top, data.n_items),
        "BPR": exposure_vector(bpr_top, data.n_items),
    }
    balance_matrix = continuous_balance_matrix(rebuilt)
    theme_codes = pd.Categorical(rebuilt.theme_group).codes
    candidate_rows = []
    candidate_start = int(values["global_seed"]) + 71000
    for seed in range(
        candidate_start,
        candidate_start + int(values["candidate_assignment_count"]),
    ):
        labels = generate_assignment(data.n_items, data.blocks, seed)
        candidate_rows.append(
            {
                "seed": seed,
                **assignment_objective(
                    labels,
                    balance_matrix,
                    theme_codes,
                    exposures,
                ),
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        "objective",
        kind="stable",
    )
    candidates.to_csv(STRICT_ASSIGNMENT_CANDIDATES, index=False)
    selected_seeds, selected = qualified_assignment_seeds(
        STRICT_ASSIGNMENT_CANDIDATES,
        int(values["assignment_count"]),
    )
    selected.to_csv(STRICT_SELECTED_ASSIGNMENTS, index=False)
    counts = strict_split.groupby("split").size().to_dict()
    (LOGS / "strict_core_preparation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "seconds": time.time() - started,
                "interactions": len(strict_split),
                "users": strict_split.user_id.nunique(),
                "items": strict_split.item_id.nunique(),
                "split_counts": counts,
                "qualified_assignments": int(
                    (
                        (
                            candidates.max_standardized_mean_difference
                            < 0.05
                        )
                        & (candidates.max_theme_TV < 0.05)
                        & (
                            candidates.max_model_exposure_deviation
                            < 0.10
                        )
                    ).sum()
                ),
                "selected_assignment_seeds": selected_seeds,
                "block_definition": (
                    "120 strict-core BPR/covariate clusters crossed with "
                    "strict-core popularity quintiles"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    evaluate_strict_static(strict_split, rebuilt, bpr)


def gini(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    total = ordered.sum()
    count = len(ordered)
    if total == 0:
        return 0.0
    return float(
        2.0
        * np.dot(np.arange(1, count + 1), ordered)
        / (count * total)
        - (count + 1) / count
    )


def evaluate_strict_static(
    strict_split: pd.DataFrame,
    assignment: pd.DataFrame,
    bpr: dict,
) -> None:
    train_validation = strict_split[
        strict_split["split"].isin(["train", "validation"])
    ].copy()
    training_path = STRICT / "strict_train_validation.csv.gz"
    train_validation.to_csv(training_path, index=False, compression="gzip")
    assignment_path = STRICT / "strict_assignment_for_static.csv.gz"
    assignment.to_csv(assignment_path, index=False, compression="gzip")
    training_data = load_data(training_path, assignment_path)
    test_users, test_items, logs = train_full_history_bpr(
        training_data,
        dimension=int(bpr["dimension"]),
        learning_rate=float(bpr["learning_rate"]),
        regularization=float(bpr["regularization"]),
        epochs=int(bpr["epochs"]),
        seed=int(bpr["seed"]),
    )
    logs.to_csv(LOGS / "strict_static_bpr_training.csv", index=False)
    test = strict_split[strict_split["split"] == "test"].copy()
    test["u"] = test.user_id.map(training_data.user_to_index)
    test["i"] = test.item_id.map(training_data.item_to_index)
    test = test.dropna(subset=["u", "i"])
    test["u"] = test.u.astype(np.int32)
    test["i"] = test.i.astype(np.int32)
    warm_items = np.where(training_data.item_counts > 0)[0].astype(np.int32)
    test = test[test.i.isin(warm_items)]
    top = exact_top_k(
        test_users,
        test_items,
        test.u.to_numpy(np.int32),
        training_data.full_csr,
        eligible_items=warm_items,
        top_k=20,
    )
    positives = test.i.to_numpy(np.int32)
    hit_matrix = top == positives.reshape(-1, 1)
    hits = hit_matrix.any(axis=1)
    ranks = np.zeros(len(hits), dtype=np.int32)
    ranks[hits] = hit_matrix[hits].argmax(axis=1) + 1
    ndcg = np.zeros(len(hits), dtype=float)
    ndcg[hits] = 1.0 / np.log2(ranks[hits] + 1)
    counts = np.bincount(top.ravel(), minlength=training_data.n_items)
    pd.DataFrame(
        [
            {
                "model": "BPR-MF",
                "core": "strict_15u5i",
                "Recall@20": float(hits.mean()),
                "HitRate@20": float(hits.mean()),
                "NDCG@20": float(ndcg.mean()),
                "Coverage@20": float((counts > 0).sum() / len(counts)),
                "Gini@20": gini(counts),
                "evaluated_users": len(test),
                "catalog_items": training_data.n_items,
                "training_interactions": len(train_validation),
            }
        ]
    ).to_csv(TABLES / "strict_static_metrics.csv", index=False)


def random_streams(
    values: dict,
    panel_size: int,
    response_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(
        int(values["global_seed"]) + 30000 + response_seed
    )
    response = rng.random(
        (int(values["rounds"]), panel_size),
        dtype=np.float32,
    )
    negative = rng.random(
        (int(values["rounds"]), panel_size),
        dtype=np.float32,
    )
    orders = np.tile(
        np.arange(panel_size, dtype=np.int32),
        (int(values["rounds"]), 1),
    )
    return response, negative, orders


def simulation_config(values: dict, regime: str) -> SimulationConfig:
    return SimulationConfig(
        rounds=int(values["rounds"]),
        top_k=int(values["top_k"]),
        online_steps=int(values["online_steps"]),
        online_learning_rate=float(values["online_learning_rate"]),
        online_regularization=float(values["online_regularization"]),
        outside_utility=float(values["outside_utility"]),
        position_scale=float(values["position_scale"]),
        minimum_score_sd=float(values["minimum_score_sd"]),
        update_regime=regime,
    )


def load_main_context(panel_index: int):
    values, bpr = read_configs()
    data = load_data(MAIN_SPLIT, MAIN_ASSIGNMENT)
    users, items = load_or_train(
        data,
        ROOT / "artifacts" / "bpr_full_history.npz",
        ROOT / "logs" / "bpr_training.csv",
        ROOT / "artifacts" / "bpr_manifest.json",
        bpr,
    )
    panel = stratified_panel(
        data.user_activity,
        int(values["panel_size"]),
        int(values["global_seed"]) + 21000 + panel_index * 1000,
    )
    pools, _ = make_candidate_pool(
        data,
        users,
        items,
        panel,
        int(values["candidate_size"]),
        int(values["top_retrieval"]),
        int(values["global_seed"])
        + 22000
        + panel_index * 1000
        + int(values["candidate_size"]),
    )
    seeds, _ = qualified_assignment_seeds(
        MAIN_ASSIGNMENT_CANDIDATES,
        int(values["assignment_count"]),
    )
    return values, data, users, items, panel, pools, seeds


def load_strict_context(panel_index: int):
    values, bpr = read_configs()
    if not (
        STRICT_SPLIT.exists()
        and STRICT_ASSIGNMENT.exists()
        and STRICT_ASSIGNMENT_CANDIDATES.exists()
    ):
        raise RuntimeError("Run prepare-strict before strict-core validation")
    data = load_data(STRICT_SPLIT, STRICT_ASSIGNMENT)
    users, items = load_or_train(
        data,
        STRICT_BPR,
        STRICT_BPR_LOG,
        STRICT_BPR_MANIFEST,
        bpr,
    )
    panel = strict_panels(data, values)[panel_index]
    pools, _ = make_candidate_pool(
        data,
        users,
        items,
        panel,
        int(values["candidate_size"]),
        int(values["top_retrieval"]),
        int(values["global_seed"])
        + 52000
        + panel_index * 1000
        + int(values["candidate_size"]),
    )
    seeds, _ = qualified_assignment_seeds(
        STRICT_ASSIGNMENT_CANDIDATES,
        int(values["assignment_count"]),
    )
    return values, data, users, items, panel, pools, seeds


def run_panel(panel_index: int, design: str) -> None:
    started = time.time()
    if design == "synchronous":
        (
            values,
            data,
            users,
            items,
            panel,
            pools,
            assignment_seeds,
        ) = load_main_context(panel_index)
        regimes = ["event_online", "round_synchronous"]
        core = "confirmatory_10u5i"
    elif design == "strict":
        (
            values,
            data,
            users,
            items,
            panel,
            pools,
            assignment_seeds,
        ) = load_strict_context(panel_index)
        regimes = ["event_online"]
        core = "strict_15u5i"
    else:
        raise ValueError(design)
    rows = []
    for assignment_seed in assignment_seeds:
        labels = generate_assignment(
            data.n_items,
            data.blocks,
            int(assignment_seed),
        )
        for response_seed in range(int(values["response_seed_count"])):
            response, negative, orders = random_streams(
                values,
                len(panel),
                response_seed,
            )
            for scenario, effect in values["scenarios"].items():
                for regime in regimes:
                    frame = simulate_paired(
                        users,
                        items,
                        panel,
                        pools,
                        labels,
                        np.asarray(effect, dtype=np.float32),
                        response,
                        negative,
                        orders,
                        simulation_config(values, regime),
                        intervention="none",
                    )
                    frame.insert(0, "model", "BPR-MF")
                    frame.insert(1, "core", core)
                    frame.insert(2, "panel", panel_index)
                    frame.insert(3, "assignment", assignment_seed)
                    frame.insert(4, "response_seed", response_seed)
                    frame.insert(5, "scenario", scenario)
                    frame.insert(6, "intervention", "none")
                    frame.insert(7, "update_regime", regime)
                    rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    output = RAW / design / f"panel_{panel_index}.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, compression="gzip")
    (LOGS / f"{design}_panel_{panel_index}.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "seconds": time.time() - started,
                "rows": len(combined),
                "design": design,
                "panel": panel_index,
                "core": core,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def crossed_bootstrap_interval(
    frame: pd.DataFrame,
    value: str,
    repetitions: int,
    seed: int,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    dimensions = ["panel", "assignment", "response_seed"]
    levels = {
        dimension: np.sort(frame[dimension].unique())
        for dimension in dimensions
    }
    values = frame[value].to_numpy(dtype=float)
    estimates = np.empty(repetitions, dtype=float)
    chunk = min(1000, repetitions)
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        size = stop - start
        weights = np.ones((size, len(frame)), dtype=np.int32)
        for dimension in dimensions:
            lookup = {
                level: index
                for index, level in enumerate(levels[dimension])
            }
            row_indices = (
                frame[dimension].map(lookup).to_numpy(dtype=int)
            )
            counts = rng.multinomial(
                len(levels[dimension]),
                np.full(
                    len(levels[dimension]),
                    1.0 / len(levels[dimension]),
                ),
                size=size,
            )
            weights *= counts[:, row_indices]
        estimates[start:stop] = (
            weights @ values
        ) / weights.sum(axis=1)
    return (
        float(np.quantile(estimates, lower)),
        float(np.quantile(estimates, upper)),
    )


def aggregate_design(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_rows = []
    zero_rows = []
    values, _ = read_configs()
    for (core, regime), subset in raw.groupby(
        ["core", "update_regime"]
    ):
        amplified = add_amplification(subset)
        final = amplified[
            amplified["round"] == int(values["rounds"])
        ]
        for scenario, group in final.groupby("scenario"):
            if scenario == "zero":
                continue
            summary = hierarchical_bootstrap_mean(
                group,
                "AA",
                int(values["bootstrap_repetitions"]),
                20278100 + len(endpoint_rows),
            )
            endpoint_rows.append(
                {
                    "core": core,
                    "update_regime": regime,
                    "scenario": scenario,
                    **summary,
                    "relative_ratio_percent": float(
                        100.0 * np.expm1(summary["estimate"])
                    ),
                    "n": len(group),
                }
            )
        final_raw = subset[
            (subset["round"] == int(values["rounds"]))
            & (subset.scenario == "zero")
        ]
        keys = [
            "panel",
            "assignment",
            "response_seed",
            "scenario",
            "intervention",
            "round",
        ]
        wide = final_raw.pivot(
            index=keys,
            columns="branch",
            values="D",
        ).reset_index()
        wide["zero_drift"] = wide.closed - wide.frozen
        summary = hierarchical_bootstrap_mean(
            wide,
            "zero_drift",
            int(values["bootstrap_repetitions"]),
            20278200 + len(zero_rows),
        )
        ci90_low, ci90_high = crossed_bootstrap_interval(
            wide,
            "zero_drift",
            int(values["bootstrap_repetitions"]),
            20278300 + len(zero_rows),
            0.05,
            0.95,
        )
        margin = float(values["equivalence_margin"])
        zero_rows.append(
            {
                "core": core,
                "update_regime": regime,
                **summary,
                "ci90_low": ci90_low,
                "ci90_high": ci90_high,
                "equivalence_margin": margin,
                "equivalent_by_90_percent_interval": bool(
                    ci90_low > -margin and ci90_high < margin
                ),
                "n": len(wide),
            }
        )
    return pd.DataFrame(endpoint_rows), pd.DataFrame(zero_rows)


def aggregate() -> None:
    values, _ = read_configs()
    expected_panels = int(values["panel_count"])
    all_frames = []
    for design in ("synchronous", "strict"):
        files = sorted((RAW / design).glob("panel_*.csv.gz"))
        if len(files) != expected_panels:
            raise RuntimeError(
                f"Expected {expected_panels} {design} shards; found {len(files)}"
            )
        design_frame = pd.concat(
            [pd.read_csv(path) for path in files],
            ignore_index=True,
        )
        design_frame.to_csv(
            RAW / f"{design}_round_level.csv.gz",
            index=False,
            compression="gzip",
        )
        all_frames.append(design_frame)
    raw = pd.concat(all_frames, ignore_index=True)
    endpoints, zero = aggregate_design(raw)
    endpoints.to_csv(TABLES / "robustness_endpoints.csv", index=False)
    zero.to_csv(TABLES / "robustness_zero_drift.csv", index=False)

    synchronous = raw[raw.core == "confirmatory_10u5i"]
    differences = []
    for regime, subset in synchronous.groupby("update_regime"):
        if regime not in {"event_online", "round_synchronous"}:
            raise RuntimeError(f"Unexpected synchronous regime {regime}")
    unit_frames = []
    for regime, subset in synchronous.groupby("update_regime"):
        amplified = add_amplification(subset)
        amplified["update_regime"] = regime
        unit_frames.append(
            amplified[
                amplified["round"] == int(values["rounds"])
            ]
        )
    units = pd.concat(unit_frames, ignore_index=True)
    keys = [
        "panel",
        "assignment",
        "response_seed",
        "scenario",
        "intervention",
        "round",
    ]
    wide = units.pivot(
        index=keys,
        columns="update_regime",
        values="AA",
    ).reset_index()
    wide["synchronous_minus_sequential"] = (
        wide.round_synchronous - wide.event_online
    )
    for scenario, group in wide[
        wide.scenario != "zero"
    ].groupby("scenario"):
        summary = hierarchical_bootstrap_mean(
            group,
            "synchronous_minus_sequential",
            int(values["bootstrap_repetitions"]),
            20278400 + len(differences),
        )
        differences.append(
            {
                "scenario": scenario,
                **summary,
                "n": len(group),
            }
        )
    pd.DataFrame(differences).to_csv(
        TABLES / "synchronous_schedule_differences.csv",
        index=False,
    )

    accounting = (
        raw[
            (raw.branch == "closed")
            & (raw["round"] == int(values["rounds"]))
        ]
        .groupby(["core", "update_regime", "scenario"], as_index=False)
        .agg(
            accepted_interactions_mean=("accepted_interactions", "mean"),
            event_gradient_evaluations_mean=(
                "event_gradient_evaluations",
                "mean",
            ),
            parameter_update_operations_mean=(
                "parameter_update_operations",
                "mean",
            ),
            event_gradient_evaluations_min=(
                "event_gradient_evaluations",
                "min",
            ),
            event_gradient_evaluations_max=(
                "event_gradient_evaluations",
                "max",
            ),
            parameter_update_operations_min=(
                "parameter_update_operations",
                "min",
            ),
            parameter_update_operations_max=(
                "parameter_update_operations",
                "max",
            ),
        )
    )
    accounting.to_csv(TABLES / "update_accounting.csv", index=False)

    selected = pd.read_csv(STRICT_SELECTED_ASSIGNMENTS)
    selected[
        [
            "seed",
            "max_standardized_mean_difference",
            "max_theme_TV",
            "max_model_exposure_deviation",
            "MostPop_HA_gap",
            "BPR_HA_gap",
        ]
    ].to_csv(TABLES / "strict_assignment_balance.csv", index=False)

    (LOGS / "aggregate_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": len(raw),
                "targeted_grid": (
                    f"{values['panel_count']} panels x "
                    f"{values['assignment_count']} assignments x "
                    f"{values['response_seed_count']} response seeds"
                ),
                "bootstrap_repetitions": int(
                    values["bootstrap_repetitions"]
                ),
                "strict_core_results_pooled_with_confirmatory": False,
                "synchronous_comparison": (
                    "same recommendation schedule and event-gradient "
                    "contributions; sequential versus simultaneous "
                    "parameter application"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-strict")
    synchronous = subparsers.add_parser("synchronous")
    synchronous.add_argument("--panel", type=int, choices=[0, 1], required=True)
    strict = subparsers.add_parser("strict")
    strict.add_argument("--panel", type=int, choices=[0, 1], required=True)
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "prepare-strict":
        prepare_strict_core()
    elif args.command == "synchronous":
        run_panel(args.panel, "synchronous")
    elif args.command == "strict":
        run_panel(args.panel, "strict")
    elif args.command == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
