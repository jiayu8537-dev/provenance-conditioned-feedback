#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.bootstrap import add_amplification, hierarchical_bootstrap_mean
from src.simulation import SimulationConfig, simulate_paired


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("JIIS_DATA_ROOT", ROOT / "data" / "derived"))
SPLIT = DATA_ROOT / "main_10u5i_leave_last_two_split.csv.gz"
ASSIGNMENT = DATA_ROOT / "model_aware_provenance_assignment_v2.csv"
ASSIGNMENT_CANDIDATES = (
    ROOT / "data" / "protocol" / "model_aware_assignment_candidates.csv"
)

MAIN = {
    "global_seed": 20260720,
    "panel_size": 150,
    "rounds": 6,
    "top_k": 20,
    "top_retrieval": 150,
    "online_steps": 3,
    "online_learning_rate": 0.01,
    "online_regularization": 0.0001,
    "minimum_score_sd": 1.0e-6,
}
SENSITIVITY = {"assignment_count": 5, "response_seed_count": 5}
SCENARIO_EFFECTS = {
    "zero": [0.00, 0.00, 0.00],
    "moderate_asymmetric": [-0.15, -0.05, 0.05],
    "strong_premium_penalty": [-0.25, 0.00, 0.15],
    "ai_appreciation": [0.10, 0.05, 0.00],
}

RAW = ROOT / "raw" / "choice_process"
TABLES = ROOT / "tables" / "choice_process"
LOGS = ROOT / "logs" / "choice_process"
for directory in (RAW, TABLES, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "zero",
    "moderate_asymmetric",
    "strong_premium_penalty",
    "ai_appreciation",
]

SPECIFICATIONS = {
    "position_0p4": (0.4, 0.5),
    "position_1p2": (1.2, 0.5),
    "outside_0p0": (0.8, 0.0),
    "outside_1p0": (0.8, 1.0),
}


@dataclass
class ContextData:
    n_items: int
    blocks: list[np.ndarray]
    item_counts: np.ndarray
    warm_items: np.ndarray
    popularity_bins: np.ndarray
    user_activity: np.ndarray
    seen_by_user: list[np.ndarray]


def stratified_panel(activity: np.ndarray, panel_size: int, seed: int) -> np.ndarray:
    bins = pd.qcut(
        pd.Series(activity).rank(method="first"),
        5,
        labels=False,
        duplicates="drop",
    ).to_numpy()
    rng = np.random.default_rng(seed)
    panel: list[int] = []
    for group in range(5):
        eligible = np.where(bins == group)[0]
        panel.extend(
            rng.choice(eligible, panel_size // 5, replace=False).astype(int).tolist()
        )
    return np.asarray(sorted(panel), dtype=np.int32)


def make_candidate_pool(
    data: ContextData,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    panel: np.ndarray,
    candidate_size: int,
    top_retrieval: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pools = np.empty((len(panel), candidate_size), dtype=np.int32)
    score_matrix = user_embeddings[panel] @ item_embeddings.T
    random_needed = candidate_size - top_retrieval
    for row, user in enumerate(panel):
        seen = data.seen_by_user[int(user)]
        if len(seen):
            score_matrix[row, seen] = -np.inf
        top = np.argpartition(score_matrix[row], -top_retrieval)[-top_retrieval:]
        top = top[np.argsort(-score_matrix[row, top], kind="stable")]
        used = set(seen.astype(int).tolist()) | set(top.astype(int).tolist())
        random_items: list[int] = []
        per_bin = int(np.ceil(random_needed / 5))
        for pop_bin in range(5):
            population = np.where(
                (data.popularity_bins == pop_bin) & (data.item_counts > 0)
            )[0]
            eligible = np.asarray(
                [item for item in population if int(item) not in used],
                dtype=np.int32,
            )
            take = min(per_bin, len(eligible))
            if take:
                values = rng.choice(eligible, take, replace=False).astype(int).tolist()
                random_items.extend(values)
                used.update(values)
        if len(random_items) < random_needed:
            eligible = np.asarray(
                [item for item in data.warm_items if int(item) not in used],
                dtype=np.int32,
            )
            values = rng.choice(
                eligible, random_needed - len(random_items), replace=False
            ).astype(int).tolist()
            random_items.extend(values)
        pools[row] = np.concatenate(
            [top.astype(np.int32), np.asarray(random_items[:random_needed], np.int32)]
        )
    return pools


def context(candidate_size: int, panel_index: int):
    split = pd.read_csv(SPLIT)
    assignment = pd.read_csv(ASSIGNMENT)
    users = sorted(split.user_id.unique().tolist())
    items = sorted(assignment.item_id.unique().tolist())
    user_to_index = {value: idx for idx, value in enumerate(users)}
    item_to_index = {value: idx for idx, value in enumerate(items)}
    split = split[split.item_id.isin(item_to_index)].copy()
    split["u"] = split.user_id.map(user_to_index).astype(np.int32)
    split["i"] = split.item_id.map(item_to_index).astype(np.int32)
    assignment = assignment.set_index("item_id").loc[items].reset_index()
    n_users, n_items = len(users), len(items)
    all_users = split.u.to_numpy(np.int32)
    all_items = split.i.to_numpy(np.int32)
    item_counts = np.bincount(all_items, minlength=n_items).astype(np.float32)
    popularity_bins = pd.qcut(
        pd.Series(item_counts).rank(method="first"),
        5,
        labels=False,
        duplicates="drop",
    ).to_numpy(np.int8)
    blocks = [
        np.asarray(indices, dtype=np.int32)
        for indices in assignment.groupby("model_aware_block", sort=True).indices.values()
    ]
    grouped = split.groupby("u", sort=True)["i"].agg(list)
    seen_by_user = [np.empty(0, dtype=np.int32) for _ in range(n_users)]
    for user, values in grouped.items():
        seen_by_user[int(user)] = np.asarray(values, dtype=np.int32)
    data = ContextData(
        n_items=n_items,
        blocks=blocks,
        item_counts=item_counts,
        warm_items=np.where(item_counts > 0)[0].astype(np.int32),
        popularity_bins=popularity_bins,
        user_activity=np.bincount(all_users, minlength=n_users).astype(np.int32),
        seen_by_user=seen_by_user,
    )
    artifact = np.load(ROOT / "artifacts" / "bpr_full_history.npz")
    users_embedding = artifact["U"].astype(np.float32)
    items_embedding = artifact["V"].astype(np.float32)
    panel = stratified_panel(
        data.user_activity,
        int(MAIN["panel_size"]),
        int(MAIN["global_seed"]) + 21000 + panel_index * 1000,
    )
    pools = make_candidate_pool(
        data,
        users_embedding,
        items_embedding,
        panel,
        candidate_size,
        int(MAIN["top_retrieval"]),
        int(MAIN["global_seed"]) + 22000 + panel_index * 1000 + candidate_size,
    )
    return MAIN, SENSITIVITY, data, users_embedding, items_embedding, panel, pools


def generate_assignment(n_items: int, blocks: list[np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.empty(n_items, np.uint8)
    global_counts = np.zeros(3, int)
    shuffled_blocks = list(blocks)
    rng.shuffle(shuffled_blocks)
    for original_indices in shuffled_blocks:
        indices = original_indices.copy()
        rng.shuffle(indices)
        base, remainder = divmod(len(indices), 3)
        quota = np.full(3, base, int)
        if remainder:
            tie = rng.random(3)
            order = np.lexsort((tie, global_counts))
            quota[order[:remainder]] += 1
        values = np.concatenate(
            [np.full(quota[group], group, np.uint8) for group in range(3)]
        )
        rng.shuffle(values)
        labels[indices] = values
        global_counts += quota
    return labels


def assignment_setup(data: ContextData, count: int):
    candidates = pd.read_csv(ASSIGNMENT_CANDIDATES)
    qualified = candidates[
        (candidates.max_standardized_mean_difference < 0.05)
        & (candidates.max_theme_TV < 0.05)
        & (candidates.max_model_exposure_deviation < 0.10)
    ].sort_values("seed")
    selected = qualified.iloc[np.linspace(0, len(qualified) - 1, count, dtype=int)]
    seeds = selected.seed.astype(int).tolist()
    return seeds, [generate_assignment(data.n_items, data.blocks, seed) for seed in seeds]


def streams(main: dict, panel_size: int, response_seed: int):
    rng = np.random.default_rng(int(main["global_seed"]) + 30000 + response_seed)
    uniforms = rng.random((int(main["rounds"]), panel_size), dtype=np.float32)
    negative = rng.random((int(main["rounds"]), panel_size), dtype=np.float32)
    orders = np.tile(np.arange(panel_size, dtype=np.int32), (int(main["rounds"]), 1))
    return uniforms, negative, orders


def effects(main: dict, scenario: str, response_seed: int, panel_size: int):
    return np.asarray(SCENARIO_EFFECTS[scenario], dtype=np.float32)


def simulation_config(main: dict, position_scale: float, outside_utility: float):
    return SimulationConfig(
        rounds=int(main["rounds"]),
        top_k=int(main["top_k"]),
        online_steps=int(main["online_steps"]),
        online_learning_rate=float(main["online_learning_rate"]),
        online_regularization=float(main["online_regularization"]),
        outside_utility=float(outside_utility),
        position_scale=float(position_scale),
        minimum_score_sd=float(main["minimum_score_sd"]),
    )


def run_panel(panel_index: int) -> None:
    started = time.time()
    main, sensitivity, data, users, items, panel, pools = context(240, panel_index)
    assignment_seeds, labels_list = assignment_setup(
        data, int(sensitivity["assignment_count"])
    )
    for name, (position_scale, outside_utility) in SPECIFICATIONS.items():
        rows = []
        for assignment_seed, labels in zip(assignment_seeds, labels_list):
            for response_seed in range(int(sensitivity["response_seed_count"])):
                uniforms, negative, orders = streams(main, len(panel), response_seed)
                for scenario in SCENARIOS:
                    delta = effects(main, scenario, response_seed, len(panel))
                    frame = simulate_paired(
                        users,
                        items,
                        panel,
                        pools,
                        labels,
                        delta,
                        uniforms,
                        negative,
                        orders,
                        simulation_config(main, position_scale, outside_utility),
                        "none",
                        1.0,
                    )
                    frame.insert(0, "panel", panel_index)
                    frame.insert(1, "assignment", assignment_seed)
                    frame.insert(2, "response_seed", response_seed)
                    frame.insert(3, "scenario", scenario)
                    frame.insert(4, "intervention", "none")
                    frame.insert(5, "position_scale", position_scale)
                    frame.insert(6, "outside_utility", outside_utility)
                    frame.insert(7, "specification", name)
                    rows.append(frame)
        combined = pd.concat(rows, ignore_index=True)
        output = RAW / f"{name}_panel_{panel_index}.csv.gz"
        combined.to_csv(output, index=False, compression="gzip")
        print(
            json.dumps(
                {
                    "status": "complete",
                    "panel": panel_index,
                    "specification": name,
                    "rows": len(combined),
                    "elapsed_seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
    (LOGS / f"panel_{panel_index}.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "panel": panel_index,
                "seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize(group: pd.DataFrame, value: str, seed: int) -> dict:
    result = hierarchical_bootstrap_mean(group, value, 2000, seed)
    return {f"{value}_{key}": val for key, val in result.items()}


def aggregate() -> None:
    frames = [pd.read_csv(path) for path in sorted(RAW.glob("*.csv.gz"))]

    baseline_paths = sorted(
        (ROOT / "raw" / "sensitivity" / "candidate").glob(
            "value_240p0_panel_*.csv.gz"
        )
    )
    if len(baseline_paths) != 2:
        raise RuntimeError("Expected two packaged baseline sensitivity shards")
    baseline = pd.concat([pd.read_csv(path) for path in baseline_paths], ignore_index=True)
    baseline = baseline[baseline["scenario"].isin(SCENARIOS)].copy()
    baseline["position_scale"] = 0.8
    baseline["outside_utility"] = 0.5
    baseline["specification"] = "baseline"
    frames.append(baseline)

    raw = pd.concat(frames, ignore_index=True)
    rows = []
    for spec_index, (specification, spec) in enumerate(
        raw.groupby("specification", sort=True)
    ):
        amplified = add_amplification(spec)
        final_aa = amplified[
            (amplified["round"] == 6)
            & (amplified["scenario"].isin(SCENARIOS[1:]))
        ]
        final_ctr = spec[
            (spec["round"] == 6)
            & (spec["branch"] == "closed")
            & (spec["scenario"].isin(SCENARIOS[1:]))
        ]
        for scenario_index, scenario in enumerate(SCENARIOS[1:]):
            aa = final_aa[final_aa["scenario"] == scenario]
            ctr = final_ctr[final_ctr["scenario"] == scenario]
            row = {
                "specification": specification,
                "position_scale": float(spec["position_scale"].iloc[0]),
                "outside_utility": float(spec["outside_utility"].iloc[0]),
                "scenario": scenario,
                "simulation_cells": len(aa),
            }
            row.update(summarize(aa, "AA", 20268100 + 10 * spec_index + scenario_index))
            row.update(
                summarize(ctr, "ctr", 20268200 + 10 * spec_index + scenario_index)
            )
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["position_scale", "outside_utility", "scenario"]
    )
    result.to_csv(TABLES / "choice_process_sensitivity.csv", index=False)

    ordering = []
    for specification, group in result.groupby("specification"):
        values = group.set_index("scenario")["AA_estimate"]
        ordering.append(
            {
                "specification": specification,
                "moderate_positive": bool(values["moderate_asymmetric"] > 0),
                "strong_exceeds_moderate": bool(
                    values["strong_premium_penalty"]
                    > values["moderate_asymmetric"]
                ),
                "ai_appreciation_negative": bool(values["ai_appreciation"] < 0),
            }
        )
    pd.DataFrame(ordering).to_csv(
        TABLES / "choice_process_ordering_checks.csv", index=False
    )
    print(result.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    panel = sub.add_parser("panel")
    panel.add_argument("--panel", type=int, required=True, choices=[0, 1])
    sub.add_parser("aggregate")
    sub.add_parser("all")
    args = parser.parse_args()
    if args.command == "panel":
        run_panel(args.panel)
    elif args.command == "aggregate":
        aggregate()
    else:
        run_panel(0)
        run_panel(1)
        aggregate()


if __name__ == "__main__":
    main()
