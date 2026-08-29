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

from src.bootstrap import (
    add_amplification,
    hierarchical_bootstrap_mean,
    paired_difference,
)
from src.bpr_training import load_or_train
from src.data_loading import (
    generate_assignment,
    load_data,
    make_candidate_pool,
    qualified_assignment_seeds,
    stratified_panel,
)
from src.lightgcn_cpu import LightGCNTrainingConfig, load_or_train_lightgcn
from src.lightgcn_simulation import (
    DynamicLightGCNConfig,
    simulate_paired_lightgcn,
)
from src.simulation import SimulationConfig, simulate_paired


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("JIIS_DATA_ROOT", ROOT / "data" / "derived"))
SPLIT = DATA_ROOT / "main_10u5i_leave_last_two_split.csv.gz"
ASSIGNMENT = DATA_ROOT / "model_aware_provenance_assignment_v2.csv"
ASSIGNMENT_CANDIDATES = (
    ROOT / "data" / "protocol" / "model_aware_assignment_candidates.csv"
)

CONFIGS = ROOT / "configs"
RAW = ROOT / "raw" / "extension"
TABLES = ROOT / "tables" / "extension"
LOGS = ROOT / "logs" / "extension"
ARTIFACTS = ROOT / "artifacts"
for directory in (RAW, TABLES, LOGS, ARTIFACTS):
    directory.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(min(8, os.cpu_count() or 4))


def read_configs() -> tuple[dict, dict, dict]:
    main = yaml.safe_load((CONFIGS / "main.yaml").read_text(encoding="utf-8"))
    bpr = yaml.safe_load((CONFIGS / "bpr.yaml").read_text(encoding="utf-8"))
    extension = yaml.safe_load(
        (CONFIGS / "extension.yaml").read_text(encoding="utf-8")
    )
    return main, bpr, extension


def lightgcn_training_config(values: dict) -> LightGCNTrainingConfig:
    return LightGCNTrainingConfig(
        dimension=int(values["dimension"]),
        layers=int(values["layers"]),
        learning_rate=float(values["static_learning_rate"]),
        regularization=float(values["static_regularization"]),
        optimization_passes=int(values["static_optimization_passes"]),
        sample_size=int(values["static_sample_size"]),
        seed=int(values["static_seed"]),
    )


def load_lightgcn_context(panel_index: int):
    main, _, extension = read_configs()
    values = extension["lightgcn"]
    data = load_data(SPLIT, ASSIGNMENT)
    raw, users, items = load_or_train_lightgcn(
        data,
        ARTIFACTS / "lightgcn_full_history_cpu.npz",
        LOGS / "lightgcn_full_history_training.csv",
        ARTIFACTS / "lightgcn_full_history_cpu_manifest.json",
        lightgcn_training_config(values),
    )
    panel = stratified_panel(
        data.user_activity,
        int(main["panel_size"]),
        int(main["global_seed"]) + 21000 + panel_index * 1000,
    )
    pools, _ = make_candidate_pool(
        data,
        users,
        items,
        panel,
        int(main["candidate_size"]),
        int(main["top_retrieval"]),
        int(main["global_seed"]) + 22000 + panel_index * 1000
        + int(main["candidate_size"]),
    )
    return main, values, data, raw, users, items, panel, pools


def load_bpr_context(panel_index: int):
    main, bpr, extension = read_configs()
    data = load_data(SPLIT, ASSIGNMENT)
    users, items = load_or_train(
        data,
        ARTIFACTS / "bpr_full_history.npz",
        LOGS / "bpr_training.csv",
        ARTIFACTS / "bpr_manifest.json",
        bpr,
    )
    panel = stratified_panel(
        data.user_activity,
        int(main["panel_size"]),
        int(main["global_seed"]) + 21000 + panel_index * 1000,
    )
    pools, _ = make_candidate_pool(
        data,
        users,
        items,
        panel,
        int(main["candidate_size"]),
        int(main["top_retrieval"]),
        int(main["global_seed"]) + 22000 + panel_index * 1000
        + int(main["candidate_size"]),
    )
    histories = [
        data.full_csr.indices[
            data.full_csr.indptr[user] : data.full_csr.indptr[user + 1]
        ].astype(np.int32)
        for user in panel
    ]
    return main, extension, data, users, items, panel, pools, histories


def random_streams(
    main: dict,
    rounds: int,
    panel_size: int,
    response_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(main["global_seed"]) + 30000 + response_seed)
    response = rng.random((rounds, panel_size), dtype=np.float32)
    negative = rng.random((rounds, panel_size), dtype=np.float32)
    replay = rng.random((rounds, panel_size), dtype=np.float32)
    orders = np.tile(
        np.arange(panel_size, dtype=np.int32),
        (rounds, 1),
    )
    return response, negative, replay, orders


def assignments(data, count: int) -> tuple[list[int], list[np.ndarray]]:
    seeds, metadata = qualified_assignment_seeds(
        ASSIGNMENT_CANDIDATES,
        count,
    )
    metadata.to_csv(
        RAW / f"qualified_assignment_metadata_{count}.csv",
        index=False,
    )
    labels = [
        generate_assignment(data.n_items, data.blocks, seed)
        for seed in seeds
    ]
    return seeds, labels


def run_lightgcn_panel(panel_index: int) -> None:
    started = time.time()
    main, values, data, raw, _, _, panel, pools = load_lightgcn_context(panel_index)
    assignment_seeds, label_sets = assignments(
        data,
        int(values["assignment_count"]),
    )
    rows = []
    for assignment_seed, labels in zip(assignment_seeds, label_sets):
        for response_seed in range(int(values["response_seed_count"])):
            response, negative, _, _ = random_streams(
                main,
                int(values["rounds"]),
                len(panel),
                response_seed,
            )
            for scenario, effect in values["scenarios"].items():
                for intervention in values["interventions"][scenario]:
                    frame = simulate_paired_lightgcn(
                        data,
                        raw,
                        panel,
                        pools,
                        labels,
                        np.asarray(effect, np.float32),
                        response,
                        negative,
                        DynamicLightGCNConfig(
                            rounds=int(values["rounds"]),
                            top_k=int(main["top_k"]),
                            learning_rate=float(values["dynamic_learning_rate"]),
                            regularization=float(values["dynamic_regularization"]),
                            update_steps=int(values["dynamic_steps"]),
                            outside_utility=float(main["outside_utility"]),
                            position_scale=float(main["position_scale"]),
                            minimum_score_sd=float(main["minimum_score_sd"]),
                            layers=int(values["layers"]),
                        ),
                        intervention=intervention,
                    )
                    frame.insert(0, "model", "LightGCN-Online")
                    frame.insert(1, "panel", panel_index)
                    frame.insert(2, "assignment", assignment_seed)
                    frame.insert(3, "response_seed", response_seed)
                    frame.insert(4, "scenario", scenario)
                    frame.insert(5, "intervention", intervention)
                    rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    output = RAW / "lightgcn" / f"panel_{panel_index}.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, compression="gzip")
    (LOGS / f"lightgcn_panel_{panel_index}.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": len(combined),
                "seconds": time.time() - started,
                "panel": panel_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_long_horizon_panel(panel_index: int, rounds: int) -> None:
    started = time.time()
    (
        main,
        extension,
        data,
        users,
        items,
        panel,
        pools,
        histories,
    ) = load_bpr_context(panel_index)
    values = extension["long_horizon"]
    regime_values = extension["update_regime"]
    assignment_seeds, label_sets = assignments(
        data,
        int(values["assignment_count"]),
    )
    rows = []
    for assignment_seed, labels in zip(assignment_seeds, label_sets):
        for response_seed in range(int(values["response_seed_count"])):
            # Draw one 24-round stream and take a prefix so that the 6-, 12-,
            # and 24-round analyses share identical stochastic histories over
            # their common intervals.
            maximum_rounds = max(int(value) for value in values["rounds"])
            response, negative, replay, orders = random_streams(
                main,
                maximum_rounds,
                len(panel),
                response_seed,
            )
            response = response[:rounds]
            negative = negative[:rounds]
            replay = replay[:rounds]
            orders = orders[:rounds]
            for scenario, effect in values["scenarios"].items():
                for regime in ("event_online", "fixed_budget_history_replay"):
                    config = SimulationConfig(
                        rounds=rounds,
                        top_k=int(main["top_k"]),
                        online_steps=int(main["online_steps"]),
                        online_learning_rate=float(main["online_learning_rate"]),
                        online_regularization=float(main["online_regularization"]),
                        outside_utility=float(main["outside_utility"]),
                        position_scale=float(main["position_scale"]),
                        minimum_score_sd=float(main["minimum_score_sd"]),
                        update_regime=regime,
                        clicked_steps=(
                            int(regime_values["clicked_steps"])
                            if regime == "fixed_budget_history_replay"
                            else int(main["online_steps"])
                        ),
                        replay_steps=(
                            int(regime_values["replay_steps"])
                            if regime == "fixed_budget_history_replay"
                            else 0
                        ),
                    )
                    frame = simulate_paired(
                        users,
                        items,
                        panel,
                        pools,
                        labels,
                        np.asarray(effect, np.float32),
                        response,
                        negative,
                        orders,
                        config,
                        intervention="none",
                        panel_histories=(
                            histories
                            if regime == "fixed_budget_history_replay"
                            else None
                        ),
                        replay_uniforms=(
                            replay
                            if regime == "fixed_budget_history_replay"
                            else None
                        ),
                    )
                    frame.insert(0, "model", "BPR-Online")
                    frame.insert(1, "panel", panel_index)
                    frame.insert(2, "assignment", assignment_seed)
                    frame.insert(3, "response_seed", response_seed)
                    frame.insert(4, "scenario", scenario)
                    frame.insert(5, "intervention", "none")
                    frame.insert(6, "update_regime", regime)
                    frame.insert(7, "horizon", rounds)
                    rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    output = RAW / "long_horizon" / f"rounds_{rounds}_panel_{panel_index}.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, compression="gzip")
    (LOGS / f"long_horizon_{rounds}_panel_{panel_index}.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": len(combined),
                "seconds": time.time() - started,
                "panel": panel_index,
                "rounds": rounds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def bootstrap_summary(
    frame: pd.DataFrame,
    value: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    return hierarchical_bootstrap_mean(
        frame,
        value,
        repetitions,
        seed,
    )


def derived_amplification(
    frame: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    keys = [
        "panel",
        "assignment",
        "response_seed",
        "scenario",
        "intervention",
        "round",
    ]
    if "update_regime" in frame.columns:
        keys.append("update_regime")
    if "horizon" in frame.columns:
        keys.append("horizon")
    wide = frame.pivot(index=keys, columns="branch", values=metric).reset_index()
    contrast = f"{metric}_closed_minus_frozen"
    wide[contrast] = wide["closed"] - wide["frozen"]
    neutral_keys = [
        key
        for key in keys
        if key not in {"scenario"}
    ]
    neutral = wide[wide.scenario == "zero"][
        neutral_keys + [contrast]
    ].rename(columns={contrast: f"{metric}_neutral_drift"})
    merged = wide.merge(
        neutral,
        on=neutral_keys,
        how="left",
        validate="many_to_one",
    )
    merged[f"{metric}_AA"] = (
        merged[contrast] - merged[f"{metric}_neutral_drift"]
    )
    return merged


def aggregate_lightgcn() -> None:
    _, _, extension = read_configs()
    expected = int(extension["lightgcn"]["panel_count"])
    files = sorted((RAW / "lightgcn").glob("panel_*.csv.gz"))
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} LightGCN shards; found {len(files)}")
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    raw.to_csv(
        RAW / "lightgcn_round_level.csv.gz",
        index=False,
        compression="gzip",
    )
    amplification = add_amplification(raw)
    share_amplification = derived_amplification(raw, "human_ai_share_gap_pp")
    final_round = int(extension["lightgcn"]["rounds"])
    final = amplification[amplification["round"] == final_round]
    final_shares = share_amplification[
        share_amplification["round"] == final_round
    ]
    rows = []
    for (scenario, intervention), group in final.groupby(
        ["scenario", "intervention"]
    ):
        summary = bootstrap_summary(group, "AA", 2000, 20267100 + len(rows))
        share_group = final_shares[
            (final_shares.scenario == scenario)
            & (final_shares.intervention == intervention)
        ]
        share_summary = bootstrap_summary(
            share_group,
            "human_ai_share_gap_pp_AA",
            2000,
            20268100 + len(rows),
        )
        rows.append(
            {
                "scenario": scenario,
                "intervention": intervention,
                **summary,
                "relative_ratio_percent": 100.0 * np.expm1(summary["estimate"]),
                "share_gap_pp_estimate": share_summary["estimate"],
                "share_gap_pp_ci_low": share_summary["ci_low"],
                "share_gap_pp_ci_high": share_summary["ci_high"],
                "n": len(group),
            }
        )
    amplification_table = pd.DataFrame(rows)
    baseline = amplification_table[
        amplification_table.intervention == "none"
    ][["scenario", "estimate"]].rename(columns={"estimate": "baseline_estimate"})
    amplification_table = amplification_table.merge(
        baseline,
        on="scenario",
        how="left",
        validate="many_to_one",
    )
    amplification_table["intervention_reduction_percent"] = np.where(
        (amplification_table.intervention == "combined")
        & (amplification_table.baseline_estimate.abs() > 1e-12),
        100.0
        * (
            1.0
            - amplification_table.estimate
            / amplification_table.baseline_estimate
        ),
        np.nan,
    )
    amplification_table.to_csv(
        TABLES / "lightgcn_dynamic_amplification.csv",
        index=False,
    )

    zero = final[
        (final.scenario == "zero")
        & (final.intervention == "none")
    ]
    zero_summary = bootstrap_summary(
        zero,
        "closed_minus_frozen",
        5000,
        20269100,
    )
    pd.DataFrame(
        [
            {
                **zero_summary,
                "equivalence_margin": 0.02,
                "equivalent_by_95_percent_interval": bool(
                    zero_summary["ci_low"] > -0.02
                    and zero_summary["ci_high"] < 0.02
                ),
                "n": len(zero),
            }
        ]
    ).to_csv(TABLES / "lightgcn_zero_drift.csv", index=False)

    final_closed = raw[
        (raw["round"] == final_round)
        & (raw.branch == "closed")
    ]
    metrics = [
        "ctr",
        "candidate_anchored_utility",
        "coverage",
        "gini",
        "human_ai_share_gap_pp",
    ]
    outcome_rows = []
    for (scenario, intervention), group in final_closed.groupby(
        ["scenario", "intervention"]
    ):
        row = {
            "scenario": scenario,
            "intervention": intervention,
            "n": len(group),
        }
        for metric in metrics:
            summary = bootstrap_summary(
                group,
                metric,
                2000,
                20269200 + len(outcome_rows) * 10 + metrics.index(metric),
            )
            row.update(
                {
                    f"{metric}_estimate": summary["estimate"],
                    f"{metric}_ci_low": summary["ci_low"],
                    f"{metric}_ci_high": summary["ci_high"],
                }
            )
        if intervention == "combined":
            scenario_frame = final_closed[
                final_closed.scenario == scenario
            ]
            for metric in metrics:
                paired = paired_difference(
                    scenario_frame,
                    metric,
                    "combined",
                )
                delta = bootstrap_summary(
                    paired,
                    f"delta_{metric}",
                    2000,
                    20269300
                    + len(outcome_rows) * 10
                    + metrics.index(metric),
                )
                row.update(
                    {
                        f"delta_{metric}_estimate": delta["estimate"],
                        f"delta_{metric}_ci_low": delta["ci_low"],
                        f"delta_{metric}_ci_high": delta["ci_high"],
                    }
                )
        outcome_rows.append(row)
    pd.DataFrame(outcome_rows).to_csv(
        TABLES / "lightgcn_dynamic_outcomes.csv",
        index=False,
    )


def aggregate_long_horizon() -> None:
    _, _, extension = read_configs()
    expected_panels = int(extension["long_horizon"]["panel_count"])
    horizons = [int(value) for value in extension["long_horizon"]["rounds"]]
    files = []
    for horizon in horizons:
        selected = sorted(
            (RAW / "long_horizon").glob(f"rounds_{horizon}_panel_*.csv.gz")
        )
        if len(selected) != expected_panels:
            raise RuntimeError(
                f"Expected {expected_panels} shards for {horizon}; found {len(selected)}"
            )
        files.extend(selected)
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    raw.to_csv(
        RAW / "long_horizon_round_level.csv.gz",
        index=False,
        compression="gzip",
    )
    rows = []
    trajectories = []
    zero_rows = []
    final_units = []
    for (horizon, regime), subset in raw.groupby(["horizon", "update_regime"]):
        amplified = add_amplification(subset)
        amplified.insert(0, "horizon", horizon)
        amplified.insert(1, "update_regime", regime)
        share_amplified = derived_amplification(
            subset,
            "human_ai_share_gap_pp",
        )
        for (round_number, scenario), group in amplified.groupby(
            ["round", "scenario"]
        ):
            summary = bootstrap_summary(
                group,
                "AA",
                2000,
                20270100 + len(trajectories),
            )
            trajectories.append(
                {
                    "horizon": horizon,
                    "update_regime": regime,
                    "round": round_number,
                    "scenario": scenario,
                    **summary,
                    "relative_ratio_percent": 100.0
                    * np.expm1(summary["estimate"]),
                    "n": len(group),
                }
            )
        final = amplified[amplified["round"] == horizon]
        final_units.append(final)
        final_shares = share_amplified[
            share_amplified["round"] == horizon
        ]
        for scenario, group in final.groupby("scenario"):
            summary = bootstrap_summary(
                group,
                "AA",
                2000,
                20271100 + len(rows),
            )
            share_group = final_shares[
                final_shares.scenario == scenario
            ]
            share_summary = bootstrap_summary(
                share_group,
                "human_ai_share_gap_pp_AA",
                2000,
                20271200 + len(rows),
            )
            rows.append(
                {
                    "horizon": horizon,
                    "update_regime": regime,
                    "scenario": scenario,
                    **summary,
                    "relative_ratio_percent": 100.0
                    * np.expm1(summary["estimate"]),
                    "share_gap_pp_estimate": share_summary["estimate"],
                    "share_gap_pp_ci_low": share_summary["ci_low"],
                    "share_gap_pp_ci_high": share_summary["ci_high"],
                    "n": len(group),
                }
            )
        zero = final[final.scenario == "zero"]
        zero_summary = bootstrap_summary(
            zero,
            "closed_minus_frozen",
            2000,
            20271300 + len(zero_rows),
        )
        zero_rows.append(
            {
                "horizon": horizon,
                "update_regime": regime,
                **zero_summary,
                "equivalence_margin": 0.02,
                "equivalent_by_95_percent_interval": bool(
                    zero_summary["ci_low"] > -0.02
                    and zero_summary["ci_high"] < 0.02
                ),
                "n": len(zero),
            }
        )
    pd.DataFrame(rows).to_csv(
        TABLES / "long_horizon_endpoints.csv",
        index=False,
    )
    pd.DataFrame(trajectories).to_csv(
        TABLES / "long_horizon_trajectories.csv",
        index=False,
    )
    pd.DataFrame(zero_rows).to_csv(
        TABLES / "long_horizon_zero_drift.csv",
        index=False,
    )

    final_unit_frame = pd.concat(final_units, ignore_index=True)
    regime_keys = [
        "panel",
        "assignment",
        "response_seed",
        "scenario",
        "intervention",
        "round",
        "horizon",
    ]
    regime_wide = final_unit_frame.pivot(
        index=regime_keys,
        columns="update_regime",
        values="AA",
    ).reset_index()
    regime_wide["replay_minus_event_online"] = (
        regime_wide["fixed_budget_history_replay"]
        - regime_wide["event_online"]
    )
    difference_rows = []
    for (horizon, scenario), group in regime_wide.groupby(
        ["horizon", "scenario"]
    ):
        summary = bootstrap_summary(
            group,
            "replay_minus_event_online",
            2000,
            20271400 + len(difference_rows),
        )
        difference_rows.append(
            {
                "horizon": horizon,
                "scenario": scenario,
                **summary,
                "n": len(group),
            }
        )
    pd.DataFrame(difference_rows).to_csv(
        TABLES / "long_horizon_regime_differences.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train-lightgcn")
    parser_lightgcn = subparsers.add_parser("lightgcn")
    parser_lightgcn.add_argument("--panel", type=int, choices=[0, 1], required=True)
    parser_horizon = subparsers.add_parser("long-horizon")
    parser_horizon.add_argument("--panel", type=int, choices=[0, 1], required=True)
    parser_horizon.add_argument("--rounds", type=int, choices=[6, 12, 24], required=True)
    subparsers.add_parser("aggregate-lightgcn")
    subparsers.add_parser("aggregate-long-horizon")
    args = parser.parse_args()

    if args.command == "train-lightgcn":
        load_lightgcn_context(0)
    elif args.command == "lightgcn":
        run_lightgcn_panel(args.panel)
    elif args.command == "long-horizon":
        run_long_horizon_panel(args.panel, args.rounds)
    elif args.command == "aggregate-lightgcn":
        aggregate_lightgcn()
    elif args.command == "aggregate-long-horizon":
        aggregate_long_horizon()


if __name__ == "__main__":
    main()
