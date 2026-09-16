#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT
sys.path.insert(0, str(PACKAGE))

from src.bootstrap import add_amplification, hierarchical_bootstrap_mean  # noqa: E402


OUT = ROOT
RAW = OUT / "raw" / "added_sensitivity"
TABLES = OUT / "tables" / "added_sensitivity"
LOGS = OUT / "logs" / "added_sensitivity"
for directory in (RAW, TABLES, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["zero", "moderate_asymmetric", "strong_premium_penalty", "ai_appreciation"]
TARGETS = [0.10, 0.20, 0.40]
EXPANDED_PANELS = tuple(range(5))
EXPANDED_ENDPOINT_BOOTSTRAP_SEED = 20269301
EXPANDED_PAIRED_LMC_BOOTSTRAP_SEED = 20269401
EXPANDED_PAIRED_ABSOLUTE_LMC_BOOTSTRAP_SEED = 20269402
BOOTSTRAP_REPETITIONS = 2000


def load_runtime():
    """Load source-dependent simulation components only for full reruns."""
    global assignment_setup, context, effects, read_configs, streams
    global choice_from_uniform, position_effect, softmax_with_outside
    global make_candidate_pool, stratified_panel, cumulative_metrics, update_pair
    global apply_anchor, candidate_anchor
    from run_all_cpu import assignment_setup, context, effects, read_configs, streams
    from src.choice_model import choice_from_uniform, position_effect, softmax_with_outside
    from src.data_loading import make_candidate_pool, stratified_panel
    from src.exposure_metrics import cumulative_metrics
    from src.online_bpr import update_pair
    from src.score_normalization import apply_anchor, candidate_anchor


def _initial_acceptance_components(U, V, pools, top_k, position_scale, minimum_sd):
    scores = np.einsum("ud,umd->um", U, V[pools], optimize=True).astype(np.float32)
    means, sds = candidate_anchor(scores, minimum_sd)
    anchored = apply_anchor(scores, means, sds)
    top = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    row = np.arange(len(U))[:, None]
    top_scores = scores[row, top]
    order = np.argsort(-top_scores, axis=1, kind="stable")
    top = top[row, order]
    displayed_anchored = anchored[row, top]
    utilities = displayed_anchored + position_effect(top_k, position_scale)[None, :]
    shifted = utilities - utilities.max(axis=1, keepdims=True)
    log_sum = utilities.max(axis=1) + np.log(np.exp(shifted).sum(axis=1))
    return log_sum


def calibrate_outside_utilities():
    from scipy.optimize import brentq
    load_runtime()
    main, _, data, U, V, _, _ = context(240, 0)
    calibration_panel = stratified_panel(
        data.user_activity, 500, int(main["global_seed"]) + 91000
    )
    pools, _ = make_candidate_pool(
        data, U, V, calibration_panel, 240, 150, int(main["global_seed"]) + 92000
    )
    log_sums = _initial_acceptance_components(
        U[calibration_panel], V, pools, int(main["top_k"]),
        float(main["position_scale"]), float(main["minimum_score_sd"]),
    )

    def average_acceptance(outside):
        # S/(exp(outside)+S), evaluated stably on the log scale.
        return float(np.mean(1.0 / (1.0 + np.exp(outside - log_sums))))

    rows = []
    for target in TARGETS:
        outside = brentq(lambda x: average_acceptance(x) - target, -20.0, 30.0)
        rows.append({
            "target_initial_acceptance": target,
            "outside_utility": outside,
            "calibrated_initial_acceptance": average_acceptance(outside),
            "calibration_users": len(calibration_panel),
            "calibration_seed": int(main["global_seed"]) + 91000,
            "candidate_seed": int(main["global_seed"]) + 92000,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "outside_utility_calibration.csv", index=False)
    return frame


def run_low_acceptance():
    from src.simulation import SimulationConfig, simulate_paired

    load_runtime()
    main, _, _ = read_configs()
    calibration = calibrate_outside_utilities()
    rows = []
    started = time.time()
    for panel_index in (0, 1):
        _, _, data, U, V, panel, pools = context(240, panel_index)
        seeds, labels_list = assignment_setup(data, 5)
        for assignment_seed, labels in zip(seeds, labels_list):
            for response_seed in range(5):
                response, negative, orders = streams(main, len(panel), response_seed)
                for record in calibration.to_dict("records"):
                    config = SimulationConfig(
                        rounds=6,
                        top_k=20,
                        online_steps=3,
                        online_learning_rate=float(main["online_learning_rate"]),
                        online_regularization=float(main["online_regularization"]),
                        outside_utility=float(record["outside_utility"]),
                        position_scale=float(main["position_scale"]),
                        minimum_score_sd=float(main["minimum_score_sd"]),
                    )
                    for scenario in SCENARIOS:
                        result = simulate_paired(
                            U, V, panel, pools, labels,
                            effects(main, scenario, response_seed, len(panel)),
                            response, negative, orders, config,
                        )
                        result.insert(0, "panel", panel_index)
                        result.insert(1, "assignment", assignment_seed)
                        result.insert(2, "response_seed", response_seed)
                        result.insert(3, "scenario", scenario)
                        result.insert(4, "target_initial_acceptance", record["target_initial_acceptance"])
                        result.insert(5, "outside_utility", record["outside_utility"])
                        rows.append(result)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(RAW / "lower_acceptance_round_level.csv.gz", index=False, compression="gzip")
    (LOGS / "lower_acceptance_run.json").write_text(json.dumps({
        "status": "complete", "rows": len(raw), "seconds": time.time() - started,
        "panels": 2, "assignments": 5, "response_seeds": 5,
        "scenarios": SCENARIOS, "targets": TARGETS,
    }, indent=2))
    return raw


def run_expanded_low_acceptance():
    """Run the lower-feedback scenarios on five panels with a matched high-feedback reference."""
    from src.simulation import SimulationConfig, simulate_paired

    load_runtime()
    main, _, _ = read_configs()
    calibration = calibrate_outside_utilities()
    conditions = [
        {
            "feedback_condition": f"target_{int(100 * float(record['target_initial_acceptance']))}",
            "target_initial_acceptance": float(record["target_initial_acceptance"]),
            "outside_utility": float(record["outside_utility"]),
        }
        for record in calibration.to_dict("records")
    ]
    conditions.append({
        "feedback_condition": "high_feedback_reference",
        "target_initial_acceptance": np.nan,
        "outside_utility": float(main["outside_utility"]),
    })

    rows = []
    started = time.time()
    for panel_index in EXPANDED_PANELS:
        _, _, data, U, V, panel, pools = context(240, panel_index)
        seeds, labels_list = assignment_setup(data, 5)
        for assignment_seed, labels in zip(seeds, labels_list):
            for response_seed in range(5):
                response, negative, orders = streams(main, len(panel), response_seed)
                for condition in conditions:
                    config = SimulationConfig(
                        rounds=6,
                        top_k=20,
                        online_steps=3,
                        online_learning_rate=float(main["online_learning_rate"]),
                        online_regularization=float(main["online_regularization"]),
                        outside_utility=float(condition["outside_utility"]),
                        position_scale=float(main["position_scale"]),
                        minimum_score_sd=float(main["minimum_score_sd"]),
                    )
                    for scenario in SCENARIOS:
                        result = simulate_paired(
                            U, V, panel, pools, labels,
                            effects(main, scenario, response_seed, len(panel)),
                            response, negative, orders, config,
                        )
                        result.insert(0, "panel", panel_index)
                        result.insert(1, "assignment", assignment_seed)
                        result.insert(2, "response_seed", response_seed)
                        result.insert(3, "scenario", scenario)
                        result.insert(4, "feedback_condition", condition["feedback_condition"])
                        result.insert(5, "target_initial_acceptance", condition["target_initial_acceptance"])
                        result.insert(6, "outside_utility", condition["outside_utility"])
                        rows.append(result)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(
        RAW / "lower_acceptance_expanded_round_level.csv.gz",
        index=False,
        compression="gzip",
    )
    (LOGS / "lower_acceptance_expanded_run.json").write_text(json.dumps({
        "status": "complete",
        "rows": len(raw),
        "seconds": time.time() - started,
        "panels": len(EXPANDED_PANELS),
        "panel_indices": list(EXPANDED_PANELS),
        "assignments": 5,
        "response_seeds": 5,
        "scenarios": SCENARIOS,
        "targets": TARGETS,
        "matched_high_feedback_reference": True,
        "high_feedback_outside_utility": float(main["outside_utility"]),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seeds": {
            "endpoint_base": EXPANDED_ENDPOINT_BOOTSTRAP_SEED,
            "paired_lmc_base": EXPANDED_PAIRED_LMC_BOOTSTRAP_SEED,
            "paired_absolute_lmc_base": EXPANDED_PAIRED_ABSOLUTE_LMC_BOOTSTRAP_SEED,
        },
    }, indent=2))
    return raw


def _warm_draw_orders(data, panel_size, round_index, panel_index, response_seed, global_seed):
    orders = []
    base = global_seed + 88000 + panel_index * 10000 + response_seed * 100 + round_index
    for row in range(panel_size):
        row_orders = []
        for pop_bin in range(5):
            population = np.where((data.popularity_bins == pop_bin) & (data.item_counts > 0))[0]
            rng = np.random.default_rng(base + row * 17 + pop_bin)
            take = min(120, len(population))
            row_orders.append(rng.choice(population, take, replace=False).astype(np.int32))
        orders.append(row_orders)
    return orders


def _dynamic_pools(data, U, V, panel, accepted_sets, warm_orders, top_retrieval=150, candidate_size=240):
    scores = U @ V.T
    pools = np.empty((len(panel), candidate_size), dtype=np.int32)
    for row, user in enumerate(panel):
        history = data.full_csr.indices[data.full_csr.indptr[user]:data.full_csr.indptr[user + 1]]
        scores[row, history] = -np.inf
        if accepted_sets[row]:
            scores[row, np.fromiter(accepted_sets[row], dtype=np.int32)] = -np.inf
        top = np.argpartition(scores[row], -top_retrieval)[-top_retrieval:]
        top = top[np.argsort(-scores[row, top], kind="stable")]
        used = set(history.astype(int).tolist()) | accepted_sets[row] | set(top.astype(int).tolist())
        random_items = []
        per_bin = math.ceil((candidate_size - top_retrieval) / 5)
        for pop_bin in range(5):
            chosen = []
            for item in warm_orders[row][pop_bin]:
                item = int(item)
                if item not in used:
                    chosen.append(item)
                    used.add(item)
                    if len(chosen) == per_bin:
                        break
            if len(chosen) < per_bin:
                raise RuntimeError("Warm candidate draw reserve exhausted")
            random_items.extend(chosen)
        candidates = np.concatenate([top.astype(np.int32), np.asarray(random_items[:candidate_size-top_retrieval], np.int32)])
        if len(np.unique(candidates)) != candidate_size:
            raise RuntimeError("Dynamic candidate construction produced duplicates")
        pools[row] = candidates
    return pools, scores


def simulate_dynamic(
    data, initial_U, initial_V, panel, initial_pools, labels, scenario_effects,
    response_uniforms, negative_uniforms, user_orders, main, panel_index, response_seed,
):
    rows = []
    positions = position_effect(20, float(main["position_scale"]))
    discount = 1.0 / np.log2(np.arange(2, 22, dtype=float))
    for branch in ("frozen", "closed"):
        U = initial_U[panel].copy()
        V = initial_V.copy()
        accepted_sets = [set() for _ in panel]
        exposure = np.zeros(data.n_items, dtype=np.float64)
        clicks = impressions = 0
        utility_sum = 0.0
        previous_pools = None
        for round_index in range(6):
            if round_index == 0:
                pools = initial_pools.copy()
                full_scores = None
            else:
                warm_orders = _warm_draw_orders(
                    data, len(panel), round_index, panel_index, response_seed, int(main["global_seed"])
                )
                pools, full_scores = _dynamic_pools(data, U, V, panel, accepted_sets, warm_orders)
            if full_scores is None:
                score_matrix = np.einsum("ud,umd->um", U, V[pools], optimize=True).astype(np.float32)
            else:
                score_matrix = np.take_along_axis(full_scores, pools, axis=1).astype(np.float32)
            means, sds = candidate_anchor(score_matrix, float(main["minimum_score_sd"]))
            anchored_matrix = apply_anchor(score_matrix, means, sds)
            accepted_events = []
            turnover = 0.0 if previous_pools is None else float(np.mean([
                1.0 - len(set(previous_pools[r]).intersection(pools[r])) / 240.0
                for r in range(len(panel))
            ]))
            for row in range(len(panel)):
                order = np.argsort(-score_matrix[row], kind="stable")[:20]
                displayed = pools[row, order]
                anchored = anchored_matrix[row, order]
                effect_row = scenario_effects[row] if scenario_effects.ndim == 2 else scenario_effects
                provenance = effect_row[labels[displayed]].astype(float)
                probabilities = softmax_with_outside(
                    anchored + positions + provenance, float(main["outside_utility"])
                )
                clicked = choice_from_uniform(probabilities, float(response_uniforms[round_index, row]))
                exposure[displayed] += discount
                impressions += 1
                if clicked >= 0:
                    positive = int(displayed[clicked])
                    accepted_sets[row].add(positive)
                    clicks += 1
                    utility_sum += float(anchored[clicked])
                    if branch == "closed":
                        negative_index = min(int(negative_uniforms[round_index, row] * 19), 18)
                        if negative_index >= clicked:
                            negative_index += 1
                        accepted_events.append((row, positive, int(displayed[negative_index])))
            if branch == "closed" and accepted_events:
                by_user = {x[0]: x for x in accepted_events}
                for row in user_orders[round_index]:
                    row = int(row)
                    if row in by_user:
                        _, positive, negative = by_user[row]
                        update_pair(
                            U, V, row, positive, negative,
                            float(main["online_learning_rate"]),
                            float(main["online_regularization"]), steps=3,
                        )
            rows.append({
                "branch": branch, "round": round_index + 1,
                "accepted_interactions": clicks,
                "candidate_turnover": turnover,
                **cumulative_metrics(exposure, clicks, impressions, utility_sum, labels),
            })
            previous_pools = pools.copy()
    return pd.DataFrame(rows)


def run_dynamic():
    load_runtime()
    main, _, _ = read_configs()
    rows = []
    started = time.time()
    for panel_index in (0, 1):
        _, _, data, U, V, panel, initial_pools = context(240, panel_index)
        seeds, labels_list = assignment_setup(data, 5)
        for assignment_seed, labels in zip(seeds, labels_list):
            for response_seed in range(5):
                response, negative, orders = streams(main, len(panel), response_seed)
                for scenario in SCENARIOS:
                    result = simulate_dynamic(
                        data, U, V, panel, initial_pools, labels,
                        effects(main, scenario, response_seed, len(panel)),
                        response, negative, orders, main, panel_index, response_seed,
                    )
                    result.insert(0, "panel", panel_index)
                    result.insert(1, "assignment", assignment_seed)
                    result.insert(2, "response_seed", response_seed)
                    result.insert(3, "scenario", scenario)
                    result.insert(4, "candidate_mode", "roundwise_refresh")
                    rows.append(result)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(RAW / "dynamic_candidates_round_level.csv.gz", index=False, compression="gzip")
    (LOGS / "dynamic_candidates_run.json").write_text(json.dumps({
        "status": "complete", "rows": len(raw), "seconds": time.time() - started,
        "panels": 2, "assignments": 5, "response_seeds": 5,
        "scenarios": SCENARIOS, "candidate_mode": "refresh_from_round_2",
    }, indent=2))
    return raw


def _summarize_mode(raw, condition_columns, output_name, seed):
    rows = []
    grouped = raw.groupby(condition_columns, sort=True) if condition_columns else [((), raw)]
    for condition_keys, condition_raw in grouped:
        condition_keys = condition_keys if isinstance(condition_keys, tuple) else (condition_keys,)
        condition_values = dict(zip(condition_columns, condition_keys))
        amplified = add_amplification(condition_raw.assign(intervention="none"))
        final = amplified[amplified["round"] == 6]
        for scenario, group in final.groupby("scenario", sort=True):
            row = {**condition_values, "scenario": scenario}
            value = "closed_minus_frozen" if scenario == "zero" else "AA"
            out = hierarchical_bootstrap_mean(group, value, 2000, seed + len(rows))
            row.update({"metric": "zero_branch_difference" if scenario == "zero" else "LMC", **out})
            subset = condition_raw[(condition_raw["round"] == 6) & (condition_raw["scenario"] == scenario)]
            row["actual_acceptance_rate"] = float(subset["ctr"].mean())
            row["mean_cumulative_acceptances"] = float(subset["accepted_interactions"].mean())
            if "candidate_turnover" in subset:
                row["mean_round6_candidate_turnover"] = float(subset["candidate_turnover"].mean())
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / output_name, index=False)
    return frame


def summarize_legacy_low_acceptance():
    low = pd.read_csv(RAW / "lower_acceptance_round_level.csv.gz")
    low_summary = _summarize_mode(
        low,
        ["target_initial_acceptance", "outside_utility"],
        "lower_acceptance_legacy_two_panel_endpoints.csv",
        20269001,
    )
    print(low_summary.to_string(index=False))


def summarize_dynamic():
    dynamic = pd.read_csv(RAW / "dynamic_candidates_round_level.csv.gz")
    dynamic_summary = _summarize_mode(dynamic, ["candidate_mode"], "dynamic_candidates_endpoints.csv", 20269101)

    # Fixed-pool paired comparison on the identical reduced grid.
    fixed_parts = []
    fixed = pd.concat([
        pd.read_csv(PACKAGE / "raw" / "sensitivity" / "candidate" / f"value_240p0_panel_{panel}.csv.gz")
        for panel in (0, 1)
    ], ignore_index=True)
    fixed["candidate_mode"] = "fixed"
    dynamic_copy = dynamic.copy()
    dynamic_copy["intervention"] = "none"
    combined = pd.concat([fixed, dynamic_copy], ignore_index=True, sort=False)
    amps = []
    for mode, group in combined.groupby("candidate_mode"):
        x = add_amplification(group)
        x["candidate_mode"] = mode
        amps.append(x)
    amp = pd.concat(amps, ignore_index=True)
    final = amp[amp["round"] == 6]
    rows = []
    for scenario in SCENARIOS[1:]:
        sub = final[final.scenario == scenario]
        wide = sub.pivot(index=["panel", "assignment", "response_seed", "scenario"], columns="candidate_mode", values="AA").reset_index()
        wide["dynamic_minus_fixed_LMC"] = wide["roundwise_refresh"] - wide["fixed"]
        out = hierarchical_bootstrap_mean(wide, "dynamic_minus_fixed_LMC", 2000, 20269201 + len(rows))
        rows.append({"scenario": scenario, **out})
    pd.DataFrame(rows).to_csv(TABLES / "dynamic_minus_fixed_paired_LMC.csv", index=False)
    print(dynamic_summary.to_string(index=False))


def summarize_expanded_low_acceptance():
    raw = pd.read_csv(RAW / "lower_acceptance_expanded_round_level.csv.gz")
    summary = _summarize_mode(
        raw,
        ["feedback_condition", "outside_utility"],
        "lower_acceptance_expanded_endpoints.csv",
        EXPANDED_ENDPOINT_BOOTSTRAP_SEED,
    )
    target_map = {
        "target_10": 0.10,
        "target_20": 0.20,
        "target_40": 0.40,
        "high_feedback_reference": np.nan,
    }
    summary.insert(
        1,
        "target_initial_acceptance",
        summary["feedback_condition"].map(target_map),
    )
    summary.to_csv(TABLES / "lower_acceptance_expanded_endpoints.csv", index=False)

    amplified_parts = []
    for condition, group in raw.groupby("feedback_condition", sort=False):
        x = add_amplification(group.assign(intervention="none"))
        x["feedback_condition"] = condition
        amplified_parts.append(x)
    amplified = pd.concat(amplified_parts, ignore_index=True)
    final = amplified[(amplified["round"] == 6) & (amplified["scenario"] != "zero")]
    paired_rows = []
    for scenario in SCENARIOS[1:]:
        sub = final[final["scenario"] == scenario]
        wide = sub.pivot(
            index=["panel", "assignment", "response_seed", "scenario"],
            columns="feedback_condition",
            values="AA",
        ).reset_index()
        for target in TARGETS:
            label = f"target_{int(100 * target)}"
            raw_name = f"LMC_{label}_minus_high"
            mag_name = f"abs_LMC_{label}_minus_high"
            wide[raw_name] = wide[label] - wide["high_feedback_reference"]
            wide[mag_name] = wide[label].abs() - wide["high_feedback_reference"].abs()
            raw_out = hierarchical_bootstrap_mean(
                wide, raw_name, BOOTSTRAP_REPETITIONS,
                EXPANDED_PAIRED_LMC_BOOTSTRAP_SEED + len(paired_rows) * 2,
            )
            mag_out = hierarchical_bootstrap_mean(
                wide, mag_name, BOOTSTRAP_REPETITIONS,
                EXPANDED_PAIRED_ABSOLUTE_LMC_BOOTSTRAP_SEED + len(paired_rows) * 2,
            )
            paired_rows.append({
                "target_initial_acceptance": target,
                "scenario": scenario,
                "lmc_difference_estimate": raw_out["estimate"],
                "lmc_difference_ci_low": raw_out["ci_low"],
                "lmc_difference_ci_high": raw_out["ci_high"],
                "absolute_lmc_difference_estimate": mag_out["estimate"],
                "absolute_lmc_difference_ci_low": mag_out["ci_low"],
                "absolute_lmc_difference_ci_high": mag_out["ci_high"],
            })
    pd.DataFrame(paired_rows).to_csv(
        TABLES / "lower_acceptance_expanded_paired_vs_high.csv", index=False
    )
    compatibility = summary[
        summary["feedback_condition"].isin({"target_10", "target_20", "target_40"})
    ].drop(columns=["feedback_condition"])
    compatibility.to_csv(TABLES / "lower_acceptance_endpoints.csv", index=False)
    print(summary.to_string(index=False))


def summarize():
    """Recreate the current five-panel lower-feedback and refresh summaries."""
    summarize_expanded_low_acceptance()
    summarize_dynamic()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "low", "dynamic", "summarize", "all", "low-expanded",
            "summarize-expanded", "summarize-legacy",
        ],
    )
    args = parser.parse_args()
    if args.command == "all":
        run_expanded_low_acceptance()
        run_dynamic()
        summarize()
    elif args.command == "low":
        run_low_acceptance()
    elif args.command == "dynamic":
        run_dynamic()
    elif args.command == "summarize":
        summarize()
    elif args.command == "low-expanded":
        run_expanded_low_acceptance()
    elif args.command == "summarize-expanded":
        summarize_expanded_low_acceptance()
    elif args.command == "summarize-legacy":
        summarize_legacy_low_acceptance()


if __name__ == "__main__":
    main()
