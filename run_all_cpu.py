#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import add_amplification, hierarchical_bootstrap_mean, paired_difference
from src.bpr_training import load_or_train
from src.data_loading import (
    generate_assignment,
    load_data,
    make_candidate_pool,
    qualified_assignment_seeds,
    stratified_panel,
)
from src.simulation import SimulationConfig, simulate_paired

DATA_ROOT = Path(os.environ.get("JIIS_DATA_ROOT", ROOT / "data" / "derived"))
SPLIT = DATA_ROOT / "main_10u5i_leave_last_two_split.csv.gz"
ASSIGNMENT = DATA_ROOT / "model_aware_provenance_assignment_v2.csv"
ASSIGNMENT_CANDIDATES = (
    ROOT / "data" / "protocol" / "model_aware_assignment_candidates.csv"
)

CONFIGS = ROOT / "configs"
RAW = ROOT / "raw"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
ARTIFACTS = ROOT / "artifacts"
for directory in (RAW, TABLES, LOGS, ARTIFACTS):
    directory.mkdir(parents=True, exist_ok=True)


def read_configs():
    main = yaml.safe_load((CONFIGS / "main.yaml").read_text())
    bpr = yaml.safe_load((CONFIGS / "bpr.yaml").read_text())
    sensitivity = yaml.safe_load((CONFIGS / "sensitivity.yaml").read_text())
    return main, bpr, sensitivity


def context(candidate_size: int, panel_index: int):
    main, bpr, sensitivity = read_configs()
    data = load_data(SPLIT, ASSIGNMENT)
    U, V = load_or_train(
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
        U,
        V,
        panel,
        candidate_size,
        int(main["top_retrieval"]),
        int(main["global_seed"]) + 22000 + panel_index * 1000 + candidate_size,
    )
    return main, sensitivity, data, U, V, panel, pools


def streams(main: dict, panel_size: int, response_seed: int):
    rng = np.random.default_rng(int(main["global_seed"]) + 30000 + response_seed)
    uniforms = rng.random((int(main["rounds"]), panel_size), dtype=np.float32)
    negative = rng.random((int(main["rounds"]), panel_size), dtype=np.float32)
    # Original executable behavior: deterministic panel order; no invented random order.
    orders = np.tile(np.arange(panel_size, dtype=np.int32), (int(main["rounds"]), 1))
    return uniforms, negative, orders


def effects(main: dict, scenario: str, response_seed: int, panel_size: int):
    if scenario != "heterogeneous_response":
        return np.asarray(main["scenarios"][scenario], np.float32)
    rng = np.random.default_rng(int(main["global_seed"]) + 24000 + response_seed)
    kinds = rng.choice(3, panel_size, p=[0.30, 0.50, 0.20])
    patterns = np.array(
        [[-0.15, -0.05, 0.05], [0.0, 0.0, 0.0], [0.10, 0.05, 0.0]],
        np.float32,
    )
    return patterns[kinds]


def sim_config(main: dict, steps: int):
    return SimulationConfig(
        rounds=int(main["rounds"]),
        top_k=int(main["top_k"]),
        online_steps=steps,
        online_learning_rate=float(main["online_learning_rate"]),
        online_regularization=float(main["online_regularization"]),
        outside_utility=float(main["outside_utility"]),
        position_scale=float(main["position_scale"]),
        minimum_score_sd=float(main["minimum_score_sd"]),
    )


def assignment_setup(data, count: int):
    seeds, metadata = qualified_assignment_seeds(ASSIGNMENT_CANDIDATES, count)
    metadata.to_csv(RAW / f"qualified_assignment_metadata_{count}.csv", index=False)
    return seeds, [generate_assignment(data.n_items, data.blocks, seed) for seed in seeds]


def run_grid(
    panel_index: int,
    candidate_size: int,
    steps: int,
    assignment_count: int,
    response_count: int,
    scenarios: list[str],
    interventions: dict[str, list[str]],
    oracle_c: float,
    output: Path,
):
    started = time.time()
    main, _, data, U, V, panel, pools = context(candidate_size, panel_index)
    assignment_seeds, labels_list = assignment_setup(data, assignment_count)
    rows = []
    for assignment_seed, labels in zip(assignment_seeds, labels_list):
        for response_seed in range(response_count):
            uniforms, negative, orders = streams(main, len(panel), response_seed)
            for scenario in scenarios:
                delta = effects(main, scenario, response_seed, len(panel))
                for intervention in interventions[scenario]:
                    result = simulate_paired(
                        U, V, panel, pools, labels, delta,
                        uniforms, negative, orders,
                        sim_config(main, steps), intervention, oracle_c,
                    )
                    result.insert(0, "panel", panel_index)
                    result.insert(1, "assignment", assignment_seed)
                    result.insert(2, "response_seed", response_seed)
                    result.insert(3, "scenario", scenario)
                    result.insert(4, "intervention", intervention)
                    result["candidate_size"] = candidate_size
                    result["online_steps"] = steps
                    result["oracle_c"] = oracle_c
                    rows.append(result)
    combined = pd.concat(rows, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, compression="gzip")
    (LOGS / f"{output.stem}.json").write_text(json.dumps({
        "status": "complete", "rows": len(combined), "seconds": time.time()-started,
        "panel": panel_index, "candidate_size": candidate_size, "online_steps": steps,
        "assignment_count": assignment_count, "response_count": response_count,
        "oracle_c": oracle_c,
    }, indent=2))


def run_main(panel_index: int):
    main, _, _ = read_configs()
    all_scenarios = list(main["scenarios"]) + ["heterogeneous_response"]
    interventions = {}
    for scenario in all_scenarios:
        interventions[scenario] = ["none"]
        if scenario in {"zero", "moderate_asymmetric", "strong_premium_penalty"}:
            interventions[scenario] += ["oracle_feedback_correction", "quota_reranking", "combined"]
    run_grid(
        panel_index, 240, 3, 10, 10, all_scenarios, interventions, 1.0,
        RAW / "main" / f"panel_{panel_index}.csv.gz",
    )


def run_sensitivity(kind: str, value: float, panel_index: int):
    main, _, sensitivity = read_configs()
    scenarios = list(sensitivity["scenarios"])
    candidate_size, steps, oracle_c = 240, 3, 1.0
    interventions = {scenario: ["none"] for scenario in scenarios}
    if kind == "candidate":
        candidate_size = int(value)
    elif kind == "updates":
        steps = int(value)
    elif kind == "oracle":
        oracle_c = float(value)
        interventions = {scenario: ["oracle_feedback_correction"] for scenario in scenarios}
    else:
        raise ValueError(kind)
    safe = str(value).replace(".", "p")
    run_grid(
        panel_index, candidate_size, steps,
        int(sensitivity["assignment_count"]), int(sensitivity["response_seed_count"]),
        scenarios, interventions, oracle_c,
        RAW / "sensitivity" / kind / f"value_{safe}_panel_{panel_index}.csv.gz",
    )


def summarize_bootstrap(group: pd.DataFrame, value: str, repetitions: int, seed: int):
    result = hierarchical_bootstrap_mean(group, value, repetitions, seed)
    result["n"] = len(group)
    return result


def aggregate():
    main, _, sensitivity = read_configs()
    files = sorted((RAW / "main").glob("panel_*.csv.gz"))
    if len(files) != int(main["panel_count"]):
        raise RuntimeError(f"Expected {main['panel_count']} main shards; found {len(files)}")
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    raw.to_csv(RAW / "main_corrected_round_level.csv.gz", index=False, compression="gzip")
    amp = add_amplification(raw)
    amp.to_csv(RAW / "main_amplification_round_level.csv.gz", index=False, compression="gzip")
    final = amp[amp["round"] == int(main["rounds"])].copy()
    zero = final[(final.scenario == "zero") & (final.intervention == "none")]
    b = summarize_bootstrap(zero, "closed_minus_frozen", int(main["bootstrap_repetitions"]), 20261001)
    n = len(zero); mean = zero.closed_minus_frozen.mean(); se = zero.closed_minus_frozen.std(ddof=1)/np.sqrt(n)
    critical = stats.t.ppf(0.95, n-1); margin = float(main["equivalence_margin"])
    p_lower = 1-stats.t.cdf((mean+margin)/se, n-1)
    p_upper = stats.t.cdf((mean-margin)/se, n-1)
    table3 = pd.DataFrame([{**b, "ci90_low": mean-critical*se, "ci90_high": mean+critical*se,
        "equivalence_margin": margin, "TOST_p": max(p_lower,p_upper),
        "equivalent": bool((mean-critical*se > -margin) and (mean+critical*se < margin))}])
    table3.to_csv(TABLES / "Table3_zero_effect_equivalence.csv", index=False)

    table4_rows=[]
    for scenario, group in final[(final.scenario != "zero") & (final.intervention == "none")].groupby("scenario"):
        table4_rows.append({"scenario": scenario, **summarize_bootstrap(group, "AA", int(main["bootstrap_repetitions"]), 20262000+len(table4_rows))})
    pd.DataFrame(table4_rows).to_csv(TABLES / "Table4_BPR_algorithmic_amplification.csv", index=False)

    metrics = ["abs_D", "exposure_tv", "ctr", "candidate_anchored_utility", "coverage", "gini"]
    closed = raw[(raw.branch == "closed") & (raw["round"] == int(main["rounds"]))]
    table5_rows=[]
    for (scenario, intervention), group in closed[closed.scenario.isin(["moderate_asymmetric","strong_premium_penalty"])].groupby(["scenario","intervention"]):
        row={"scenario":scenario,"intervention":intervention}
        amp_group=final[(final.scenario==scenario)&(final.intervention==intervention)]
        if not amp_group.empty:
            aa=summarize_bootstrap(amp_group,"AA",int(main["bootstrap_repetitions"]),20263000+len(table5_rows)); row.update({f"AA_{k}":v for k,v in aa.items()})
        if intervention != "none":
            for metric in metrics:
                paired=paired_difference(closed[closed.scenario==scenario],metric,intervention)
                out=summarize_bootstrap(paired,f"delta_{metric}",int(main["bootstrap_repetitions"]),20264000+len(table5_rows))
                row.update({f"delta_{metric}_{k}":v for k,v in out.items() if k!="n"})
        for metric in metrics: row[f"mean_{metric}"]=group[metric].mean()
        table5_rows.append(row)
    pd.DataFrame(table5_rows).to_csv(TABLES / "Table5_intervention_tradeoffs.csv", index=False)

    for kind, filename in [("candidate","candidate_pool_sensitivity.csv"),("updates","online_update_sensitivity.csv"),("oracle","oracle_misspecification_sensitivity.csv")]:
        paths=sorted((RAW/"sensitivity"/kind).glob("*.csv.gz"))
        if not paths: continue
        parameter={"candidate":"candidate_size","updates":"online_steps","oracle":"oracle_c"}[kind]
        rows=[]
        data=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True)
        for value, value_frame in data.groupby(parameter):
            amplified = add_amplification(value_frame)
            final_sensitivity = amplified[
                amplified["round"] == int(main["rounds"])
            ]
            for scenario, group in final_sensitivity.groupby("scenario"):
                rows.append({parameter:value,"scenario":scenario,**summarize_bootstrap(group,"AA",int(sensitivity["bootstrap_repetitions"]),20265000+len(rows))})
        pd.DataFrame(rows).to_csv(TABLES/filename,index=False)

    (LOGS / "aggregate_manifest.json").write_text(json.dumps({
        "status":"complete","main_rows":len(raw),
        "main_shards":[str(p.relative_to(ROOT)) for p in files],
        "supersedes":"all previous dynamic BPR point estimates",
    },indent=2))


def run_everything():
    subprocess.run([sys.executable, __file__, "train"], check=True)
    for panel in range(5): subprocess.run([sys.executable,__file__,"main","--panel",str(panel)],check=True)
    for kind, values in {"candidate":[240,500,1000],"updates":[1,3,10],"oracle":[0.5,1.0,1.5]}.items():
        for value in values:
            for panel in (0,1):
                subprocess.run([sys.executable,__file__,"sensitivity","--kind",kind,"--value",str(value),"--panel",str(panel)],check=True)
    aggregate()


def main_cli():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("train")
    p=sub.add_parser("main"); p.add_argument("--panel",type=int,required=True,choices=range(5))
    p=sub.add_parser("sensitivity"); p.add_argument("--kind",choices=["candidate","updates","oracle"],required=True); p.add_argument("--value",type=float,required=True); p.add_argument("--panel",type=int,required=True,choices=[0,1])
    sub.add_parser("aggregate"); sub.add_parser("all")
    args=parser.parse_args()
    if args.command=="train": context(240,0)
    elif args.command=="main": run_main(args.panel)
    elif args.command=="sensitivity": run_sensitivity(args.kind,args.value,args.panel)
    elif args.command=="aggregate": aggregate()
    else: run_everything()


if __name__ == "__main__": main_cli()
