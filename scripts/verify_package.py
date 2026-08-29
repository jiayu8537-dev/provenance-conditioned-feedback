#!/usr/bin/env python3
"""Audit completeness, disclosure safety, hashes, and reported result anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []
CHECKS: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        CHECKS.append(label)
    else:
        FAILURES.append(label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    check(math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance), label)


def required_files() -> None:
    paths = [
        "README.md", "RUN_ORDER.md", "DATA_ACCESS.md", "DATA_DICTIONARY.md",
        "ENVIRONMENT.md", "MANUSCRIPT_CROSSWALK.md", "SEED_MANIFEST.csv",
        "LICENSE.md", "CITATION.cff",
        "requirements.txt", "configs/main.yaml", "configs/bpr.yaml",
        "configs/extension.yaml", "configs/robustness.yaml",
        "scripts/prepare_pixelrec_inputs.py", "scripts/run_pipeline.py",
        "data/protocol/model_aware_assignment_protocol.npz",
        "artifacts/bpr_full_history.npz", "artifacts/lightgcn_full_history_cpu.npz",
        "raw/main_corrected_round_level.csv.gz",
        "raw/extension/lightgcn_round_level.csv.gz",
        "raw/extension/long_horizon_round_level.csv.gz",
        "raw/robustness/synchronous_round_level.csv.gz",
        "raw/robustness/strict_round_level.csv.gz",
        "tables/Table3_zero_effect_equivalence.csv",
        "tables/Table4_BPR_algorithmic_amplification.csv",
        "tables/Table5_intervention_tradeoffs.csv",
        "tables/extension/lightgcn_dynamic_amplification.csv",
        "tables/extension/long_horizon_zero_drift.csv",
        "tables/robustness/robustness_endpoints.csv",
        "tables/choice_process/choice_process_sensitivity.csv",
        "publication_assets/figures/main/Fig1.pdf",
        "publication_assets/figures/main/Fig2.pdf",
        "publication_assets/figures/main/Fig3.pdf",
        "publication_assets/figures/main/Fig4.pdf",
        "publication_assets/figures/supplementary/SFig1_sensitivity.png",
        "publication_assets/figures/supplementary/SFig2_targeted_validation.png",
    ]
    for relative in paths:
        check((ROOT / relative).is_file(), f"required file: {relative}")


def disclosure_audit() -> None:
    prohibited_names = {".DS_Store", "__pycache__", ".pytest_cache"}
    for path in ROOT.rglob("*"):
        check(path.name not in prohibited_names, f"no hidden/cache item: {path.relative_to(ROOT)}")
        if path.is_file():
            check(path.suffix != ".pyc", f"no bytecode: {path.relative_to(ROOT)}")
            check(path.suffix.lower() != ".zip", f"no nested ZIP: {path.relative_to(ROOT)}")

    text_suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".csv", ".sh"}
    needles = (
        "/" + "Users" + "/",
        "/" + "mnt" + "/" + "data",
        "tig" + "her",
        "\u7b2c\u4e8c" + "SCI",
        "\u672a\u547d\u540d\u6587\u4ef6\u5939",
    )
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_suffixes:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for needle in needles:
                check(needle not in content, f"no private/legacy token {needle!r}: {path.relative_to(ROOT)}")

    derived = [p for p in (ROOT / "data" / "derived").rglob("*") if p.is_file()]
    check([p.name for p in derived] == ["README.md"], "licensed derived-data directory is empty")


def main_grid_checks() -> None:
    files = sorted((ROOT / "raw" / "main").glob("panel_*.csv.gz"))
    check(len(files) == 5, "five confirmatory panel shards")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    check(len(frame) == 90000, "confirmatory raw row count is 90,000")
    check(set(frame.panel) == set(range(5)), "confirmatory panels 0-4")
    check(frame.assignment.nunique() == 10, "ten confirmatory assignments")
    check(frame.response_seed.nunique() == 10, "ten confirmatory response streams")
    check(set(frame["round"]) == set(range(1, 7)), "six confirmatory rounds")
    check(set(frame.branch) == {"frozen", "closed"}, "matched branches present")
    numeric = frame.select_dtypes(include=[np.number])
    check(np.isfinite(numeric.to_numpy()).all(), "all confirmatory numeric values finite")
    frozen = frame[frame.branch.eq("frozen")]
    closed = frame[frame.branch.eq("closed")]
    check((frozen.online_sgd_updates == 0).all(), "frozen branch has no SGD updates")
    check(
        (closed.online_sgd_updates == 3 * closed.accepted_interactions).all(),
        "closed branch update accounting equals three steps per accepted event",
    )


def extension_checks() -> None:
    light = pd.read_csv(ROOT / "raw/extension/lightgcn_round_level.csv.gz")
    check(len(light) == 4200, "LightGCN raw row count")
    check(light.panel.nunique() == 2 and light.assignment.nunique() == 5 and light.response_seed.nunique() == 5,
          "LightGCN reduced crossed grid")
    horizons = pd.read_csv(ROOT / "raw/extension/long_horizon_round_level.csv.gz")
    check(set(horizons.horizon) == {6, 12, 24}, "6/12/24 horizons present")
    check(set(horizons.update_regime) == {"event_online", "fixed_budget_history_replay"},
          "event-online and replay regimes present")
    robust = pd.read_csv(ROOT / "tables/robustness/robustness_endpoints.csv")
    check(set(robust.core) == {"confirmatory_10u5i", "strict_15u5i"}, "both structural cores present")
    choice = pd.read_csv(ROOT / "tables/choice_process/choice_process_sensitivity.csv")
    check(len(choice) == 15 and choice.specification.nunique() == 5, "five choice specifications by three scenarios")


def result_anchor_checks() -> None:
    table3 = pd.read_csv(ROOT / "tables/Table3_zero_effect_equivalence.csv").iloc[0]
    close(table3.estimate, -0.0007634518798332572, "Section 5.1 neutral-drift estimate")
    check(bool(table3.equivalent), "Section 5.1 equivalence conclusion")

    table4 = pd.read_csv(ROOT / "tables/Table4_BPR_algorithmic_amplification.csv").set_index("scenario")
    anchors = {
        "ai_appreciation": -0.006883054134401714,
        "moderate_asymmetric": 0.014933671409202657,
        "strong_premium_penalty": 0.02886565187408185,
    }
    for scenario, expected in anchors.items():
        close(table4.loc[scenario, "estimate"], expected, f"Fig. 3 endpoint {scenario}")

    table5 = pd.read_csv(ROOT / "tables/Table5_intervention_tradeoffs.csv")
    key = table5.set_index(["scenario", "intervention"])
    close(key.loc[("moderate_asymmetric", "quota_reranking"), "AA_estimate"],
          0.0066856971335070995, "Table 4 quota estimate")
    close(key.loc[("strong_premium_penalty", "combined"), "AA_estimate"],
          -0.00014651001641893772, "Table 4 combined estimate")

    light = pd.read_csv(ROOT / "tables/extension/lightgcn_dynamic_amplification.csv")
    light = light.set_index(["scenario", "intervention"])
    close(light.loc[("moderate_asymmetric", "none"), "estimate"],
          0.06249784510824654, "LightGCN moderate estimate")
    close(light.loc[("strong_premium_penalty", "none"), "estimate"],
          0.1124331595883206, "LightGCN strong estimate")

    zero = pd.read_csv(ROOT / "tables/extension/long_horizon_zero_drift.csv")
    equivalent = zero.set_index(["horizon", "update_regime"])["equivalent_by_95_percent_interval"].to_dict()
    check(all(equivalent[(h, r)] for h in (6, 12) for r in ("event_online", "fixed_budget_history_replay")),
          "zero-response equivalence through round 12")
    check(not any(equivalent[(24, r)] for r in ("event_online", "fixed_budget_history_replay")),
          "round-24 zero-response intervals exceed the equivalence margin")


def manifest_checks() -> None:
    manifest = ROOT / "MANIFEST_SHA256.txt"
    if not manifest.is_file():
        CHECKS.append("manifest check deferred until final assembly")
        return
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
    actual_files = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path != manifest
    }
    check(actual_files == set(expected), "manifest file set")
    for relative, digest in expected.items():
        check((ROOT / relative).is_file() and sha256(ROOT / relative) == digest,
              f"manifest hash: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()
    required_files()
    disclosure_audit()
    main_grid_checks()
    extension_checks()
    result_anchor_checks()
    if args.skip_manifest:
        CHECKS.append("manifest check skipped for a recomputed working copy")
    else:
        manifest_checks()
    report = {
        "status": "passed" if not FAILURES else "failed",
        "checks_passed": len(CHECKS),
        "failures": FAILURES,
    }
    if args.write_report:
        path = ROOT / "logs" / "package_validation.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
