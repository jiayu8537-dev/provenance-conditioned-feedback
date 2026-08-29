#!/usr/bin/env python3
"""Run archive-level or complete JIIS reproduction workflows."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str, root: Path = ROOT) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(command, cwd=root, check=True, env=environment)


def aggregate(root: Path = ROOT) -> None:
    run("run_all_cpu.py", "aggregate", root=root)
    run("run_extension.py", "aggregate-lightgcn", root=root)
    run("run_extension.py", "aggregate-long-horizon", root=root)
    run("run_robustness.py", "aggregate", root=root)
    run("run_choice_process_sensitivity.py", "aggregate", root=root)
    if (root / "publication_assets" / "make_jiis_final_figures.py").is_file():
        run("publication_assets/make_jiis_final_figures.py", root=root)


def audit() -> None:
    """Validate the immutable archive, then recompute summaries in a temporary copy."""
    run("scripts/verify_package.py")
    run(
        "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--import-mode=importlib", "tests",
    )
    reference = {
        str(path.relative_to(ROOT)): pd.read_csv(path)
        for path in sorted((ROOT / "tables").rglob("*.csv"))
    }
    with tempfile.TemporaryDirectory(prefix="ipm_reproduction_audit_") as directory:
        work = Path(directory) / "package"
        shutil.copytree(
            ROOT,
            work,
            ignore=shutil.ignore_patterns(
                "MANIFEST_SHA256.txt", "__pycache__", ".pytest_cache", "*.pyc"
            ),
        )
        aggregate(work)
        max_numeric_difference = 0.0
        point_columns_checked = 0
        for relative, expected in reference.items():
            observed = pd.read_csv(work / relative)
            if list(expected.columns) != list(observed.columns) or expected.shape != observed.shape:
                raise RuntimeError(f"Recomputed table structure differs: {relative}")
            for column in expected.columns:
                left, right = expected[column], observed[column]
                if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                    difference = np.abs(left.to_numpy(float) - right.to_numpy(float))
                    finite = difference[np.isfinite(difference)]
                    if len(finite):
                        max_numeric_difference = max(max_numeric_difference, float(finite.max()))
                    is_interval = "ci_" in column.lower() or column.lower().startswith("ci")
                    if not is_interval:
                        if not np.allclose(
                            left.to_numpy(float), right.to_numpy(float),
                            rtol=0, atol=1e-12, equal_nan=True,
                        ):
                            raise RuntimeError(
                                f"Recomputed point/statistic differs: {relative}:{column}"
                            )
                        point_columns_checked += 1
                elif not left.fillna("<NA>").astype(str).equals(
                    right.fillna("<NA>").astype(str)
                ):
                    raise RuntimeError(f"Recomputed categorical field differs: {relative}:{column}")
        if max_numeric_difference > 0.002:
            raise RuntimeError(
                "Recomputed percentile interval differs from the archived value by "
                f"{max_numeric_difference:.6g}, exceeding the 0.002 audit tolerance"
            )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "tables_recomputed": len(reference),
                    "point_columns_checked": point_columns_checked,
                    "max_interval_endpoint_difference": max_numeric_difference,
                    "audit_copy_removed": True,
                },
                indent=2,
            ),
            flush=True,
        )


def full() -> None:
    data_root = Path(os.environ.get("JIIS_DATA_ROOT", ROOT / "data" / "derived"))
    required = [
        "main_10u5i_leave_last_two_split.csv.gz",
        "model_aware_provenance_assignment_v2.csv",
        "strict_15u5i_user_index.csv",
        "strict_15u5i_item_index.csv",
        "strict_15u5i_interaction_row_ids.csv.gz",
    ]
    absent = [name for name in required if not (data_root / name).is_file()]
    if absent:
        raise SystemExit(
            "Reconstruct the licensed inputs first; missing: " + ", ".join(absent)
        )

    run("run_all_cpu.py", "train")
    for panel in range(5):
        run("run_all_cpu.py", "main", "--panel", str(panel))
    for kind, values in {
        "candidate": (240, 500, 1000),
        "updates": (1, 3, 10),
        "oracle": (0.5, 1.0, 1.5),
    }.items():
        for value in values:
            for panel in (0, 1):
                run(
                    "run_all_cpu.py", "sensitivity", "--kind", kind,
                    "--value", str(value), "--panel", str(panel),
                )
    run("run_all_cpu.py", "aggregate")

    run("run_extension.py", "train-lightgcn")
    for panel in (0, 1):
        run("run_extension.py", "lightgcn", "--panel", str(panel))
        for rounds in (6, 12, 24):
            run(
                "run_extension.py", "long-horizon", "--panel", str(panel),
                "--rounds", str(rounds),
            )
    run("run_extension.py", "aggregate-lightgcn")
    run("run_extension.py", "aggregate-long-horizon")

    run("run_robustness.py", "prepare-strict")
    for panel in (0, 1):
        run("run_robustness.py", "synchronous", "--panel", str(panel))
        run("run_robustness.py", "strict", "--panel", str(panel))
    run("run_robustness.py", "aggregate")

    run("run_choice_process_sensitivity.py", "all")
    if (ROOT / "publication_assets" / "make_jiis_final_figures.py").is_file():
        run("publication_assets/make_jiis_final_figures.py")
    run(
        "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--import-mode=importlib", "tests",
    )
    run("scripts/verify_package.py", "--skip-manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "aggregate", "full"))
    args = parser.parse_args()
    if args.mode == "audit":
        audit()
    elif args.mode == "aggregate":
        aggregate()
    else:
        full()


if __name__ == "__main__":
    main()
