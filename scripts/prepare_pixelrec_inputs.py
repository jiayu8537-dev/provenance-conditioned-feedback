#!/usr/bin/env python3
"""Reconstruct the licensed PixelRec-derived inputs used by the JIIS study.

The official PixelRec files are intentionally not redistributed.  This script
recreates the k-cores, leave-last-two split, indices, and model-aware block file
from the official PixelRec50K interaction and item-information CSV files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PACKAGE_ROOT / "data" / "protocol" / "model_aware_assignment_protocol.npz"

OFFICIAL_INPUT_SHA256 = {
    "interaction.csv": "638b53ec100f760cb9bd540c361f6d6e3617c81b1c054ced63fffa41da909e4d",
    "item_info.csv": "a073c2c65900f215a8137929b27dc57cf6f4f8fa11453a5c74fa8ff3a730a04e",
}

EXPECTED = {
    "raw_rows": 989_494,
    "raw_users": 50_000,
    "raw_items": 82_865,
    "wide": (956_817, 49_993, 59_781),
    "main": (816_905, 38_921, 44_923),
    "strict": (567_403, 20_132, 37_576),
    "main_split_uncompressed_sha256": "4445f6421e99bdc1c6d0dba029bca5de73cc9d6ace11818d394e376e0ed1fbf5",
    "main_user_index_sha256": "ccb358622fc9dbe3b8afbb4457bac23fb9589b75dfffcc58214f57309c99a00e",
    "main_item_index_sha256": "8e0dd7be7f1f1270f23f04f2b5ef51a94ab68e217adba85b7be219860ea542d0",
    "main_row_index_uncompressed_sha256": "4f3d5ddcf6935e03de1f75771fa1dcba9093844078de8b67d9093dbf48efda9a",
    "strict_user_index_sha256": "2ae230534fa35cd88b3bca92688479e3d3b251cf197c6e383efbc8821c44fa32",
    "strict_item_index_sha256": "ac56a041a7e2092636a0c776db77a72e7899acfdd2d0816160f0271589910310",
    "strict_row_index_uncompressed_sha256": "2f8d825a484ecee026321681de8d611d9068a1ec3ef74a86f7ba5f2d33fb915e",
    "catalog_item_id_vector_sha256": "6468dd1e8e6d88956d6044650927437c481827d97f928ce2a89c9941b64bab5e",
    "model_aware_block_vector_sha256": "5789c72f5b535d6d6324a498bee266082550132f694bc8716d77b79b4551c4fd",
    "selected_label_vector_sha256": "a7a1efe2e783a46d4404572940ee38717da202e0368f8358f801f71a29831d14",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_uncompressed(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_sha256(values: pd.Series | np.ndarray) -> str:
    payload = ("\n".join(pd.Series(values).astype(str).tolist()) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iterative_kcore(frame: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    current = frame
    previous = -1
    while len(current) != previous:
        previous = len(current)
        user_counts = current["user_id"].value_counts()
        item_counts = current["item_id"].value_counts()
        current = current[
            current["user_id"].isin(user_counts[user_counts >= min_user].index)
            & current["item_id"].isin(item_counts[item_counts >= min_item].index)
        ]
    return current.copy()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        frame.to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
    else:
        frame.to_csv(path, index=False)


def qbin(series: pd.Series, bins: int, prefix: str) -> pd.Series:
    return prefix + pd.qcut(
        series.rank(method="first"), bins, labels=False
    ).astype(int).astype(str)


def build_assignment_metadata(
    interactions: pd.DataFrame,
    main_row_ids: np.ndarray,
    item_info: pd.DataFrame,
) -> pd.DataFrame:
    observed = interactions.iloc[main_row_ids].copy()
    observed["timestamp_dt"] = pd.to_datetime(observed["timestamp"], unit="s", utc=True)
    stats = (
        observed.groupby("item_id")
        .agg(
            sampled_interactions=("user_id", "size"),
            first_observed_timestamp=("timestamp_dt", "min"),
            last_observed_timestamp=("timestamp_dt", "max"),
        )
        .reset_index()
    )
    frame = item_info.merge(stats, on="item_id")
    frame["observed_item_age_days"] = (
        observed["timestamp_dt"].max() - frame["first_observed_timestamp"]
    ).dt.total_seconds() / 86_400

    tags = frame["tag"].fillna("Missing")
    major = set(tags.value_counts()[lambda values: values >= 100].index)
    frame["theme_group"] = tags.where(tags.isin(major), "Other_or_rare")
    frame["sample_popularity_bin"] = qbin(
        frame["sampled_interactions"], 5, "p"
    )
    frame["observed_age_bin"] = qbin(frame["observed_item_age_days"], 4, "a")
    frame["allocation_block"] = (
        frame["theme_group"] + "|" + frame["sample_popularity_bin"]
    )

    metrics = [
        "view_number",
        "comment_number",
        "thumbup_number",
        "share_number",
        "coin_number",
        "favorite_number",
        "barrage_number",
    ]
    ranks = []
    for column in metrics:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric.fillna(numeric.median())
        ranks.append(np.log1p(frame[column]).rank(pct=True))
    frame["platform_engagement_score"] = pd.concat(ranks, axis=1).mean(axis=1)

    frame = frame.sort_values("item_id", kind="stable").reset_index(drop=True)
    with np.load(PROTOCOL, allow_pickle=False) as protocol:
        labels = protocol["labels"].astype(np.uint8)
        legacy_labels = protocol["legacy_labels"].astype(np.uint8)
        clusters = protocol["latent_cluster"].astype(np.int16)
        blocks = protocol["blocks"].astype(str)
    if not (len(frame) == len(labels) == len(clusters) == len(blocks)):
        raise RuntimeError("Protocol arrays do not align with the reconstructed catalog")

    group_names = np.array(
        ["AI-attributed", "Human-AI-attributed", "Human-attributed"]
    )
    legacy_names = np.array(
        ["Human-attributed", "Human-AI-attributed", "AI-attributed"]
    )
    frame["simulated_provenance"] = legacy_names[legacy_labels]
    frame["latent_cluster"] = clusters
    frame["model_aware_block"] = blocks
    frame["simulated_provenance_v2"] = group_names[labels]
    frame["provenance_code_v2"] = labels

    columns = [
        "item_id",
        "simulated_provenance",
        "allocation_block",
        "theme_group",
        "sample_popularity_bin",
        "observed_age_bin",
        "sampled_interactions",
        "first_observed_timestamp",
        "last_observed_timestamp",
        "observed_item_age_days",
        *metrics,
        "platform_engagement_score",
        "latent_cluster",
        "model_aware_block",
        "simulated_provenance_v2",
        "provenance_code_v2",
    ]
    return frame[columns]


def assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, obtained {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction", type=Path, required=True)
    parser.add_argument("--item-info", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "derived",
    )
    parser.add_argument(
        "--allow-official-file-update",
        action="store_true",
        help="Proceed after an official PixelRec file update; all semantic checks still apply.",
    )
    args = parser.parse_args()

    for path in (args.interaction, args.item_info):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        expected_hash = OFFICIAL_INPUT_SHA256.get(path.name)
        if expected_hash and observed_hash != expected_hash and not args.allow_official_file_update:
            raise RuntimeError(
                f"{path.name} differs from the locked official download. "
                "Use --allow-official-file-update only after documenting the new version."
            )

    interactions = pd.read_csv(args.interaction)
    required_interaction = {"user_id", "item_id", "timestamp"}
    if not required_interaction.issubset(interactions.columns):
        raise ValueError(f"interaction.csv lacks {sorted(required_interaction - set(interactions.columns))}")
    interactions.insert(0, "row_id", np.arange(len(interactions), dtype=np.int64))
    item_info = pd.read_csv(args.item_info, low_memory=False)

    assert_equal("raw rows", len(interactions), EXPECTED["raw_rows"])
    assert_equal("raw users", interactions.user_id.nunique(), EXPECTED["raw_users"])
    assert_equal("raw items", interactions.item_id.nunique(), EXPECTED["raw_items"])

    definitions = {
        "wide": (5, 3, "wide_5u3i"),
        "main": (10, 5, "main_10u5i"),
        "strict": (15, 5, "strict_15u5i"),
    }
    cores: dict[str, pd.DataFrame] = {}
    for key, (min_user, min_item, stem) in definitions.items():
        core = iterative_kcore(interactions, min_user, min_item)
        cores[key] = core
        observed = (len(core), core.user_id.nunique(), core.item_id.nunique())
        assert_equal(f"{key} core dimensions", observed, EXPECTED[key])
        write_csv(
            pd.DataFrame({"user_idx": range(core.user_id.nunique()), "user_id": sorted(core.user_id.unique())}),
            args.output_dir / f"{stem}_user_index.csv",
        )
        write_csv(
            pd.DataFrame({"item_idx": range(core.item_id.nunique()), "item_id": sorted(core.item_id.unique())}),
            args.output_dir / f"{stem}_item_index.csv",
        )
        write_csv(
            pd.DataFrame({"row_id": np.sort(core.row_id.to_numpy(np.int64))}),
            args.output_dir / f"{stem}_interaction_row_ids.csv.gz",
        )

    main_core = cores["main"].sort_values(
        ["user_id", "timestamp", "row_id"], kind="stable"
    ).copy()
    reverse_rank = main_core.groupby("user_id", sort=False).cumcount(ascending=False)
    main_core["split"] = np.where(
        reverse_rank.eq(0), "test", np.where(reverse_rank.eq(1), "validation", "train")
    )
    train_items = set(main_core.loc[main_core.split.eq("train"), "item_id"])
    main_core["warm_item"] = main_core.item_id.isin(train_items)
    split_path = args.output_dir / "main_10u5i_leave_last_two_split.csv.gz"
    write_csv(
        main_core[["row_id", "user_id", "item_id", "timestamp", "split", "warm_item"]],
        split_path,
    )

    assignment = build_assignment_metadata(
        interactions,
        np.sort(cores["main"].row_id.to_numpy(np.int64)),
        item_info,
    )
    assignment_path = args.output_dir / "model_aware_provenance_assignment_v2.csv"
    write_csv(assignment, assignment_path)

    checks = {
        "main_split_uncompressed_sha256": sha256_uncompressed(split_path),
        "main_user_index_sha256": sha256_file(args.output_dir / "main_10u5i_user_index.csv"),
        "main_item_index_sha256": sha256_file(args.output_dir / "main_10u5i_item_index.csv"),
        "main_row_index_uncompressed_sha256": sha256_uncompressed(
            args.output_dir / "main_10u5i_interaction_row_ids.csv.gz"
        ),
        "strict_user_index_sha256": sha256_file(args.output_dir / "strict_15u5i_user_index.csv"),
        "strict_item_index_sha256": sha256_file(args.output_dir / "strict_15u5i_item_index.csv"),
        "strict_row_index_uncompressed_sha256": sha256_uncompressed(
            args.output_dir / "strict_15u5i_interaction_row_ids.csv.gz"
        ),
        "catalog_item_id_vector_sha256": vector_sha256(assignment["item_id"]),
        "model_aware_block_vector_sha256": vector_sha256(assignment["model_aware_block"]),
        "selected_label_vector_sha256": vector_sha256(assignment["provenance_code_v2"]),
    }
    for key, value in checks.items():
        assert_equal(key, value, EXPECTED[key])

    report = {
        "status": "passed",
        "official_inputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in (args.interaction, args.item_info)
        },
        "dimensions": {
            key: {
                "interactions": len(frame),
                "users": frame.user_id.nunique(),
                "items": frame.item_id.nunique(),
            }
            for key, frame in cores.items()
        },
        "checks": checks,
    }
    (args.output_dir / "data_preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
