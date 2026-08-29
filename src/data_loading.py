from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


GROUP_NAMES = np.array(
    ["AI-attributed", "Human-AI-attributed", "Human-attributed"]
)


@dataclass
class DataBundle:
    split: pd.DataFrame
    assignment: pd.DataFrame
    users: list[str]
    items: list[str]
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    n_users: int
    n_items: int
    all_users: np.ndarray
    all_items: np.ndarray
    full_keys: np.ndarray
    full_csr: sparse.csr_matrix
    item_counts: np.ndarray
    warm_items: np.ndarray
    user_activity: np.ndarray
    popularity_bins: np.ndarray
    blocks: list[np.ndarray]


def load_data(split_path: str | Path, assignment_path: str | Path) -> DataBundle:
    split = pd.read_csv(split_path)
    assignment = pd.read_csv(assignment_path)
    required_split = {"user_id", "item_id", "split"}
    required_assignment = {"item_id", "model_aware_block"}
    if not required_split.issubset(split.columns):
        raise ValueError(f"Split file lacks {sorted(required_split - set(split.columns))}")
    if not required_assignment.issubset(assignment.columns):
        raise ValueError(
            f"Assignment file lacks {sorted(required_assignment - set(assignment.columns))}"
        )

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
    full_keys = np.unique(all_users.astype(np.int64) * n_items + all_items)
    full_csr = sparse.csr_matrix(
        (np.ones(len(all_users), np.int8), (all_users, all_items)),
        shape=(n_users, n_items),
    )
    item_counts = np.bincount(all_items, minlength=n_items).astype(np.float32)
    warm_items = np.where(item_counts > 0)[0].astype(np.int32)
    user_activity = np.bincount(all_users, minlength=n_users).astype(np.int32)
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
    return DataBundle(
        split=split,
        assignment=assignment,
        users=users,
        items=items,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        n_users=n_users,
        n_items=n_items,
        all_users=all_users,
        all_items=all_items,
        full_keys=full_keys,
        full_csr=full_csr,
        item_counts=item_counts,
        warm_items=warm_items,
        user_activity=user_activity,
        popularity_bins=popularity_bins,
        blocks=blocks,
    )


def qualified_assignment_seeds(
    candidate_path: str | Path,
    count: int,
    evenly_spaced: bool = True,
) -> tuple[list[int], pd.DataFrame]:
    candidates = pd.read_csv(candidate_path)
    qualified = candidates[
        (candidates.max_standardized_mean_difference < 0.05)
        & (candidates.max_theme_TV < 0.05)
        & (candidates.max_model_exposure_deviation < 0.10)
    ].sort_values("seed")
    if len(qualified) < count:
        raise ValueError(f"Only {len(qualified)} qualified assignments; need {count}")
    if evenly_spaced:
        indices = np.linspace(0, len(qualified) - 1, count, dtype=int)
        selected = qualified.iloc[indices]
    else:
        selected = qualified.iloc[:count]
    return selected.seed.astype(int).tolist(), selected.copy()


def generate_assignment(n_items: int, blocks: Iterable[np.ndarray], seed: int) -> np.ndarray:
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


def stratified_panel(
    activity: np.ndarray,
    panel_size: int,
    seed: int,
) -> np.ndarray:
    if panel_size % 5:
        raise ValueError("panel_size must be divisible by five")
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
    return np.array(sorted(panel), dtype=np.int32)


def make_candidate_pool(
    data: DataBundle,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    panel: np.ndarray,
    candidate_size: int,
    top_retrieval: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_size < top_retrieval:
        raise ValueError("candidate_size must be >= top_retrieval")
    rng = np.random.default_rng(seed)
    panel_size = len(panel)
    pools = np.empty((panel_size, candidate_size), np.int32)
    base_scores = np.empty((panel_size, candidate_size), np.float32)
    score_matrix = user_embeddings[panel] @ item_embeddings.T
    random_needed = candidate_size - top_retrieval
    for row, user in enumerate(panel):
        seen = data.full_csr.indices[
            data.full_csr.indptr[user] : data.full_csr.indptr[user + 1]
        ]
        if len(seen):
            score_matrix[row, seen] = -np.inf
        top = np.argpartition(score_matrix[row], -top_retrieval)[-top_retrieval:]
        top = top[np.argsort(-score_matrix[row, top], kind="stable")]
        used = set(seen.astype(int).tolist()) | set(top.astype(int).tolist())
        random_items: list[int] = []
        if random_needed:
            per_bin = int(np.ceil(random_needed / 5))
            for pop_bin in range(5):
                population = np.where(
                    (data.popularity_bins == pop_bin) & (data.item_counts > 0)
                )[0]
                eligible = np.array(
                    [item for item in population if int(item) not in used],
                    dtype=np.int32,
                )
                if len(eligible):
                    take = min(per_bin, len(eligible))
                    values = rng.choice(eligible, take, replace=False).astype(int).tolist()
                    random_items.extend(values)
                    used.update(values)
            if len(random_items) < random_needed:
                eligible = np.array(
                    [item for item in data.warm_items if int(item) not in used],
                    dtype=np.int32,
                )
                values = rng.choice(
                    eligible, random_needed - len(random_items), replace=False
                ).astype(int).tolist()
                random_items.extend(values)
        candidates = np.concatenate(
            [top.astype(np.int32), np.asarray(random_items[:random_needed], np.int32)]
        )
        if len(np.unique(candidates)) != candidate_size:
            raise RuntimeError("Candidate construction produced duplicates")
        pools[row] = candidates
        base_scores[row] = score_matrix[row, candidates]
    return pools, base_scores

