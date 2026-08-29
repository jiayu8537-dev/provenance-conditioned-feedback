from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bpr_training import sample_negatives
from .data_loading import DataBundle


@dataclass(frozen=True)
class LightGCNTrainingConfig:
    dimension: int = 32
    layers: int = 2
    learning_rate: float = 0.01
    regularization: float = 0.0001
    optimization_passes: int = 8
    sample_size: int = 80000
    seed: int = 20264720


def normalized_bipartite_adjacency(
    n_users: int,
    n_items: int,
    users: np.ndarray,
    items: np.ndarray,
) -> torch.Tensor:
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    item_nodes = n_users + items
    rows = np.concatenate([users, item_nodes])
    columns = np.concatenate([item_nodes, users])
    degree = np.bincount(rows, minlength=n_users + n_items).astype(np.float32)
    values = 1.0 / np.sqrt(
        np.maximum(degree[rows], 1.0) * np.maximum(degree[columns], 1.0)
    )
    indices = torch.from_numpy(np.vstack([rows, columns])).long()
    tensor_values = torch.from_numpy(values).float()
    return torch.sparse_coo_tensor(
        indices,
        tensor_values,
        size=(n_users + n_items, n_users + n_items),
    ).coalesce()


class FullGraphLightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dimension: int,
        layers: int,
        adjacency: torch.Tensor,
        raw_embeddings: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.layers = int(layers)
        self.raw_embeddings = nn.Parameter(
            torch.empty(self.n_users + self.n_items, int(dimension))
        )
        if raw_embeddings is None:
            nn.init.normal_(self.raw_embeddings, std=0.01)
        else:
            values = np.asarray(raw_embeddings, dtype=np.float32)
            if values.shape != tuple(self.raw_embeddings.shape):
                raise ValueError(
                    f"Expected raw embeddings {tuple(self.raw_embeddings.shape)}; "
                    f"received {values.shape}"
                )
            with torch.no_grad():
                self.raw_embeddings.copy_(torch.from_numpy(values))
        self.adjacency = adjacency

    def set_adjacency(self, adjacency: torch.Tensor) -> None:
        self.adjacency = adjacency

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.raw_embeddings
        layers = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            layers.append(current)
        combined = torch.stack(layers, dim=0).mean(dim=0)
        return combined[: self.n_users], combined[self.n_users :]


def weighted_bpr_loss(
    model: FullGraphLightGCN,
    users: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    regularization: float,
    weights: np.ndarray | None = None,
) -> torch.Tensor:
    user_index = torch.from_numpy(np.asarray(users, dtype=np.int64))
    positive_index = torch.from_numpy(np.asarray(positives, dtype=np.int64))
    negative_index = torch.from_numpy(np.asarray(negatives, dtype=np.int64))
    user_embeddings, item_embeddings = model.propagate()
    margins = (
        user_embeddings[user_index]
        * (
            item_embeddings[positive_index]
            - item_embeddings[negative_index]
        )
    ).sum(dim=1)
    losses = -F.logsigmoid(margins)
    if weights is not None:
        tensor_weights = torch.from_numpy(np.asarray(weights, dtype=np.float32))
        losses = losses * tensor_weights
    raw = model.raw_embeddings
    penalty = (
        raw[user_index].square().sum(dim=1)
        + raw[model.n_users + positive_index].square().sum(dim=1)
        + raw[model.n_users + negative_index].square().sum(dim=1)
    ).mean()
    return losses.mean() + float(regularization) * penalty


def train_full_history_lightgcn(
    data: DataBundle,
    config: LightGCNTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    adjacency = normalized_bipartite_adjacency(
        data.n_users,
        data.n_items,
        data.all_users,
        data.all_items,
    )
    model = FullGraphLightGCN(
        data.n_users,
        data.n_items,
        config.dimension,
        config.layers,
        adjacency,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rows = []
    for optimization_pass in range(1, config.optimization_passes + 1):
        size = min(config.sample_size, len(data.all_users))
        selected = rng.choice(len(data.all_users), size=size, replace=False)
        users = data.all_users[selected]
        positives = data.all_items[selected]
        negatives = sample_negatives(users, data.full_keys, data.n_items, rng)
        loss = weighted_bpr_loss(
            model,
            users,
            positives,
            negatives,
            config.regularization,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rows.append(
            {
                "optimization_pass": optimization_pass,
                "loss": float(loss.detach()),
            }
        )
    with torch.no_grad():
        users, items = model.propagate()
    return (
        model.raw_embeddings.detach().cpu().numpy().astype(np.float32),
        users.detach().cpu().numpy().astype(np.float32),
        items.detach().cpu().numpy().astype(np.float32),
        pd.DataFrame(rows),
    )


def load_or_train_lightgcn(
    data: DataBundle,
    artifact_path: Path,
    log_path: Path,
    manifest_path: Path,
    config: LightGCNTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = {
        "n_users": data.n_users,
        "n_items": data.n_items,
        "dimension": config.dimension,
        "layers": config.layers,
        "learning_rate": config.learning_rate,
        "regularization": config.regularization,
        "optimization_passes": config.optimization_passes,
        "sample_size": config.sample_size,
        "seed": config.seed,
        "history": "all rows of leave-last-two split",
        "implementation": "LightGCN with full-graph normalized propagation",
    }
    force_retrain = os.environ.get("JIIS_FORCE_RETRAIN", "0") == "1"
    if not force_retrain and artifact_path.exists() and manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current == expected:
            artifact = np.load(artifact_path)
            return (
                artifact["raw"].astype(np.float32),
                artifact["U"].astype(np.float32),
                artifact["V"].astype(np.float32),
            )
    raw, users, items, logs = train_full_history_lightgcn(data, config)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, raw=raw, U=users, V=items)
    logs.to_csv(log_path, index=False)
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return raw, users, items
