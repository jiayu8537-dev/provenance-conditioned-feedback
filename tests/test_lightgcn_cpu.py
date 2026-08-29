import numpy as np
import torch

from src.lightgcn_cpu import (
    FullGraphLightGCN,
    normalized_bipartite_adjacency,
    weighted_bpr_loss,
)


def test_normalized_adjacency_is_symmetric():
    users = np.array([0, 0, 1], np.int32)
    items = np.array([0, 1, 1], np.int32)
    adjacency = normalized_bipartite_adjacency(2, 2, users, items).to_dense()
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.count_nonzero(torch.diag(adjacency)) == 0


def test_full_graph_propagation_matches_layer_average():
    users = np.array([0, 1], np.int32)
    items = np.array([0, 1], np.int32)
    adjacency = normalized_bipartite_adjacency(2, 2, users, items)
    raw = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    model = FullGraphLightGCN(2, 2, 3, 1, adjacency, raw)
    got_users, got_items = model.propagate()
    expected = (torch.from_numpy(raw) + adjacency @ torch.from_numpy(raw)) / 2
    assert torch.allclose(got_users, expected[:2])
    assert torch.allclose(got_items, expected[2:])


def test_lightgcn_bpr_update_changes_raw_embeddings():
    users = np.array([0, 0, 1], np.int32)
    items = np.array([0, 1, 2], np.int32)
    adjacency = normalized_bipartite_adjacency(2, 3, users, items)
    model = FullGraphLightGCN(2, 3, 4, 2, adjacency)
    before = model.raw_embeddings.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = weighted_bpr_loss(
        model,
        np.array([0, 1], np.int32),
        np.array([0, 2], np.int32),
        np.array([2, 0], np.int32),
        0.0001,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.raw_embeddings.detach())
