import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_case():
    rng = np.random.default_rng(11)
    n_users, n_items, dimension = 2, 30, 5
    U = rng.normal(0, 0.1, (n_users, dimension)).astype(np.float32)
    V = rng.normal(0, 0.1, (n_items, dimension)).astype(np.float32)
    panel = np.array([0, 1], np.int32)
    pools = np.tile(np.arange(24, dtype=np.int32), (2, 1))
    groups = np.arange(n_items, dtype=np.uint8) % 3
    orders = np.tile(np.arange(2, dtype=np.int32), (2, 1))
    negative = np.full((2, 2), 0.25)
    return U, V, panel, pools, groups, orders, negative
