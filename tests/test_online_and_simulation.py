import numpy as np
import pandas as pd

from src.online_bpr import (
    HISTORICAL_REPLAY,
    update_batch_synchronous,
    update_pair,
)
from src.simulation import SimulationConfig, simulate_paired, zero_effect_drift


def test_frozen_parameters_unchanged(tiny_case):
    U, V, panel, pools, groups, orders, negative = tiny_case
    before_u, before_v = U.copy(), V.copy()
    simulate_paired(U, V, panel, pools, groups, np.zeros(3), np.full((2, 2), 0.5), negative, orders, SimulationConfig(rounds=2), "none")
    assert np.array_equal(U, before_u)
    assert np.array_equal(V, before_v)


def test_closed_parameters_change_after_accepted_interaction():
    U = np.zeros((1, 3), np.float32); U[0, 0] = 0.2
    V = np.zeros((2, 3), np.float32); V[0, 1] = 0.3; V[1, 2] = -0.2
    before = (U.copy(), V.copy())
    assert update_pair(U, V, 0, 0, 1, 0.01, 0.0001, steps=3) == 3
    assert not np.array_equal(U, before[0])
    assert not np.array_equal(V, before[1])


def test_synchronous_single_event_matches_sequential_update():
    U = np.zeros((1, 3), np.float32); U[0, 0] = 0.2
    V = np.zeros((2, 3), np.float32); V[0, 1] = 0.3; V[1, 2] = -0.2
    sequential_u, sequential_v = U.copy(), V.copy()
    synchronous_u, synchronous_v = U.copy(), V.copy()
    update_pair(
        sequential_u,
        sequential_v,
        0,
        0,
        1,
        0.01,
        0.0001,
        steps=3,
    )
    evaluations, operations = update_batch_synchronous(
        synchronous_u,
        synchronous_v,
        np.array([0], np.int32),
        np.array([0], np.int32),
        np.array([1], np.int32),
        0.01,
        0.0001,
        steps=3,
    )
    assert evaluations == 3
    assert operations == 3
    np.testing.assert_allclose(synchronous_u, sequential_u)
    np.testing.assert_allclose(synchronous_v, sequential_v)


def test_synchronous_update_is_event_order_invariant():
    U = np.array([[0.2, 0.1], [-0.1, 0.3]], np.float32)
    V = np.array([[0.4, -0.2], [0.1, 0.2], [-0.3, 0.5]], np.float32)
    first_u, first_v = U.copy(), V.copy()
    second_u, second_v = U.copy(), V.copy()
    update_batch_synchronous(
        first_u,
        first_v,
        np.array([0, 1], np.int32),
        np.array([0, 0], np.int32),
        np.array([1, 2], np.int32),
        0.01,
        0.0001,
        steps=3,
    )
    update_batch_synchronous(
        second_u,
        second_v,
        np.array([1, 0], np.int32),
        np.array([0, 0], np.int32),
        np.array([2, 1], np.int32),
        0.01,
        0.0001,
        steps=3,
    )
    np.testing.assert_allclose(first_u, second_u)
    np.testing.assert_allclose(first_v, second_v)


def test_exactly_k_updates_per_accepted_interaction(tiny_case):
    U, V, panel, pools, groups, orders, negative = tiny_case
    frame = simulate_paired(U,V,panel,pools,groups,np.zeros(3),np.full((2,2),0.5),negative,orders,SimulationConfig(rounds=2,online_steps=3),"none")
    closed = frame[(frame.branch=="closed") & (frame["round"]==2)].iloc[0]
    assert closed.online_sgd_updates == 3 * closed.accepted_interactions


def test_no_hidden_historical_replay():
    assert HISTORICAL_REPLAY is False
    assert SimulationConfig().update_regime == "event_online"
    assert SimulationConfig().replay_steps == 0


def test_fixed_budget_history_replay_preserves_update_count(tiny_case):
    U, V, panel, pools, groups, orders, negative = tiny_case
    histories = [np.array([24, 25], np.int32), np.array([26, 27], np.int32)]
    config = SimulationConfig(
        rounds=2,
        online_steps=3,
        update_regime="fixed_budget_history_replay",
        clicked_steps=2,
        replay_steps=1,
    )
    frame = simulate_paired(
        U,
        V,
        panel,
        pools,
        groups,
        np.zeros(3),
        np.full((2, 2), 0.5),
        negative,
        orders,
        config,
        "none",
        panel_histories=histories,
        replay_uniforms=np.full((2, 2), 0.25),
    )
    closed = frame[(frame.branch == "closed") & (frame["round"] == 2)].iloc[0]
    assert closed.online_sgd_updates == 3 * closed.accepted_interactions


def test_round_synchronous_separates_gradient_and_parameter_counts(tiny_case):
    U, V, panel, pools, groups, orders, negative = tiny_case
    frame = simulate_paired(
        U,
        V,
        panel,
        pools,
        groups,
        np.zeros(3),
        np.full((2, 2), 0.5),
        negative,
        orders,
        SimulationConfig(
            rounds=2,
            online_steps=3,
            update_regime="round_synchronous",
        ),
        "none",
    )
    closed = frame[(frame.branch == "closed") & (frame["round"] == 2)].iloc[0]
    assert (
        closed.event_gradient_evaluations
        == 3 * closed.accepted_interactions
    )
    assert closed.parameter_update_operations <= 3 * 2


def test_zero_effect_no_click_drift_is_zero(tiny_case):
    U,V,panel,pools,groups,orders,negative=tiny_case
    frame=simulate_paired(U,V,panel,pools,groups,np.zeros(3),np.zeros((2,2)),negative,orders,SimulationConfig(rounds=2),"none")
    assert zero_effect_drift(frame) == 0.0


def test_deterministic_reproducibility(tiny_case):
    U,V,panel,pools,groups,orders,negative=tiny_case
    args=(U,V,panel,pools,groups,np.array([-0.15,-0.05,0.05]),np.full((2,2),0.2),negative,orders,SimulationConfig(rounds=2),"none")
    pd.testing.assert_frame_equal(simulate_paired(*args), simulate_paired(*args))
