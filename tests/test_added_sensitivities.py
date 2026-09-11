from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_lower_feedback_calibration_and_outputs():
    calibration = pd.read_csv(ROOT / "tables/added_sensitivity/outside_utility_calibration.csv")
    assert calibration["target_initial_acceptance"].tolist() == [0.1, 0.2, 0.4]
    assert np.allclose(
        calibration["calibrated_initial_acceptance"],
        calibration["target_initial_acceptance"],
        rtol=0,
        atol=1e-12,
    )
    endpoints = pd.read_csv(ROOT / "tables/added_sensitivity/lower_acceptance_endpoints.csv")
    assert set(endpoints["scenario"]) == {
        "zero", "moderate_asymmetric", "strong_premium_penalty", "ai_appreciation"
    }
    assert set(endpoints["target_initial_acceptance"]) == {0.1, 0.2, 0.4}
    assert endpoints["actual_acceptance_rate"].between(0, 1).all()


def test_candidate_refresh_pairing_and_turnover():
    dynamic = pd.read_csv(ROOT / "tables/added_sensitivity/dynamic_candidates_endpoints.csv")
    assert dynamic["mean_round6_candidate_turnover"].between(0.384, 0.386).all()
    paired = pd.read_csv(ROOT / "tables/added_sensitivity/dynamic_minus_fixed_paired_LMC.csv")
    assert ((paired["ci_low"] <= 0) & (paired["ci_high"] >= 0)).all()
    expected = {
        "moderate_asymmetric": -5.644488223402466e-06,
        "strong_premium_penalty": -0.0008298290247163337,
        "ai_appreciation": -0.001013269852375709,
    }
    observed = paired.set_index("scenario")["estimate"]
    for scenario, value in expected.items():
        assert np.isclose(observed.loc[scenario], value, rtol=0, atol=1e-12)
