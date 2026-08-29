import numpy as np

from src.interventions import select_display
from src.choice_model import choice_from_uniform, softmax_with_outside
from src.score_normalization import apply_anchor, candidate_anchor


def test_candidate_anchor_shared_across_intervention_branches():
    scores = np.linspace(-2, 3, 30, dtype=np.float32)[None, :]
    mean, sd = candidate_anchor(scores)
    groups = np.arange(30, dtype=np.uint8) % 3
    candidates = np.arange(30, dtype=np.int32)
    normal = select_display("none", candidates, scores[0], groups, 0, 20)
    quota = select_display("quota_reranking", candidates, scores[0], groups, 0, 20)
    anchored = apply_anchor(scores, mean, sd)[0]
    assert np.isclose(anchored[normal[-1]], (scores[0, normal[-1]] - mean[0, 0]) / sd[0, 0])
    assert np.isclose(anchored[quota[-1]], (scores[0, quota[-1]] - mean[0, 0]) / sd[0, 0])


def test_lower_ranked_replacement_retains_lower_anchored_utility():
    scores = np.arange(30, dtype=np.float32)[None, :]
    mean, sd = candidate_anchor(scores)
    z = apply_anchor(scores, mean, sd)[0]
    assert z[5] < z[20]


def test_zero_variance_safeguard():
    scores = np.ones((2, 24), np.float32)
    mean, sd = candidate_anchor(scores)
    assert np.all(sd == 1.0)
    assert np.all(apply_anchor(scores, mean, sd) == 0.0)


def test_outside_option_is_first_category():
    probabilities = softmax_with_outside(np.array([0.0, 0.0]), outside_utility=0.5)
    assert choice_from_uniform(probabilities, 0.0) == -1
    assert choice_from_uniform(probabilities, float(probabilities[0] + 1e-6)) == 0
