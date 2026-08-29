import numpy as np

from src.exposure_metrics import discounted_exposure, human_ai_log_gap, relative_exposure, total_variation_exposure
from src.quota_reranker import hard_quota_rerank, quota_for_round


def test_rotating_quota_counts():
    candidates=np.arange(60,dtype=np.int32); scores=np.arange(60,dtype=float); groups=np.repeat(np.arange(3,dtype=np.uint8),20)
    for round_index, expected in enumerate(([7,7,6],[7,6,7],[6,7,7])):
        selected=hard_quota_rerank(candidates,scores,groups,round_index)
        assert np.bincount(groups[selected],minlength=3).tolist()==list(expected)
        assert quota_for_round(round_index).tolist()==list(expected)


def test_quota_fallback_uses_highest_remaining():
    candidates=np.arange(30,dtype=np.int32); scores=np.arange(30,dtype=float); groups=np.array([0]*3+[1]*14+[2]*13,dtype=np.uint8)
    selected=hard_quota_rerank(candidates,scores,groups,0)
    assert len(selected)==20 and len(set(selected))==20
    assert np.bincount(groups[selected],minlength=3)[0]==3
    assert set(range(3)).issubset(set(selected))


def test_discounted_exposure():
    items=np.array([0,1,2]); groups=np.array([0,1,2],dtype=np.uint8)
    got=discounted_exposure(items,groups)
    assert np.allclose(got,1/np.log2(np.arange(2,5)))


def test_relative_exposure_and_log_gap():
    exposure=np.array([1.,2.,3.]); catalog=np.array([1/3,1/3,1/3])
    assert np.allclose(relative_exposure(exposure,catalog),[.5,1.,1.5])
    assert np.isclose(human_ai_log_gap(exposure,catalog),np.log(3))


def test_three_group_total_variation():
    exposure=np.array([1.,2.,3.]); catalog=np.array([1/3,1/3,1/3])
    expected=.5*np.abs(np.array([1/6,2/6,3/6])-catalog).sum()
    assert np.isclose(total_variation_exposure(exposure,catalog),expected)
