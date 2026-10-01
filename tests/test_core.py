import numpy as np
import pytest
from ipad_jepa.temporal import clip_indices, common_mask, circular_phase, temporal_score
from ipad_jepa.memory import PrototypeMemory
from ipad_jepa.scoring import NormalCalibration


def test_online_uses_no_future_and_common_mask():
    assert np.array_equal(clip_indices(30,100,"online"), np.arange(15,31))
    assert np.array_equal(clip_indices(30,100,"offline"), np.arange(22,38))
    assert clip_indices(0,100,"online",padding=True).max() == 0
    assert np.flatnonzero(common_mask(100)).tolist() == list(range(19,93))
    with pytest.raises(ValueError, match="Incomplete"):
        clip_indices(0,100,"online")
    assert not common_mask(10).any()


def test_phase_wrap_progression_stop_and_reverse():
    phases = np.mod(.97+np.arange(20)*.01, 1)
    assert np.nanmax(temporal_score(phases,100)) < 1e-12
    assert np.nanmean(temporal_score(np.zeros(20),100)) > .02
    assert np.nanmean(temporal_score(phases[::-1],100)) > .04
    p = np.zeros((1,200)); p[0,0] = p[0,199] = .5
    predicted = circular_phase(p)[0]
    assert min(predicted,1-predicted) < 1e-6
    # Changing future phase estimates cannot change an earlier temporal score.
    altered = phases.copy(); altered[10:] = .4
    np.testing.assert_allclose(temporal_score(phases,100)[:10], temporal_score(altered,100)[:10], equal_nan=True)


def test_phase_memory_boundary_global_budget_and_projection():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80,6)).astype(np.float32)
    phase = np.repeat(np.arange(4)/4+.1,20)
    memory = PrototypeMemory(bins=4, per_bin=3, dimensions=3).fit(x,phase)
    assert memory.prototypes.shape == (4,3,3)
    assert memory.candidates(.01).shape == (9,3)
    assert memory.candidates(.99).shape == (9,3)
    assert memory.candidates(.01,True).shape == (12,3)
    query = x[:8]
    z, neighbour, distances = memory.nearest(query,.01,k=1)
    np.testing.assert_allclose(memory.patch_scores(query,.01,k=1), ((z-neighbour[:,0])**2).sum(1), atol=1e-6)
    memory.calibrate_temperature([(query,.01)])
    assert memory.temperature >= 1e-6
    assert np.isfinite(memory.frame_score(query,.01))
    with pytest.raises(ValueError, match="insufficient"):
        PrototypeMemory(bins=5, per_bin=3, dimensions=3).fit(x,phase)


def test_calibration_normal_only_and_zero_mad():
    cal = NormalCalibration().fit(np.ones((100,2)))
    np.testing.assert_allclose(cal.combine(np.ones((2,2))), 0)
    assert cal.combine(np.array([[2.,2.]]))[0] > cal.threshold
    assert cal.types(np.array([[1,1],[2,1],[1,2],[2,2]])).tolist() == [0,1,2,3]
    with pytest.raises(ValueError, match="finite"):
        NormalCalibration().fit(np.array([[np.nan,0]]))
