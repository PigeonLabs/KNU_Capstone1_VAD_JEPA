"""Trace diagnosis must follow actual FP32 phase casting and fixed P3 calibration."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_runtime_pair import normalized_delta_components, phase_bins


def test_phase_bank_bins_follow_tensor_float32_cast_at_rounding_boundary():
    values = [0., .062499999, .0625, .7956128982293261, .999]
    actual = (torch.tensor(values, dtype=torch.float32) * 16).long().numpy()
    np.testing.assert_array_equal(phase_bins(values, 16), actual)
    assert phase_bins([.062499999], 16)[0] == 1
    assert int(np.floor(.062499999 * 16)) == 0


@pytest.mark.parametrize('phases', [[1.], [1. - 1e-10], [-.01], [np.nan], [np.inf]])
def test_invalid_phase_after_runtime_cast_is_not_assigned_a_bank(phases):
    with pytest.raises(ValueError, match='Valid FP32'):
        phase_bins(phases, 16)


def test_two_component_contributions_reproduce_score_change_with_same_calibration():
    old = np.array([[.4, .10], [.7, .02]])
    new = np.array([[.5, .099], [.6, .03]])
    median, scale = np.array([.2, .05]), np.array([2., .001])
    contributions = normalized_delta_components(old, new, scale)
    expected = ((new - median) / scale).mean(1) - ((old - median) / scale).mean(1)
    np.testing.assert_allclose(contributions.sum(1), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(contributions[0], [.025, -.5], atol=1e-12, rtol=0)


@pytest.mark.parametrize('scale', [[0., 1.], [-1., 1.], [np.nan, 1.], [1.]])
def test_invalid_fixed_calibration_cannot_explain_score_deltas(scale):
    with pytest.raises(ValueError, match='positive calibration'):
        normalized_delta_components([[.4, .1]], [[.5, .1]], scale)
