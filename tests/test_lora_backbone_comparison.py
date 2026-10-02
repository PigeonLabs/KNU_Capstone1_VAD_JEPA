"""Verify matched groups, paired direction, seed means and equal-device weights."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_lora_backbone_comparison import matched_pairs, paired_statistics


def run(perfect=True, factor=1):
    labels = np.tile(np.array([0, 1, 0, 1]), factor)
    score = labels.astype(float) if perfect else 1. - labels
    return {v: (np.arange(len(labels)), labels.copy(), score.copy()) for v in ('01', '02')}


def test_only_matching_mode_device_groups_are_eligible():
    groups = {('dinov3-l', 'offline', 'R01'): {}, ('vjepa21-l', 'offline', 'R01'): {},
              ('dinov3-l', 'online', 'R02'): {}, ('vjepa21-l', 'online', 'R03'): {}}
    assert matched_pairs(groups, False) == [('offline', 'R01')]
    groups.pop(('vjepa21-l', 'offline', 'R01'))
    with pytest.raises(ValueError, match='Both backbones'): matched_pairs(groups, False)


def test_complete_macro_backbone_pair_keeps_its_distinct_scope():
    assert matched_pairs({(m, 'offline', None): {} for m in ('dinov3-l', 'vjepa21-l')}, True) == [('offline', None)]
    with pytest.raises(ValueError):
        matched_pairs({(m, 'offline', 'R01'): {} for m in ('dinov3-l', 'vjepa21-l')}, True)


def test_candidate_minus_reference_averages_seed_metrics_without_pooling_scores():
    reference = [run(), run(), run(False)]
    # A large seed-specific offset must not enter the mean of seed metrics.
    for values in reference[1].values(): values[2][:] += 100
    candidate = [run(), run(), run()]
    point, low, high, rejected = paired_statistics({'R01': reference}, {'R01': candidate}, False)
    np.testing.assert_allclose(point, [1/3, 1/6], atol=1e-14)
    np.testing.assert_allclose(low, point); np.testing.assert_allclose(high, point)
    assert rejected == 0
    reverse, _, _, _ = paired_statistics({'R01': candidate}, {'R01': reference}, False)
    np.testing.assert_allclose(reverse, -point)


def test_macro_uses_equal_device_weights_despite_unequal_frame_counts():
    reference, candidate = {}, {}
    for device in ('R01', 'R02', 'R03', 'R04'):
        factor = 10 if device == 'R04' else 1
        reference[device] = [run(factor=factor) for _ in range(3)]
        candidate[device] = [run(device != 'R01', factor) for _ in range(3)]
    point, low, high, rejected = paired_statistics(reference, candidate, True)
    np.testing.assert_allclose(point, [-.25, -.125], atol=1e-14)
    np.testing.assert_allclose(low, point); np.testing.assert_allclose(high, point)
    assert rejected == 0
    reference.pop('R04'); candidate.pop('R04')
    with pytest.raises(ValueError, match='Macro4'): paired_statistics(reference, candidate, True)


@pytest.mark.parametrize('fault', ['missing_seed', 'different_device', 'different_frame', 'different_gt', 'different_video'])
def test_unpaired_inputs_are_rejected(fault):
    reference = {'R01': [run() for _ in range(3)]}; candidate = copy.deepcopy(reference)
    if fault == 'missing_seed': candidate['R01'].pop()
    if fault == 'different_device': candidate['R02'] = candidate.pop('R01')
    if fault == 'different_frame': candidate['R01'][2]['01'][0][0] += 1
    if fault == 'different_gt': candidate['R01'][2]['01'][1][0] = 1
    if fault == 'different_video': candidate['R01'][2]['03'] = candidate['R01'][2].pop('01')
    with pytest.raises(ValueError): paired_statistics(reference, candidate, False)
