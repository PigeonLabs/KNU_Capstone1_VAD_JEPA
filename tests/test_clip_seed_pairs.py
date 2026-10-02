"""Partial-pair reporting must preserve individual metrics and exact paired targets."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_clip_seed_pairs import pair_points, partial_seeds


@pytest.mark.parametrize('seeds', [[], [0, 0], [0, 1, 2], [-1], [3]])
def test_invalid_or_complete_seed_group_cannot_be_presented_as_partial(seeds):
    with pytest.raises(ValueError, match='One or two distinct'):
        partial_seeds(seeds)


@pytest.mark.parametrize('seeds,expected', [([1], [1]), ([1, 0], [0, 1]), ([2, 0], [0, 2])])
def test_partial_report_keeps_exact_fixed_seed_members(seeds, expected):
    assert partial_seeds(seeds) == expected


def test_individual_points_do_not_create_seed_mean_or_interval():
    frames, labels = np.arange(4), np.array([0, 1, 0, 1])
    runs = {16: {'01': (frames, labels, labels.astype(float))},
            8: {'01': (frames, labels, np.zeros(4))}}
    rows = pair_points(runs, 1)
    assert [row['seed'] for row in rows] == [1, 1]
    assert [row['clip_frames'] for row in rows] == [16, 8]
    assert [row['auroc'] for row in rows] == [1., .5]
    assert [row['ap'] for row in rows] == [1., .5]
    assert all(row['frames'] == 4 and row['anomaly_frames'] == 2 for row in rows)
    assert all(not any('mean' in key or 'ci' in key for key in row) for row in rows)


def test_paired_valid_targets_cannot_differ_even_when_metrics_match():
    frames, labels = np.arange(4), np.array([0, 1, 0, 1])
    runs = {16: {'01': (frames, labels, labels.astype(float))},
            8: {'01': (frames + 1, labels, labels.astype(float))}}
    with pytest.raises(ValueError, match='Paired valid frame/label'):
        pair_points(runs, 0)
