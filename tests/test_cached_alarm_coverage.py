"""Cached alarm accounting must keep coverage, unknown boundaries and seed scope."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_cached_alarm_coverage import cached_events, aggregate, group_conditions


def test_detection_and_outside_coverage_remain_distinct_without_emission_times():
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    row = cached_events(labels, np.arange(3, 11), np.array([0, 1, 0, 1, 0, 0, 0, 0]))
    assert row['covered_events'] == 2 and row['events_outside_coverage'] == 1
    assert row['detected_events'] == 1 and row['missed_events'] == 1
    assert row['events'][1]['first_alarm_target'] == 6
    assert row['normal_alarm_episode_starts'] == 1 and row['normal_alarm_frames'] == 1
    assert not any('seconds' in key or 'delay' in key for key in row['events'][1])
    json.dumps(row)


def test_unknown_gt_does_not_create_a_new_alarm_episode_after_filtering():
    labels = np.array([0, -1, 0, 1, 1, -1, 0])
    row = cached_events(labels, np.arange(7), np.ones(7, dtype=int))
    assert row['normal_alarm_episode_starts'] == 1
    assert row['unknown_alarm_episode_starts'] == 0
    assert row['normal_alarm_frames'] == 3 and row['unknown_targets'] == 2
    assert row['uncertain_boundary_covered_events'] == 1
    assert row['events'][0]['offset_uncertain']
    json.dumps(row)


def test_alarm_carrying_from_anomaly_into_normal_is_not_a_new_normal_episode():
    row = cached_events(np.array([0, 1, 1, 0, 0]), np.arange(5), np.array([0, 1, 1, 1, 0]))
    assert row['normal_alarm_episode_starts'] == 0 and row['normal_alarm_frames'] == 1


def test_unobserved_frame_gap_starts_a_new_episode_without_merging_gt_events():
    row = cached_events(np.zeros(7, dtype=int), np.array([1, 2, 4, 5]), np.ones(4, dtype=int))
    assert row['normal_alarm_episode_starts'] == 2
    summary = aggregate({'01': row})
    assert summary['observed_event_recall'] is None and summary['normal_alarm_frame_rate'] == 1


@pytest.mark.parametrize('frames,alarms', [([1, 1], [0, 1]), ([2, 1], [0, 1]),
                                        ([-1], [0]), ([8], [0]), ([1], [2])])
def test_invalid_inventory_rejected(frames, alarms):
    with pytest.raises(ValueError):
        cached_events(np.zeros(8, dtype=int), np.array(frames), np.array(alarms))


def condition(seed, detected, covered=2):
    return {'treatment': 'lora', 'backbone': 'dinov3-l', 'mode': 'offline', 'device': 'R01', 'seed': seed,
            'aggregate': {'covered_events': covered, 'normal_targets': 5, 'unknown_targets': 0,
                          'inference_targets': 8, 'observed_event_recall': detected / covered}}


def test_exact_three_seed_mean_requires_same_coverage_and_rejects_duplicate_seed():
    rows = [condition(0, 0), condition(1, 1), condition(2, 2)]
    group = group_conditions(rows)[0]
    assert group['observed_event_recall_mean'] == .5
    assert group['observed_event_recall_seed_values'] == [0, .5, 1]
    assert group_conditions(rows[:2]) == []
    with pytest.raises(ValueError, match='Duplicate'): group_conditions(rows + [rows[0]])
    with pytest.raises(ValueError, match='coverage'): group_conditions(rows[:2] + [condition(2, 2, 3)])
