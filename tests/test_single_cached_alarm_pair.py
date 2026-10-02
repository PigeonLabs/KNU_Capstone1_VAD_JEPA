"""Frozen/LoRA alarm comparisons must pair actual coverage and GT boundaries."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_cached_alarm_coverage import aggregate, cached_events
from report_single_cached_alarm_pair import verify_pair


def pair():
    rows = []
    for kind, alarms in [('frozen', [0, 0, 0, 0, 0, 0]), ('lora', [0, 1, 1, 1, 0, 0])]:
        videos = {'01': cached_events(np.array([0, 1, 1, 0, -1, 0]), np.arange(6), np.array(alarms))}
        rows.append(dict(treatment=kind, backbone='dinov3-l', mode='offline', device='R02', seed=0,
                         threshold_fixed_on_normal=True, test_videos=1, videos=videos,
                         aggregate=aggregate(videos)))
    return rows


def test_pair_keeps_detected_event_and_normal_carryover_with_no_new_episode():
    rows = pair(); verify_pair(rows)
    assert rows[0]['aggregate']['detected_events'] == 0
    assert rows[1]['aggregate']['detected_events'] == 1
    assert rows[1]['aggregate']['normal_alarm_frames'] == 1
    assert rows[1]['aggregate']['normal_alarm_episode_starts'] == 0


@pytest.mark.parametrize('key,value', [('seed', 1), ('device', 'R01'), ('mode', 'online'),
                                      ('backbone', 'vjepa21-l')])
def test_mixed_condition_refused(key, value):
    rows = pair(); rows[1][key] = value
    with pytest.raises(ValueError, match='identity'): verify_pair(rows)


def test_equal_total_counts_cannot_hide_shifted_gt_boundaries():
    rows = pair(); rows[1]['videos']['01']['events'][0]['start_frame'] = 0
    with pytest.raises(ValueError, match='boundaries'): verify_pair(rows)


def test_equal_global_totals_cannot_hide_different_per_video_coverage():
    rows = pair()
    for row in rows:
        row['videos']['02'] = deepcopy(row['videos']['01']); row['test_videos'] = 2
    rows[1]['videos']['01']['normal_targets'] += 1
    rows[1]['videos']['02']['normal_targets'] -= 1
    for row in rows: row['aggregate'] = aggregate(row['videos'])
    with pytest.raises(ValueError, match='coverage'): verify_pair(rows)


def test_missing_test_video_refused():
    rows = pair(); rows[1]['videos'] = {'02': rows[1]['videos']['01']}
    with pytest.raises(ValueError, match='inventory'): verify_pair(rows)


@pytest.mark.parametrize('change', ['aggregate', 'normal_threshold', 'video_count'])
def test_inconsistent_or_retuned_accounting_refused(change):
    rows = pair()
    if change == 'aggregate': rows[1]['aggregate']['detected_events'] = 2
    elif change == 'normal_threshold': rows[1]['threshold_fixed_on_normal'] = False
    else: rows[1]['test_videos'] = 2
    with pytest.raises(ValueError, match='accounting'): verify_pair(rows)
