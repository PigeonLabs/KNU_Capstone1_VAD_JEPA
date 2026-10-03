"""Prevent cross-device caption leakage and replay alarms before filtering GT."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ipad_jepa.experiment import metrics
from report_cached_score_distributions import collect, population_note


def populations(normal, anomaly, unknown, valid_cal, trace_cal, videos):
    return [dict(treatment=kind, known_normal_test={'targets': normal},
                 known_anomaly_test={'targets': anomaly}, unknown_targets=unknown,
                 inference_targets=normal + anomaly + unknown, test_videos=videos,
                 normal_calibration={'targets': valid_cal}, calibration_total_trace_rows=trace_cal)
            for kind in ['frozen', 'lora']]


@pytest.mark.parametrize('values,expected,excluded', [
    ((6261, 2949, 18, 2809, 2864, 15), ['9,210', '9,228', '18 unknown', '2,809'], ['11,563', '6,641']),
    ((6641, 4922, 0, 2727, 2771, 17), ['11,563', '6,641', '0 unknown', '2,727'], ['9,210', '9,228', '2,809']),
])
def test_caption_uses_actual_device_population_without_previous_device_counts(values, expected, excluded):
    note = population_note(populations(*values))
    assert all(word in note for word in expected)
    assert all(word not in note for word in excluded)


@pytest.mark.parametrize('fault', ['unequal_classes', 'unknown_not_in_total', 'calibration_overflow', 'reversed_pair'])
def test_equal_global_sizes_cannot_hide_incompatible_caption_populations(fault):
    rows = populations(6641, 4922, 0, 2727, 2771, 17)
    if fault == 'unequal_classes':
        rows[1]['known_normal_test']['targets'] -= 1
        rows[1]['known_anomaly_test']['targets'] += 1
    elif fault == 'unknown_not_in_total':
        rows[1]['unknown_targets'] = 1
    elif fault == 'calibration_overflow':
        rows[1]['normal_calibration']['targets'] = 2772
    else:
        rows.reverse()
    with pytest.raises(ValueError):
        population_note(rows)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def replay_fixture(tmp_path):
    # Unknown GT at index1 must preserve the streak. Tied q99 at4 and invalid
    # inference at5 reset it. Removing unknown rows first changes known alarms.
    labels = np.array([0, -1, 1, 0, 0, -1, 0, 1, 0, 1])
    scores = np.array([2., 2., 2., 2., 1., 2., 2., 2., 2., 2.])
    inference = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    valid = inference & (labels != -1)
    alarm = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    rows = [dict(frame=i, label=int(labels[i]), score=scores[i], inference_valid=int(inference[i]),
                 valid=int(valid[i]), alarm=alarm[i]) for i in range(len(labels))]
    write_csv(tmp_path / 'P3/01.csv', rows)
    cal = [dict(P3=1, valid=1) for _ in range(100)] + [dict(P3=99, valid=0)]
    write_csv(tmp_path / 'normal_calibration.csv', cal)
    (tmp_path / 'normal_fit.json').write_text(json.dumps(dict(calibration_valid_frames=100,
        calibration={'P3': {'threshold': 1}})))
    (tmp_path / 'metrics.json').write_text(json.dumps(dict(test_videos=1,
        variants={'P3': metrics(labels[valid], scores[valid])})))
    audited = dict(treatment='frozen', test_videos=1,
        videos={'01': dict(normal_targets=5, normal_alarm_frames=2, unknown_targets=1, inference_targets=9)},
        aggregate=dict(normal_targets=5, normal_alarm_frames=2, unknown_targets=1, inference_targets=9))
    return audited, rows


def test_unknown_gt_preserves_streak_but_q99_tie_and_invalid_inference_reset_it(tmp_path):
    audited, _ = replay_fixture(tmp_path)
    summary, _, _ = collect(tmp_path, audited)
    assert summary['normal_calibration']['targets'] == 100
    assert summary['normal_calibration']['strict_threshold_crossings'] == 0
    assert summary['calibration_total_trace_rows'] == 101
    assert (summary['inference_targets'], summary['unknown_targets']) == (9, 1)
    assert summary['known_normal_test']['strict_threshold_crossings'] == 4
    assert summary['known_normal_test']['streak_alarm_frames'] == 2
    assert summary['known_anomaly_test']['strict_threshold_crossings'] == 3
    assert summary['known_anomaly_test']['streak_alarm_frames'] == 2


@pytest.mark.parametrize('fault', ['filtered_GT_alarm', 'inconsistent_unknown_total'])
def test_replay_refuses_a_filtered_GT_alarm_or_unreconciled_audit_total(tmp_path, fault):
    audited, rows = replay_fixture(tmp_path)
    if fault == 'filtered_GT_alarm':
        rows[2]['alarm'] = 0
        write_csv(tmp_path / 'P3/01.csv', rows)
    else:
        audited = deepcopy(audited)
        audited['aggregate']['unknown_targets'] = 0
    with pytest.raises((AssertionError, ValueError)):
        collect(tmp_path, audited)
