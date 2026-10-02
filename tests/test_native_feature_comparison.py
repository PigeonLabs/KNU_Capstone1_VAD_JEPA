"""Whole-protocol comparison requires actual matched targets and seed metrics."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_native_feature_methods import METHODS, matched_statistics, check_sources


def run(perfect=True):
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    return {v: (np.arange(4), labels.copy(), labels.astype(float) if perfect else 1. - labels)
            for v in ('01', '02')}


def methods():
    return {m: {s: run() for s in (0, 1, 2)} for m in METHODS}


def test_seed_metrics_not_pooled_scores_and_correct_delta_direction():
    data = methods()
    data['native_B0'][2] = run(False)
    for values in data['native_B0'][1].values():
        values[2][:] += 1000
    report = matched_statistics(data, count=20)
    pair = next(r for r in report['paired_deltas']
                if r['candidate'] == 'dinov3_L_P3' and r['reference'] == 'native_B0')
    np.testing.assert_allclose([pair['auroc_delta'], pair['ap_delta']], [1/3, 1/6])
    np.testing.assert_allclose([pair['auroc_delta_ci_low'], pair['auroc_delta_ci_high']], [1/3, 1/3])
    assert report['test_videos'] == 2 and report['common_frames'] == 8
    assert report['anomaly_frames'] == 4 and report['bootstrap_rejected'] == 0


def test_original_video_multiplicity_and_shared_draws_independent_reference():
    data = methods()
    labels = [np.array([0, 1, 0, 1]), np.array([0, 1, 1])]
    scores = [np.array([.1, .4, .8, .7]), np.array([.6, .2, .9])]
    for m in METHODS:
        for s in (0, 1, 2):
            data[m][s] = {v: (np.arange(len(y)), y, x + s * 100)
                          for v, y, x in zip(('01', '02'), labels, scores)}
    data['dinov3_L_P3'][2]['02'] = (np.arange(3), labels[1], scores[1][::-1])
    result = matched_statistics(data, count=30, seed=7)
    draws = np.random.default_rng(7).choice(['01', '02'], (30, 2), replace=True)
    def calc(m, chosen):
        values = []
        for s in (0, 1, 2):
            y = np.concatenate([data[m][s][v][1] for v in chosen])
            x = np.concatenate([data[m][s][v][2] for v in chosen])
            values.append([roc_auc_score(y, x), average_precision_score(y, x)])
        return np.mean(values, axis=0)
    direct = np.array([calc('dinov3_L_P3', d) - calc('native_B0', d) for d in draws])
    pair = next(r for r in result['paired_deltas']
                if r['candidate'] == 'dinov3_L_P3' and r['reference'] == 'native_B0')
    low, high = np.quantile(direct, [.025, .975], axis=0)
    np.testing.assert_allclose([pair['auroc_delta_ci_low'], pair['ap_delta_ci_low']], low)
    np.testing.assert_allclose([pair['auroc_delta_ci_high'], pair['ap_delta_ci_high']], high)


@pytest.mark.parametrize('fault', ['missing_method', 'extra_method', 'missing_seed', 'extra_seed',
                                  'frame', 'gt', 'video_order', 'video_inventory'])
def test_unmatched_evidence_rejected(fault):
    data = methods()
    if fault == 'missing_method': data.pop('native_B1')
    if fault == 'extra_method': data['other'] = copy.deepcopy(data['native_B0'])
    if fault == 'missing_seed': data['vjepa21_L_P3'].pop(2)
    if fault == 'extra_seed': data['vjepa21_L_P3'][3] = run()
    if fault == 'frame': data['vjepa21_L_P3'][2]['01'][0][0] += 1
    if fault == 'gt': data['vjepa21_L_P3'][2]['01'][1][0] = 1
    if fault == 'video_order': data['vjepa21_L_P3'][2] = dict(reversed(list(run().items())))
    if fault == 'video_inventory': data['vjepa21_L_P3'][2].pop('02')
    with pytest.raises(ValueError): matched_statistics(data, count=10)


def test_degenerate_video_bootstrap_rejected():
    data = methods()
    for runs in data.values():
        for seed in runs:
            runs[seed] = {v: (np.arange(4), np.zeros(4), np.zeros(4)) for v in ('01', '02')}
    with pytest.raises(ValueError, match='degenerate'): matched_statistics(data, count=10)


def test_changed_source_pin_rejected(tmp_path):
    import hashlib
    source = tmp_path / 'source.json'
    source.write_text('complete')
    pins = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    check_sources(pins)
    source.write_text('changed')
    with pytest.raises(ValueError, match='source changed'): check_sources(pins)
