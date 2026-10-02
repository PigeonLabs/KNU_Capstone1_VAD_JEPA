import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from lora_reporting import summarize
from summarize_retained_lora_matrix import route_condition
from plot_retained_lora import verify_mixed_sources


def runs(condition_count=3, teacher=False):
    result = {}
    for seed in range(condition_count):
        labels = np.array([0, 1, 0, 1])
        frames = np.arange(4)
        base_scores = [labels.astype(float), 100. + labels, -1. - labels][seed]
        base = {video: (frames, labels, base_scores) for video in ['01', '02']}
        improved = {video: (frames, labels, labels.astype(float)) for video in ['01', '02']}
        first, second = ('T1', 'T0') if teacher else ('F', 'L')
        result['dinov3-l', 'offline', 'R01', seed] = {
            first: {variant: copy.deepcopy(base) for variant in ['P0', 'P1', 'P2', 'P3']},
            second: {variant: copy.deepcopy(improved) for variant in ['P0', 'P1', 'P2', 'P3']}}
    return result


def test_one_or_two_seeds_do_not_form_rows_or_macro4():
    for count in [1, 2]:
        device, macro, groups = summarize(runs(count), 'lora')
        assert device['results'] == [] and macro['results'] == [] and groups == []


def test_retained_statistics_average_seed_metrics_and_do_not_pool_shifted_scores():
    inputs = runs()
    device, macro, groups = summarize(inputs, 'lora')
    assert len(groups) == 1 and len(device['results']) == 8 and macro['results'] == []
    row = next(r for r in device['results'] if r['variant'] == 'F_P3')
    # Analytical per-seed AUROC = 1,1,0; AP = 1,1,1/2.
    assert row['auroc_mean'] == pytest.approx(2 / 3)
    assert row['ap_mean'] == pytest.approx(5 / 6)
    labels, scores = [], []
    for values in inputs.values():
        for _, gt, score in values['F']['P3'].values():
            labels.extend(gt); scores.extend(score)
    assert not np.isclose(row['auroc_mean'], roc_auc_score(labels, scores))
    assert not np.isclose(row['ap_mean'], average_precision_score(labels, scores))
    pair = next(r for r in device['paired_deltas'] if r['comparison'] == 'LoRA_minus_frozen_P3')
    assert pair['auroc_delta'] == pytest.approx(1 / 3)
    assert pair['ap_delta'] == pytest.approx(1 / 6)
    assert pair['auroc_delta_ci_low'] == pytest.approx(pair['auroc_delta'])


def test_actual_teacher_difference_is_zero_minus_one_and_not_frozen():
    device, _, _ = summarize(runs(teacher=True), 'teacher')
    assert {r['variant'].split('_')[0] for r in device['results']} == {'T0', 'T1'}
    pair = next(r for r in device['paired_deltas'] if r['comparison'] == 'T0_minus_T1_P3')
    assert pair['auroc_delta'] == pytest.approx(1 / 3)
    assert pair['ap_delta'] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match='Both actual treatments'):
        summarize(runs(), 'teacher')


def test_mismatched_video_targets_gt_or_missing_variants_fail_before_statistics():
    for change in ['frame', 'GT', 'variant']:
        inputs = runs()
        value = inputs['dinov3-l', 'offline', 'R01', 1]['L']['P3']
        if change == 'frame':
            value['01'] = (np.arange(4) + 1, value['01'][1], value['01'][2])
        elif change == 'GT':
            value['01'] = (value['01'][0], 1 - value['01'][1], value['01'][2])
        else:
            inputs['dinov3-l', 'offline', 'R01', 1]['L'].pop('P0')
        with pytest.raises((ValueError, AssertionError)):
            summarize(inputs, 'lora')


def test_macro4_requires_all_four_devices_and_keeps_equal_device_weights():
    inputs = runs()
    for device in ['R02', 'R03', 'R04']:
        for key, value in runs().items():
            new = copy.deepcopy(value)
            if device != 'R01':
                for variant in new['F']:
                    for video, (frames, labels, scores) in new['F'][variant].items():
                        # Perfect fixed backbone on three devices, with more frames on R04.
                        factor = 10 if device == 'R04' else 1
                        new['F'][variant][video] = (np.arange(4 * factor), np.tile(labels, factor), np.tile(labels, factor).astype(float))
                    for video, (frames, labels, _) in new['F'][variant].items():
                        new['L'][variant][video] = (frames.copy(), labels.copy(), labels.astype(float))
            inputs[key[0], key[1], device, key[3]] = new
    device, macro, groups = summarize(inputs, 'lora')
    assert len(groups) == 4
    row = next(r for r in macro['results'] if r['variant'] == 'F_P3')
    assert row['devices'] == 4 and row['seeds'] == 3
    assert row['auroc_mean'] == pytest.approx((2 / 3 + 3) / 4)
    inputs = {k: v for k, v in inputs.items() if k[2] != 'R04'}
    _, macro, _ = summarize(inputs, 'lora')
    assert macro['results'] == []


def test_incomplete_retirement_cannot_fall_back_to_array_audit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = Path('results/cache_retention/T1/dinov3-l/offline/R01/seed0')
    folder.mkdir(parents=True)
    (folder / 'retirement.json').write_text(json.dumps({'status': 'retiring_verified_derived_payloads'}))
    with pytest.raises(ValueError, match='incomplete'):
        route_condition(('dinov3-l', 'offline', 'R01', 0), 1., Path('missing'), Path('missing'), tmp_path / 'out')


def test_missing_retirement_requires_actual_current_array_audit(tmp_path, monkeypatch):
    import summarize_retained_lora_matrix as router
    called = []
    def strict_current(*args):
        called.append(True)
        raise ValueError('Missing actual derived arrays')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router, 'audit_lora', strict_current)
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{"sequences":[]}')
    with pytest.raises(ValueError, match='Missing actual derived arrays'):
        route_condition(('dinov3-l', 'offline', 'R01', 0), 1., manifest, tmp_path, tmp_path / 'out')
    assert called == [True]


def test_mixed_plot_rejects_original_validation_and_wrong_treatment_substitution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(kind='teacher', manifest=tmp_path/'manifest.json', root=tmp_path)
    args.manifest.write_text('{"sequences":[]}')
    from plot_retained_lora import digest
    with pytest.raises(ValueError, match='matching mixed'):
        verify_mixed_sources(args, {'status': 'passed', 'kind': 'teacher'})
    proof = {'status': 'passed_mixed_current_retained_evidence', 'kind': 'teacher',
             'manifest_sha256': digest(args.manifest), 'source_sha256': {},
             'condition_checks': [{'condition': ['dinov3-l', 'offline', 'R01', 0],
                                   'treatments': [{'teacher_weight': 1.}]}],
             'completed_conditions_or_pairs': 1, 'complete_three_seed_groups': 0,
             'matrix_complete': False, 'normal_thresholds_replayed': 8}
    (tmp_path / 'artifacts/tmp').mkdir(parents=True)
    with pytest.raises(ValueError, match='actual teacher treatments'):
        verify_mixed_sources(args, proof)
