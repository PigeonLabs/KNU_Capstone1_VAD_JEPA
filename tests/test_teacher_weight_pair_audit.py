"""Reject unpaired/frozen replacements and permit independently learned selections."""
import copy
import json
import sys
from pathlib import Path
import pytest
from ipad_jepa.teacher_weight_audit import check_pair_controls

CONDITION = ('vjepa21-l', 'offline', 'R01', 1)


def pair():
    base = {'status': 'complete_training', 'teacher_weight': 0., 'epochs': 20,
            'completed_epochs': 20, 'micro_batch': 1, 'accumulation': 8,
            'rank': 8, 'alpha': 16, 'dropout': .05, 'blocks': 4,
            'encoder_lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': 1e-4,
            'precision': 'bf16', 'torch': '2.8.0+cu128', 'max_steps': None,
            'encoder_trainable_parameters': 131072, 'head_trainable_parameters': 313800,
            'backbone': 'vjepa21-l', 'mode': 'offline', 'device': 'R01', 'seed': 1,
            'fit_clips': 9, 'calibration_clips': 4, 'updates': 40,
            'selected_epoch': 3, 'best_normal_calibration_ce': 2.,
            'teacher_identity': {'backbone': 'vjepa21-l', 'mode': 'offline',
                                 'weights_sha256': 'base', 'adapter_sha256': 'frozen',
                                 'fit_stride': 4},
            'teacher_fingerprints': ['fit', 'cal'], 'trainer_sha256': 'trainer',
            'adaptation_sha256': 'adaptation', 'dense_loss': 'normalized dense L2',
            'selection': 'minimum normal CE', 'training': 'joint normal LoRA'}
    curve = [{'epoch': i+1, 'train_ce': 1., 'train_dense_l2': .4,
              'normal_calibration_ce': 2.+abs(i-2), 'normal_calibration_circular_mae': .1,
              'clips': 9, 'updates': (i+1)*2} for i in range(20)]
    normal = {'status': 'normal_fit_and_calibration_complete',
              'backbone': 'vjepa21-l', 'mode': 'offline', 'device': 'R01', 'seed': 1,
              'cycle_length_fit_median': 160., 'bins': 16, 'prototypes_per_bin': 128,
              'total_prototypes': 2048, 'pca_dimensions': 256, 'pca_samples': 50000,
              'temperature_samples': 50000, 'calibration_valid_frames': 100,
              'code_sha256': 'experiment', 'memory_code_sha256': 'memory',
              'phase_selected_epoch': 3, 'cache_identity': {**base['teacher_identity'],
                                                         'adapter_sha256': 'selected-zero'}}
    one = copy.deepcopy(base); one['teacher_weight'] = 1.
    one_normal = copy.deepcopy(normal); one_normal['cache_identity']['adapter_sha256'] = 'selected-one'
    return base, one, curve, copy.deepcopy(curve), normal, one_normal


def test_independent_normal_ce_selection_and_learned_calibration_are_allowed():
    zero, one, zcurve, ocurve, znorm, onorm = pair()
    one['selected_epoch'] = onorm['phase_selected_epoch'] = 4
    for i, row in enumerate(ocurve): row['normal_calibration_ce'] = 2.+abs(i-3)
    znorm['temperature'], onorm['temperature'] = .2, .9
    proof = check_pair_controls(zero, one, zcurve, ocurve, znorm, onorm, CONDITION)
    assert proof['selected_epochs'] == [3, 4]
    assert proof['optimizer_updates_per_treatment'] == 40


@pytest.mark.parametrize('field,value', [
    ('teacher_identity', {'weights_sha256': 'another-model'}),
    ('teacher_fingerprints', ['different-fit', 'cal']), ('fit_clips', 8),
    ('calibration_clips', 3), ('trainer_sha256', 'other-trainer'),
    ('head_trainable_parameters', 0), ('updates', 39)])
def test_same_named_condition_cannot_hide_changed_teacher_data_or_optimizer(field, value):
    values = list(pair()); values[1][field] = value
    with pytest.raises(ValueError): check_pair_controls(*values, CONDITION)


def test_teacher_one_cannot_be_replaced_with_zero_weight_or_frozen_encoder():
    values = list(pair()); values[1]['teacher_weight'] = 0.
    with pytest.raises(ValueError): check_pair_controls(*values, CONDITION)
    values = list(pair()); values[1]['encoder_trainable_parameters'] = 0
    with pytest.raises(ValueError): check_pair_controls(*values, CONDITION)


def test_missing_or_changed_control_field_is_rejected():
    values = list(pair()); del values[1]['teacher_identity']
    with pytest.raises(ValueError): check_pair_controls(*values, CONDITION)
    values = list(pair()); values[5]['total_prototypes'] = 4096
    with pytest.raises(ValueError): check_pair_controls(*values, CONDITION)
    values = list(pair()); values[5]['cache_identity']['fit_stride'] = 1
    with pytest.raises(ValueError): check_pair_controls(*values, CONDITION)


def test_full_final_update_count_does_not_hide_incomplete_epoch_update_curve():
    values = list(pair()); values[3][10]['updates'] -= 1
    with pytest.raises(ValueError, match='update curve'): check_pair_controls(*values, CONDITION)


def test_no_completed_teacher_pairs_produces_no_validation_or_summary(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from summarize_teacher_weight import main
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps({'sequences': []}))
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['summarize_teacher_weight.py', '--manifest', str(manifest),
                                    '--zero-root', str(tmp_path/'T0'), '--one-root', str(tmp_path/'T1'),
                                    '--out', str(out)])
    with pytest.raises(ValueError, match='Actual completed teacher 0/1 pairs'): main()
    assert not out.exists()
