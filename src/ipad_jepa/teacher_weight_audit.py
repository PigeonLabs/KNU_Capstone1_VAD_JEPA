"""Matched controls for teacher-zero versus teacher-one joint LoRA treatments."""
from .lora_audit import check_training


TRAINING_CONTROLS = (
    'backbone', 'mode', 'device', 'seed', 'epochs', 'completed_epochs',
    'micro_batch', 'accumulation', 'rank', 'alpha', 'dropout', 'blocks',
    'encoder_lr', 'head_lr', 'weight_decay', 'precision', 'torch', 'max_steps',
    'teacher_identity', 'teacher_fingerprints', 'trainer_sha256',
    'adaptation_sha256', 'dense_loss', 'selection', 'training', 'fit_clips',
    'calibration_clips', 'encoder_trainable_parameters',
    'head_trainable_parameters', 'updates',
)
MEMORY_CONTROLS = (
    'backbone', 'mode', 'device', 'seed', 'cycle_length_fit_median',
    'bins', 'prototypes_per_bin', 'total_prototypes', 'pca_dimensions',
    'pca_samples', 'temperature_samples', 'calibration_valid_frames',
    'code_sha256', 'memory_code_sha256',
)


def check_pair_controls(zero, one, zero_curve, one_curve, zero_normal,
                        one_normal, condition):
    """Weights and learned selections may differ; inputs/optimization must match."""
    check_training(zero, zero_curve, condition, 0.)
    check_training(one, one_curve, condition, 1.)
    for field in TRAINING_CONTROLS:
        if field not in zero or field not in one or zero[field] != one[field]:
            raise ValueError('Teacher 0/1 training controls differ: ' + field)
    expected_updates = ((zero['fit_clips'] + zero['accumulation'] - 1)
                        // zero['accumulation']) * 20
    if zero['updates'] != expected_updates:
        raise ValueError('Teacher pair lacks complete optimizer updates')
    per_epoch = expected_updates // 20
    for curve in (zero_curve, one_curve):
        if any(int(row.get('updates', -1)) != i * per_epoch
               for i, row in enumerate(curve, 1)):
            raise ValueError('Teacher pair optimizer update curve differs')
    for field in MEMORY_CONTROLS:
        if (field not in zero_normal or field not in one_normal
                or zero_normal[field] != one_normal[field]):
            raise ValueError('Teacher 0/1 memory controls differ: ' + field)
    for meta, normal in ((zero, zero_normal), (one, one_normal)):
        if normal.get('status') != 'normal_fit_and_calibration_complete':
            raise ValueError('Teacher pair memory is incomplete')
        if normal['phase_selected_epoch'] != meta['selected_epoch']:
            raise ValueError('Teacher memory used another selected epoch')
        base = {k: v for k, v in normal['cache_identity'].items()
                if k != 'adapter_sha256'}
        teacher = {k: v for k, v in meta['teacher_identity'].items()
                   if k != 'adapter_sha256'}
        if base != teacher:
            raise ValueError('Teacher pair changed base input/encoder identity')
    return {'status': 'passed_matched_teacher_zero_one_controls',
            'training_fields_checked': list(TRAINING_CONTROLS),
            'memory_fields_checked': list(MEMORY_CONTROLS),
            'optimizer_updates_per_treatment': expected_updates,
            'selected_epochs': [zero['selected_epoch'], one['selected_epoch']],
            'scope': 'Same condition, teacher base, normal inputs, optimizer and '
                     '20-epoch CE selection rule. Learned epochs, features, '
                     'heads, PCA/banks, temperatures and thresholds can differ.'}
