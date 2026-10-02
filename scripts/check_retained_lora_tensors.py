"""Exercise retained tensor replay on fresh current-array audits before any cache removal."""
import argparse
import json
from pathlib import Path

from audit_retained_lora import replay_tensors
from ipad_jepa.cache_retention import digest
from summarize_retained_lora_matrix import treatment_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, default=Path('results/stage04/retained_matrix_audit/validation.json'))
    parser.add_argument('--out', type=Path, default=Path('results/setup/retained_lora_tensor_preflight.json'))
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    if (validation['status'] != 'passed_mixed_current_retained_evidence'
            or validation['kind'] != 'lora' or validation['retained_treatments_replayed'] != 0):
        raise ValueError('Fresh current-array primary audits required for this pre-removal probe')
    pins = dict(validation['source_sha256'])
    pins[str(args.validation)] = digest(args.validation)
    pins['scripts/check_retained_lora_tensors.py'] = digest(__file__)
    pins['src/ipad_jepa/adapted_features.py'] = digest('src/ipad_jepa/adapted_features.py')
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('Fresh current audit source changed')
    checks = []
    for row in validation['condition_checks']:
        condition = tuple(row['condition'])
        if len(row['treatments']) != 1:
            raise ValueError('Primary teacher=1 treatment required')
        treatment = row['treatments'][0]
        if treatment['teacher_weight'] != 1. or treatment['evidence'] != 'current_actual_adapted_arrays':
            raise ValueError('Actual current-array treatment required')
        witness_path = args.validation.parent / treatment['condition_audit']
        if digest(witness_path) != treatment['condition_audit_sha256']:
            raise ValueError('Original current-array condition witness changed')
        pins[str(witness_path)] = digest(witness_path)
        witness = json.loads(witness_path.read_text())
        paths = treatment_paths(condition, 1.)
        for name, sha in witness['public_sources_sha256'].items():
            path = paths['public'] / name
            if digest(path) != sha:
                raise ValueError('Audited public result changed')
            pins[str(path)] = sha
        relative = Path(condition[0]) / condition[1] / condition[2] / f'seed{condition[3]}'
        normal = json.loads((paths['public'] / 'normal_fit.json').read_text())
        meta = replay_tensors(paths['training'], paths['local'], Path('results/stage02') / relative,
                              Path('artifacts/runs') / relative, paths['public'], witness, normal, condition, 1.)
        checks.append({'condition': list(condition), 'selected_epoch': meta['selected_epoch'],
                       'selected_adapter_sha256': digest(paths['training'] / 'selected_adapter.pt'),
                       'joint_head_sha256': digest(paths['local'] / 'phase_head.pt'),
                       'rebuilt_memory_sha256': digest(paths['local'] / 'memory.npz'),
                       'original_current_audit_sha256': digest(witness_path)})
    if not checks or len(checks) != validation['completed_conditions_or_pairs']:
        raise ValueError('Actual completed current audit inventory differs')
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('Probe inputs changed during selected tensor replay')
    result = {'status': 'passed_actual_retained_tensor_preflight_on_current_conditions',
              'completed_conditions_checked': len(checks), 'conditions': checks, 'source_sha256': pins,
              'cache_removal_performed': False,
              'scope': 'Actual selected adapter/joint head/bank and training/normal teacher provenance replay '
                       'through the retained helper while full current-array witnesses still exist. '
                       'No archived-release/post-removal audit, encoder rerun or whole-matrix completion claim.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
