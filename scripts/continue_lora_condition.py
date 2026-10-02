"""Continue one primary full-20-epoch LoRA condition through independent evaluation.

The full 48-condition plan remains unchanged. This command completes one member
so GPU accuracy work can alternate with the held runtime measurement controller.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ipad_jepa.cache_retirement import atomic_record
from ipad_jepa.teacher_ablation import expected_cache_bytes
from run_runtime_matrix import digest, is_live, process_identity
from retire_lora_cache import check_measurement_gate
from summarize_retained_lora_matrix import route_condition, treatment_paths

MODELS = ('dinov3-l', 'vjepa21-l')
MODES = ('offline', 'online')
DEVICES = ('R01', 'R02', 'R03', 'R04')
SEEDS = (0, 1, 2)
WEIGHTS = {'dinov3-l': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
           'vjepa21-l': 'vjepa2_1_vitl_dist_vitG_384.pt'}


def training_action(training, condition, resume):
    """The original trainer validates full checkpoint protocol and RNG on resume."""
    path = training / 'lora_training.json'
    if not path.exists():
        if training.exists() and any(training.iterdir()):
            raise ValueError('Unidentified training output requires inspection')
        return 'fresh'
    meta = json.loads(path.read_text())
    expected = dict(zip(('backbone', 'mode', 'device', 'seed'), condition))
    expected.update(epochs=20, teacher_weight=1., accumulation=8, max_steps=None)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('Primary full20 training identity or protocol differs')
    if meta.get('status') == 'complete_training' and meta.get('completed_epochs') == 20:
        return 'selected_complete'
    if (meta.get('status') != 'running' or not resume
            or not (training / 'resume.pt').is_file()):
        raise ValueError('Incomplete training requires explicit resume and a full checkpoint')
    return 'resume'


def check_previous_owner(path, condition, resume):
    if not path.exists():
        return
    old = json.loads(path.read_text())
    if old.get('condition') != list(condition):
        raise ValueError('Existing pipeline ledger belongs to another condition')
    if is_live(old.get('owner')):
        raise ValueError('Previous pipeline owner is still live')
    if not resume:
        raise ValueError('Existing pipeline ledger requires explicit resume')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--device', choices=DEVICES, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--disk-reserve-gib', type=float, default=64.)
    args = parser.parse_args()
    if not 0 <= args.disk_reserve_gib < 1024:
        raise ValueError('Invalid disk reserve')
    condition = (args.model, args.mode, args.device, args.seed)
    relative = Path(args.model) / args.mode / args.device / f'seed{args.seed}'
    ledger = args.ledger or Path('artifacts/tmp') / ('lora_condition_' + '_'.join(map(str, condition)) + '.json')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with (ledger.parent / 'isolated_accuracy.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_previous_owner(ledger, condition, args.resume)
        check_measurement_gate()
        paths = treatment_paths(condition, 1.)
        action = training_action(paths['training'], condition, args.resume)
        weights = Path('artifacts/weights') / WEIGHTS[args.model]
        upstream = Path('third_party') / ('dinov3' if args.model == 'dinov3-l' else 'vjepa2')
        source_paths = [Path(__file__).resolve().relative_to(Path.cwd().resolve()), args.manifest,
                        Path('configs/experiment_matrix.yaml'), Path('scripts/retire_lora_cache.py'),
                        Path('scripts/summarize_retained_lora_matrix.py'), Path('scripts/audit_retained_lora.py'),
                        Path('scripts/summarize_lora_matrix.py'), Path('scripts/plot_lora_evaluation.py'),
                        Path('scripts/summarize_clip_ablation.py'), Path('scripts/summarize_experiments.py')]
        source_paths.extend(Path('src/ipad_jepa') / (name + '.py') for name in
                            ['train_lora', 'adaptation', 'adapted_features', 'backbones', 'features', 'temporal',
                             'audit', 'cache_data', 'train_phase', 'experiment', 'memory', 'torch_memory',
                             'scoring', 'alignment', 'lora_audit', 'cache_retention', 'cache_retirement', 'teacher_ablation'])
        pins = {str(path): digest(path) for path in source_paths}
        rows = [row for row in json.loads(args.manifest.read_text())['sequences'] if row['device'] == args.device]
        required = expected_cache_bytes(rows, args.mode) + int(args.disk_reserve_gib * 1024 ** 3)
        if shutil.disk_usage(Path.cwd()).free < required:
            raise ValueError('Insufficient free space for a complete adapted cache and reserve')
        state = {'status': 'running', 'owner': process_identity(os.getpid()), 'condition': list(condition),
                 'teacher_weight': 1., 'planned_primary_conditions': 48,
                 'full_primary_plan': [[model, mode, device, seed] for model in MODELS for mode in MODES
                                       for device in DEVICES for seed in SEEDS],
                 'training_action': action, 'source_sha256': pins, 'child': None, 'completed_steps': [],
                 'scope': 'One unchanged full20 normal-only primary LoRA condition through selected reencoding, new PCA/memory/calibration, all actual test videos and independent audit. Full48 and runtime remain separate requirements.'}
        atomic_record(ledger, state)

        def check_sources():
            if any(digest(path) != expected for path, expected in pins.items()):
                raise ValueError('Active accuracy pipeline source changed')

        def run_child(stage, command):
            check_sources()
            check_measurement_gate()
            state.update(stage=stage, command=[sys.executable, *command], child=None)
            atomic_record(ledger, state)
            child = subprocess.Popen([sys.executable, *command])
            state['child'] = process_identity(child.pid)
            atomic_record(ledger, state)
            exit_code = child.wait()
            state.update(child=None, actual_child_exit_code=exit_code)
            atomic_record(ledger, state)
            if exit_code != 0:
                raise subprocess.CalledProcessError(exit_code, [sys.executable, *command])
            check_sources()
            state['completed_steps'].append(stage)
            atomic_record(ledger, state)

        try:
            if action != 'selected_complete':
                command = ['-m', 'ipad_jepa.train_lora', '--data-root', str(args.data_root),
                           '--manifest', str(args.manifest), '--teacher-cache', f'artifacts/features/{args.model}/{args.mode}',
                           '--model', args.model, '--mode', args.mode, '--device', args.device, '--seed', str(args.seed),
                           '--weights', str(weights), '--upstream', str(upstream), '--out', str(paths['training']),
                           '--epochs', '20', '--accumulation', '8', '--teacher-weight', '1.0']
                if action == 'resume':
                    command.append('--resume')
                run_child('full20_normal_only_training', command)
            if training_action(paths['training'], condition, True) != 'selected_complete':
                raise ValueError('Trainer exited without full20 completion')
            metrics = paths['public'] / 'metrics.json'
            if not metrics.exists():
                run_child('selected_feature_reencoding', ['-m', 'ipad_jepa.adapted_features',
                          '--run', str(paths['training']), '--weights', str(weights), '--upstream', str(upstream),
                          '--data-root', str(args.data_root), '--manifest', str(args.manifest),
                          '--cache', str(paths['cache']), '--phase-out', str(paths['local'])])
                paths['public'].mkdir(parents=True, exist_ok=True)
                for name in ('lora_training.json', 'lora_training.csv'):
                    shutil.copyfile(paths['training'] / name, paths['public'] / name)
                shutil.copyfile(paths['local'] / 'phase_training.json', paths['public'] / 'phase_training.json')
                run_child('new_memory_calibration_all_test_evaluation', ['-m', 'ipad_jepa.experiment',
                          '--cache', str(paths['cache']), '--manifest', str(args.manifest),
                          '--phase-run', str(paths['local']), '--data-root', str(args.data_root),
                          '--device', args.device, '--seed', str(args.seed), '--out', str(paths['public']),
                          '--local', str(paths['local'])])
            check_measurement_gate()
            state.update(stage='independent_actual_condition_audit'); atomic_record(ledger, state)
            out = Path('results/stage04/condition_pipeline_checks')
            _, audit, _ = route_condition(condition, 1., args.manifest, args.data_root, out)
            check_sources()
            proof = {'status': 'complete_independently_audited_primary_lora_condition',
                     'condition': list(condition), 'teacher_weight': 1., 'training_action': action,
                     'source_sha256': pins, 'condition_audit': audit,
                     'metrics_sha256': digest(metrics), 'normal_fit_sha256': digest(paths['public'] / 'normal_fit.json'),
                     'scope': state['scope']}
            proof_path = out / relative / 'pipeline_completion.json'
            atomic_record(proof_path, proof)
            state.update(status=proof['status'], completion_proof=str(proof_path),
                         completion_proof_sha256=digest(proof_path), stage='complete')
            atomic_record(ledger, state)
            print(json.dumps(proof, indent=2), flush=True)
        except BaseException as error:
            state.update(status='failed', error=repr(error)); atomic_record(ledger, state); raise


if __name__ == '__main__':
    main()
