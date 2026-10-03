"""Check preserved LoRA runtime inputs after an explicit cache retirement, on CPU only.

This checks readiness, selected adapter/head binding and fitted bank geometry.
It does not execute an encoder/search, measure runtime, or authorize retirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_runtime_matrix import readiness
from ipad_jepa.adapted_features import selected_protocol
from ipad_jepa.runtime_adaptation import selected_runtime_adapter


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(model, mode, device, seed):
    if torch.cuda.is_initialized():
        raise ValueError('Use a fresh CPU-only process for this check')
    torch.set_num_threads(2)
    relative = Path(model) / mode / device / f'seed{seed}'
    cache = Path('artifacts/features_lora') / relative
    training = Path('artifacts/lora') / relative
    local = Path('artifacts/runs_lora') / relative
    public = Path('results/stage04') / relative
    release_path = Path('results/cache_retention/T1') / relative / 'retirement.json'
    release = read(release_path)
    if release['status'] != 'complete_verified_derived_payload_retirement':
        raise ValueError('Actual completed retirement journal required')
    planned = release['planned_files']
    if not planned or any((cache / item['relative']).exists() for item in planned):
        raise ValueError('Retired payload was recreated or journal is empty')
    if list(cache.rglob('patch.npy')) or list(cache.rglob('global.npy')):
        raise ValueError('This check requires the actual absent derived payloads')
    targets, metadata = list(cache.rglob('targets.npy')), list(cache.rglob('meta.json'))
    if not targets or len(targets) != len(metadata) or len(planned) != 2 * len(targets):
        raise ValueError('Retained metadata/targets differ from payload inventory')
    condition = dict(adaptation='lora', backbone=model, mode=mode, device=device, seed=seed)
    ready = readiness(condition)
    if not ready['ready']:
        raise ValueError('Preserved runtime inputs are not ready: ' + str(ready))
    normal, phase = read(public / 'normal_fit.json'), read(local / 'phase_training.json')
    meta, checkpoint = selected_protocol(training)
    state = torch.load(local / 'phase_head.pt', map_location='cpu', weights_only=True)
    if (normal['phase_selected_epoch'] != phase['selected_epoch']
            or phase['selected_epoch'] != meta['selected_epoch']
            or digest(local / 'phase_head.pt') != normal['phase_checkpoint_sha256']
            or sorted(normal['normal_cache_fingerprints']) != sorted(phase['cache_fingerprints'])):
        raise ValueError('Preserved head/normal calibration provenance differs')
    args = SimpleNamespace(adaptation='lora', lora_run=training,
                           model=model, mode=mode, device=device, seed=seed)
    adapter, binding = selected_runtime_adapter(
        args, normal['cache_identity'], phase, state, meta['teacher_identity'])
    if (adapter.keys() != checkpoint['adapter'].keys()
            or any(not torch.equal(adapter[k], checkpoint['adapter'][k])
                   or adapter[k].device.type != 'cpu'
                   or not torch.isfinite(adapter[k]).all() for k in adapter)
            or any(t.device.type != 'cpu' or not torch.isfinite(t).all() for t in state.values())):
        raise ValueError('Selected adapter/head CPU tensors differ or are nonfinite')
    with np.load(local / 'memory.npz', allow_pickle=False) as bank:
        shapes = {k: list(bank[k].shape) for k in ('mean', 'components', 'prototypes')}
        if (shapes != {'mean': [1024], 'components': [256, 1024], 'prototypes': [16, 128, 256]}
                or not all(np.isfinite(bank[k]).all() for k in shapes)
                or float(bank['temperature']) != normal['temperature']
                or float(bank['cycle_length']) != normal['cycle_length_fit_median']):
            raise ValueError('Preserved fitted PCA/prototype bank differs')
    if torch.cuda.is_initialized():
        raise ValueError('CPU check unexpectedly initialized CUDA')
    paths = [release_path, training / 'lora_training.json', training / 'lora_training.csv',
             training / 'selected_adapter.pt', local / 'phase_head.pt',
             local / 'phase_training.json', local / 'memory.npz', public / 'normal_fit.json',
             Path('src/ipad_jepa/runtime_adaptation.py'), Path('src/ipad_jepa/adapted_features.py'),
             Path('src/ipad_jepa/train_lora.py'), Path('src/ipad_jepa/adaptation.py'),
             Path('scripts/run_runtime_matrix.py'), Path('scripts/benchmark_runtime.py'),
             Path(__file__).resolve().relative_to(Path.cwd().resolve())]
    return dict(status='passed_retired_primary_runtime_inputs_CPU_only', condition=condition,
                actual_absent_derived_payload_files=len(planned), retained_sequence_meta_and_targets=len(targets),
                original_runtime_readiness=ready, original_selected_runtime_binding=binding,
                selected_epoch=meta['selected_epoch'], CPU_adapter_tensor_count=len(adapter),
                CPU_head_tensor_count=len(state), fitted_memory_shapes=shapes,
                CUDA_context_initialized=False, source_sha256={str(p): digest(p) for p in paths},
                limits='Actual CPU readiness and selected adapter/head binding, fitted bank shape/finite check. '
                       'No GPU encoder/search/parity/FPS/latency, new full runtime result, base-weight/raw JPEG '
                       'full-content rehash, encoder regeneration, broader cache retirement eligibility or permission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('dinov3-l', 'vjepa21-l'), required=True)
    parser.add_argument('--mode', choices=('offline', 'online'), required=True)
    parser.add_argument('--device', choices=('R01', 'R02', 'R03', 'R04'), required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh check output; preserve previous evidence')
    report = check(args.model, args.mode, args.device, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('source_sha256', 'original_runtime_readiness')}, indent=2))


if __name__ == '__main__':
    main()
