"""Untimed GPU capture of the failed DINO FP32 contexts; never a parity repair.

Run full and reuse as separate processes after accuracy work has released its
lock. Arrays stay in artifacts/. This probe does not overwrite measured traces,
relax their gates, update the bank, or measure throughput.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmark_runtime import fixed_components, require_isolated_gpu, score_features
from ipad_jepa.audit import inspect_frames
from ipad_jepa.cache_retirement import atomic_record
from ipad_jepa.features import ClipDataset, file_hash as _file_hash
from ipad_jepa.streaming import DinoFrameRing
from retire_lora_cache import check_measurement_gate
from verify_runtime_parity import compare, GATES


PAIR = Path('results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/fp32')
SOURCES = Path('results/setup/runtime_controller_source_check.json')
CONDITION = {'backbone': 'dinov3-l', 'mode': 'online', 'device': 'R01', 'seed': 0,
             'precision': 'fp32', 'adaptation': 'frozen', 'all_test_videos': True}


def file_hash(path):
    return _file_hash(Path(path))


def search_capture(scorer, features, phase):
    """Expose exact fast-search IDs, then independently recompute distances in FP64.

    The FP64 calculation promotes the *original FP32 queries and prototypes*.
    It is a diagnostic of distance rounding, not FP64 encoder/PCA inference.
    """
    if (features.ndim != 3 or features.dtype != torch.float32 or phase.dtype != torch.float32
            or phase.shape != (len(features),) or not features.isfinite().all()
            or not phase.isfinite().all() or torch.any((phase < 0) | (phase >= 1))
            or scorer.bins < 3):
        raise ValueError('Finite FP32 features/phases and at least three bins required')
    z = scorer.transform(features)
    center = (phase * scorer.bins).long()
    bins = torch.stack([(center - 1) % scorer.bins, center, (center + 1) % scorer.bins], 1).sort(1).values
    bank = scorer.prototypes[bins].reshape(len(z), -1, scorer.dimensions)
    if bank.shape[1] < 6:
        raise ValueError('At least six candidates needed for the k=5 boundary diagnostic')
    ids = (bins[:, :, None] * scorer.prototypes.shape[1]
           + torch.arange(scorer.prototypes.shape[1], device=z.device)).reshape(len(z), -1)
    fast = (z.square().sum(-1, keepdim=True) + bank.square().sum(-1)[:, None, :]
            - 2 * torch.bmm(z, bank.transpose(1, 2))).clamp_min(0)
    distance, local_ids = fast.topk(5, dim=-1, largest=False, sorted=True)
    batches = torch.arange(len(z), device=z.device)[:, None, None]
    neighbours = bank[batches, local_ids]
    actual_z, actual_neighbours, actual_distance = scorer.nearest(features, phase, k=5)
    if not all(torch.equal(a, b) for a, b in [(z, actual_z), (neighbours, actual_neighbours),
                                             (distance, actual_distance)]):
        raise ValueError('Captured search differs from unchanged scorer.nearest')
    weights = (-distance / max(scorer.temperature, 1e-6)).softmax(-1)
    patches = (z - (neighbours * weights[..., None]).sum(-2)).square().sum(-1)
    count = max(1, math.ceil(z.shape[1] * .05))
    top, top_ids = patches.topk(count, dim=1)
    score = top.mean(1)
    if not torch.equal(score, scorer.frame_scores(features, phase, k=5)):
        raise ValueError('Captured frame score differs from unchanged scorer.frame_scores')
    # Chunking limits diagnostic allocations without changing any fast-search arithmetic.
    stable = torch.cat([(part.double()[:, :, None] - bank.double()[:, None]).square().sum(-1)
                        for part in z.split(32, dim=1)], dim=1)
    stable_distance, stable_local = stable.topk(5, dim=-1, largest=False, sorted=True)
    stable_neighbours = bank.double()[batches, stable_local]
    stable_weights = (-stable_distance / max(scorer.temperature, 1e-6)).softmax(-1)
    stable_patches = (z.double() - (stable_neighbours * stable_weights[..., None]).sum(-2)).square().sum(-1)
    stable_top, stable_top_ids = stable_patches.topk(count, dim=1)
    fast_six = fast.topk(6, dim=-1, largest=False, sorted=True).values
    stable_six = stable.topk(6, dim=-1, largest=False, sorted=True).values
    return {'query': z, 'phase_bins': bins, 'fast_ids': ids[batches, local_ids],
            'fast_distances': distance, 'fast_patch_scores': patches, 'fast_top_patch_ids': top_ids,
            'fast_frame_score': score, 'fast_k5_k6_margin': fast_six[..., 5] - fast_six[..., 4],
            'stable_ids': ids[batches, stable_local], 'stable_distances': stable_distance,
            'stable_patch_scores': stable_patches, 'stable_top_patch_ids': stable_top_ids,
            'stable_frame_score': stable_top.mean(1),
            'stable_k5_k6_margin': stable_six[..., 5] - stable_six[..., 4],
            'distance_rounding_max_abs': (fast.double() - stable).abs().amax()}


def verify_pair_inputs(pair=PAIR, sources_path=SOURCES):
    """Bind probes to the failed measured pair and its actual source snapshot."""
    diagnostic_path = pair / 'failure_diagnostics.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    parity = compare(pair / 'full', pair / 'reuse')
    if (parity != json.loads((pair / 'parity.json').read_text()) or parity['status'] != 'failed'
            or any(parity['condition'].get(k) != v for k, v in CONDITION.items())
            or diagnostic['status'] != 'replayed_failed_pair_and_localized_trace_divergence'
            or diagnostic['parity_status'] != 'failed' or diagnostic['condition'] != parity['condition']):
        raise ValueError('Original failed DINO FP32 pair required')
    source = json.loads(sources_path.read_text())
    if source['status'] != 'passed_actual_live_runtime_controller_source_and_completed_output_check':
        raise ValueError('Actual measurement source snapshot required')
    pins = {str(diagnostic_path): file_hash(diagnostic_path), str(sources_path): file_hash(sources_path)}
    for mapping in (diagnostic['source_sha256'], source['source_sha256']):
        if any(file_hash(Path(path)) != expected for path, expected in mapping.items()):
            raise ValueError('Original measured sources or diagnosis changed')
        pins.update(mapping)
    for implementation in ('full', 'reuse'):
        meta_path = pair / implementation / 'runtime.json'
        matches = [item for item in source['completed_runs'].values()
                   if str(meta_path) in item['sources_sha256']]
        if len(matches) != 1 or matches[0]['status'] != 'completed_child_exit_zero':
            raise ValueError('Actual completed measured implementation required')
        mapping = matches[0]['sources_sha256']
        if any(file_hash(Path(path)) != expected for path, expected in mapping.items()):
            raise ValueError('Original measured output changed')
        pins.update(mapping)
    targets = [(row['sequence'], row['target_frame']) for row in diagnostic['failed_targets']]
    if len(targets) != diagnostic['raw_feature_failed_targets'] or len(targets) != len(set(targets)):
        raise ValueError('Invalid distinct failed-target inventory')
    if not targets or any(not isinstance(t, int) or t < 19 for _, t in targets):
        raise ValueError('Actual alarm-eligible failed contexts required')
    if any(file_hash(Path(path)) != expected for path, expected in pins.items()):
        raise ValueError('Original measured sources or diagnosis changed')
    return diagnostic, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=['full', 'reuse'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path, required=True, help='Fresh ignored artifacts/ directory')
    parser.add_argument('--pair', type=Path, default=PAIR)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    output = args.capture_root.resolve()
    if not output.is_relative_to(repo / 'artifacts') or output == repo / 'artifacts':
        raise ValueError('Diagnostic arrays must remain under ignored artifacts/')
    if output.exists():
        raise ValueError('Use a fresh capture directory; existing captures are never replaced')
    # Use the same exclusion lock as full accuracy conditions. No GPU allocation
    # or raw-data traversal occurs until both accuracy and runtime gates pass.
    with Path('artifacts/tmp/isolated_accuracy.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_measurement_gate()
        diagnostic, pins = verify_pair_inputs(args.pair)
        meta = json.loads((args.pair / args.implementation / 'runtime.json').read_text())
        selected = SimpleNamespace(model='dinov3-l', mode='online', device='R01', seed=0,
                                   adaptation='frozen', lora_run=None,
                                   results=Path('results/stage02/dinov3-l/online/R01/seed0'),
                                   run=Path('artifacts/runs/dinov3-l/online/R01/seed0'),
                                   weights=Path('artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
                                   upstream=Path('third_party/dinov3'))
        manifest = Path('results/stage00/manifest.json')
        for path, key in [(manifest, 'manifest_sha256'), (selected.weights, 'weights_sha256'),
                          (selected.results / 'normal_fit.json', 'normal_fit_sha256'),
                          (selected.run / 'phase_head.pt', 'phase_head_sha256'),
                          (selected.run / 'memory.npz', 'memory_sha256')]:
            if file_hash(path) != meta[key]:
                raise ValueError('Original measured model/bank/calibration/manifest changed')
            pins[str(path)] = meta[key]
        pins[str(Path(__file__).resolve().relative_to(repo))] = file_hash(__file__)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU substitute for actual encoder probes')
        if str(torch.__version__) != meta['torch'] or torch.cuda.get_device_name() != meta['gpu']:
            raise ValueError('Original GPU and torch build required for the probe')
        require_isolated_gpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = json.loads(manifest.read_text())['sequences']
        normal = next(row for row in rows if row['device'] == 'R01' and row.get('split') == 'fit')
        folder = args.data_root / normal['relative_directory']
        if inspect_frames(folder)['frames_content_sha256'] != normal['frames_content_sha256']:
            raise ValueError('Warmup normal input changed')
        model, head, scorer, _, _, _ = fixed_components(selected)
        warm = ClipDataset(folder, np.array([20]), 'online')[0][0][None].cuda()
        warm_ring = DinoFrameRing(model.encoder) if args.implementation == 'reuse' else None
        with torch.inference_mode():
            for index in range(20):
                if warm_ring is None:
                    local, pooled = model(warm)
                else:
                    warm_ring.push(warm[0, :, index % 16], index)
                    local, pooled = warm_ring.features(padding=True)
                score_features(local, pooled, head, scorer, 'fp32')
        del warm, warm_ring, local, pooled
        output.mkdir(parents=True)
        record = {'status': 'running_untimed_gpu_capture', 'implementation': args.implementation,
                  'condition': CONDITION, 'source_sha256': pins, 'captures': [],
                  'warmup_forward_calls': 20, 'original_parity_status': 'failed',
                  'limits': 'Only original failed contexts, replayed from frame zero through their final failed target. '
                            'Diagnostic FP64 distances use promoted FP32 queries/bank. No new FPS, alarm replay, '
                            'all-video equivalence, gate change, accuracy improvement or causal proof.'}
        atomic_record(output / 'capture.json', record)
        with torch.inference_mode():
            for sequence in sorted({row['sequence'] for row in diagnostic['failed_targets']}):
                require_isolated_gpu()
                row = next(row for row in rows if row['device'] == 'R01' and row['partition'] == 'testing'
                           and row['sequence'] == sequence)
                folder = args.data_root / row['relative_directory']
                if inspect_frames(folder)['frames_content_sha256'] != row['frames_content_sha256']:
                    raise ValueError('Probe raw input changed')
                reader = ClipDataset(folder, np.array([], dtype=np.int64), 'online', cache_frames=16)
                targets = {item['target_frame']: item for item in diagnostic['failed_targets']
                           if item['sequence'] == sequence}
                if len(reader.paths) != row['frames'] or max(targets) >= row['frames']:
                    raise ValueError('Failed contexts outside original frame inventory')
                images = deque(maxlen=16)
                ring = DinoFrameRing(model.encoder) if args.implementation == 'reuse' else None
                for index in range(max(targets) + 1):
                    frame = reader.frame(index)
                    images.append(frame)
                    if ring is not None:
                        ring.push(frame.cuda(), index)
                        if index < 15:
                            continue
                        local, pooled = ring.features()
                    elif index >= 15:
                        local, pooled = model(torch.stack(list(images), dim=1)[None].cuda())
                    else:
                        continue
                    phase, raw, _, _, _ = score_features(local, pooled, head, scorer, 'fp32')
                    if index not in targets:
                        continue
                    capture = search_capture(scorer, local.float(), torch.tensor([phase], device='cuda'))
                    if float(capture['fast_frame_score'].item()) != raw:
                        raise ValueError('Captured score does not match original score_features API')
                    capture.update(local_features=local, global_features=pooled)
                    path = output / f'{sequence}_{index}.npz'
                    np.savez(path, **{key: value.cpu().numpy() for key, value in capture.items()})
                    original = targets[index]
                    prefix = 'reference' if args.implementation == 'full' else 'candidate'
                    record['captures'].append({'sequence': sequence, 'target_frame': index,
                        'frames_content_sha256': row['frames_content_sha256'], 'phase': phase,
                        'feature_raw': raw, 'original_phase': original[prefix + '_phase'],
                        'original_feature_raw': original[prefix + '_feature_raw'],
                        'feature_raw_difference_from_original': raw - original[prefix + '_feature_raw'],
                        'original_raw_within_fixed_gate': bool(np.isclose(original[prefix + '_feature_raw'], raw,
                            rtol=GATES['raw_rtol'], atol=GATES['raw_atol'])),
                        'original_phase_within_fixed_gate': bool(abs((phase - original[prefix + '_phase'] + .5) % 1 - .5)
                            <= GATES['phase_circular_max_abs']),
                        'array_file': str(path.relative_to(repo)), 'array_sha256': file_hash(path)})
                    atomic_record(output / 'capture.json', record)
                require_isolated_gpu()
        expected_targets = {(item['sequence'], item['target_frame']) for item in diagnostic['failed_targets']}
        if {(item['sequence'], item['target_frame']) for item in record['captures']} != expected_targets:
            raise ValueError('Incomplete original failed-context capture')
        if any(file_hash(Path(path)) != expected for path, expected in pins.items()):
            raise ValueError('Probe inputs changed during actual capture')
        record.update(status='complete_untimed_gpu_capture', torch=str(torch.__version__),
                      gpu=torch.cuda.get_device_name(), cuda_matmul_tf32=False, cudnn_tf32=False)
        atomic_record(output / 'capture.json', record)
        print(json.dumps({'status': record['status'], 'captures': len(record['captures']),
                          'implementation': args.implementation, 'original_parity_status': 'failed'}))


if __name__ == '__main__':
    main()
