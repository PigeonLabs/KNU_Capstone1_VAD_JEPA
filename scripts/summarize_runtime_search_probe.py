"""Independently compare untimed GPU captures without changing measured parity.

FP64 arithmetic promotes captured FP32 queries/prototypes. It does not replay
the encoder, PCA, primary FP32 CUDA kernels, alarms, or throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ipad_jepa.cache_retirement import atomic_record
from probe_runtime_search import CONDITION, PAIR, file_hash, verify_pair_inputs
from verify_runtime_parity import GATES


BANK = Path('artifacts/runs/dinov3-l/online/R01/seed0/memory.npz')


def validate_pins(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError('Nonempty source hashes required')
    for path, expected in pins.items():
        if file_hash(path) != expected:
            raise ValueError(f'Changed source: {path}')


def neighbor_changes(left, right):
    """Distinguish ordered-rank changes from membership changes."""
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 5
            or left.dtype.kind not in 'iu' or right.dtype.kind not in 'iu'
            or np.any(np.diff(np.sort(left, axis=-1), axis=-1) == 0)
            or np.any(np.diff(np.sort(right, axis=-1), axis=-1) == 0)):
        raise ValueError('Paired distinct integer k=5 neighbor IDs required')
    return {'ordered_changed_patches': int(np.any(left != right, axis=-1).sum()),
            'membership_changed_patches': int(np.any(np.sort(left, axis=-1)
                                                       != np.sort(right, axis=-1), axis=-1).sum())}


def score_with_ids(query, prototypes, ids, temperature):
    """Explicit FP64 distances, softmax projection, and freshly selected top 5%."""
    query, prototypes, ids = [np.asarray(a) for a in (query, prototypes, ids)]
    if (query.ndim != 3 or prototypes.ndim != 2 or query.shape[-1] != prototypes.shape[-1]
            or ids.shape != query.shape[:2] + (5,) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= len(prototypes))
            or not np.isfinite(query).all() or not np.isfinite(prototypes).all()
            or not np.isfinite(temperature) or temperature <= 0
            or np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0)):
        raise ValueError('Finite queries/bank, positive temperature, and valid distinct k=5 IDs required')
    query = query.astype(np.float64)
    neighbors = prototypes.astype(np.float64)[ids]
    distances = np.sum((query[:, :, None] - neighbors) ** 2, axis=-1)
    logits = -distances / max(float(temperature), 1e-6)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    patches = np.sum((query - np.sum(neighbors * weights[..., None], axis=-2)) ** 2, axis=-1)
    count = max(1, math.ceil(query.shape[1] * .05))
    top_ids = np.argsort(-patches, axis=1, kind='stable')[:, :count]
    scores = np.take_along_axis(patches, top_ids, axis=1).mean(axis=1)
    return {'distances': distances, 'patches': patches, 'top_ids': top_ids, 'scores': scores}


def check_arrays(arrays, prototypes, temperature, phase):
    """Check captured schema and independently replay stable neighbors/scores."""
    shapes = {'query': ((1, 576, 256), 'float32'), 'phase_bins': ((1, 3), 'int64'),
              'local_features': ((1, 576, 1024), 'float32'),
              'global_features': ((1, 1024), 'float32'),
              'distance_rounding_max_abs': ((), 'float64')}
    for prefix, precision in [('fast', 'float32'), ('stable', 'float64')]:
        shapes.update({prefix + '_ids': ((1, 576, 5), 'int64'),
                       prefix + '_distances': ((1, 576, 5), precision),
                       prefix + '_patch_scores': ((1, 576), precision),
                       prefix + '_top_patch_ids': ((1, 29), 'int64'),
                       prefix + '_frame_score': ((1,), precision),
                       prefix + '_k5_k6_margin': ((1, 576), precision)})
    if set(arrays) != set(shapes):
        raise ValueError('Incomplete or unexpected capture fields')
    for key, (shape, dtype) in shapes.items():
        if arrays[key].shape != shape or str(arrays[key].dtype) != dtype or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid capture array: {key}')
    if not np.isfinite(phase) or not 0 <= phase < 1:
        raise ValueError('Valid recorded phase required')
    center = int(np.float32(phase) * np.float32(16))
    bins = np.sort([(center - 1) % 16, center, (center + 1) % 16])
    if not np.array_equal(arrays['phase_bins'][0], bins):
        raise ValueError('Captured bins differ from actual FP32 phase selection')
    candidate_ids = (bins[:, None] * 128 + np.arange(128)).ravel()
    flat = prototypes.reshape(2048, 256)
    query = arrays['query'].astype(np.float64)
    # Independent explicit differences for all 384 candidates, bounded chunks.
    distances = np.concatenate([np.sum((part[:, :, None] - flat[candidate_ids].astype(np.float64)[None, None]) ** 2,
                                       axis=-1) for part in np.split(query, 18, axis=1)], axis=1)
    ordered = np.sort(distances, axis=-1)
    stable = score_with_ids(arrays['query'], flat, arrays['stable_ids'], temperature)
    for prefix in ('fast', 'stable'):
        ids = arrays[prefix + '_ids']
        neighbor_changes(ids, ids)
        if not np.isin(ids, candidate_ids).all():
            raise ValueError('Neighbor outside selected phase bins')
        top = arrays[prefix + '_top_patch_ids']
        if (np.any(top < 0) or np.any(top >= 576)
                or np.any(np.diff(np.sort(top, axis=1), axis=1) == 0)
                or np.any(arrays[prefix + '_k5_k6_margin'] < 0)
                or np.any(arrays[prefix + '_distances'] < 0)):
            raise ValueError('Invalid selected top patches or distances')
        patches = arrays[prefix + '_patch_scores'].astype(np.float64)
        selected = np.take_along_axis(patches, top, axis=1)
        if not np.allclose(np.sort(selected, axis=1), np.sort(patches, axis=1)[:, -29:], rtol=0, atol=1e-12):
            raise ValueError('Captured top patch selection is not the upper 5%')
        tolerance = 1e-7 if prefix == 'fast' else 1e-12
        if not np.allclose(selected.mean(axis=1), arrays[prefix + '_frame_score'], rtol=0, atol=tolerance):
            raise ValueError('Captured top patch mean differs from captured score')
    checks = [(stable['distances'], arrays['stable_distances']),
              (stable['distances'], ordered[..., :5]),
              (stable['patches'], arrays['stable_patch_scores']),
              (stable['scores'], arrays['stable_frame_score']),
              (ordered[..., 5] - ordered[..., 4], arrays['stable_k5_k6_margin'])]
    errors = [float(np.max(np.abs(a - b))) for a, b in checks]
    if max(errors) > 1e-12:
        raise ValueError('Independent FP64 replay disagrees with actual GPU capture')
    return max(errors)


def verify_capture_pair(records, diagnostic, pins):
    targets = {(r['sequence'], r['target_frame']): r for r in diagnostic['failed_targets']}
    if set(records) != {'full', 'reuse'}:
        raise ValueError('Both captured implementations required')
    merged = dict(pins)
    inventories = {}
    for implementation, record in records.items():
        if (record.get('status') != 'complete_untimed_gpu_capture' or record.get('implementation') != implementation
                or record.get('condition') != CONDITION or record.get('original_parity_status') != 'failed'
                or record.get('warmup_forward_calls') != 20 or record.get('cuda_matmul_tf32') is not False
                or record.get('cudnn_tf32') is not False):
            raise ValueError('Original completed FP32 GPU captures required')
        if any(record.get('source_sha256', {}).get(k) != v for k, v in pins.items()):
            raise ValueError('Capture missing original measured source bindings')
        for path, expected in record['source_sha256'].items():
            if path in merged and merged[path] != expected:
                raise ValueError('Conflicting capture provenance')
            merged[path] = expected
        inventory = {(r['sequence'], r['target_frame']): r for r in record['captures']}
        if len(inventory) != len(record['captures']) or set(inventory) != set(targets):
            raise ValueError('Exact distinct original failed targets required')
        prefix = 'reference' if implementation == 'full' else 'candidate'
        for key, row in inventory.items():
            original = targets[key]
            if (row['original_feature_raw'] != original[prefix + '_feature_raw']
                    or row['original_phase'] != original[prefix + '_phase']
                    or row['feature_raw_difference_from_original'] != row['feature_raw'] - row['original_feature_raw']
                    or row['original_raw_within_fixed_gate'] is not True
                    or row['original_phase_within_fixed_gate'] is not True
                    or not np.isclose(row['original_feature_raw'], row['feature_raw'],
                                      rtol=GATES['raw_rtol'], atol=GATES['raw_atol'])
                    or abs((row['phase'] - row['original_phase'] + .5) % 1 - .5) > GATES['phase_circular_max_abs']):
                raise ValueError('Original measured raw/phase reproduction failed')
        inventories[implementation] = inventory
    if (records['full']['torch'] != records['reuse']['torch']
            or records['full']['gpu'] != records['reuse']['gpu']
            or records['full']['source_sha256'] != records['reuse']['source_sha256']):
        raise ValueError('Same GPU/build and original model/bank sources required')
    return inventories, merged


def compare_arrays(left, right, prototypes, temperature):
    if not np.array_equal(left['phase_bins'], right['phase_bins']):
        raise ValueError('This comparison requires the same selected phase bank')
    a, b = [float(x['stable_frame_score'][0]) for x in (left, right)]
    fixed = score_with_ids(right['query'], prototypes.reshape(2048, 256), left['stable_ids'], temperature)
    fixed_score = float(fixed['scores'][0])
    fast_delta = float(right['fast_frame_score'][0]) - float(left['fast_frame_score'][0])
    stable_delta = b - a
    row = {'fast_raw_delta': fast_delta, 'stable_raw_delta': stable_delta,
           'stable_fixed_reference_ids_score': fixed_score,
           'fixed_ids_query_effect': fixed_score - a, 'neighbor_selection_effect': b - fixed_score,
           'stable_raw_within_original_tolerance': bool(np.isclose(a, b, rtol=GATES['raw_rtol'], atol=GATES['raw_atol'])),
           'fast_raw_within_original_tolerance': bool(np.isclose(float(left['fast_frame_score'][0]),
                                                               float(right['fast_frame_score'][0]),
                                                               rtol=GATES['raw_rtol'], atol=GATES['raw_atol']))}
    for key in ('local_features', 'global_features', 'query'):
        diff = right[key].astype(np.float64) - left[key].astype(np.float64)
        row[key + '_max_abs_difference'] = float(np.max(np.abs(diff)))
        row[key + '_relative_l2_difference'] = float(np.linalg.norm(diff) / max(np.linalg.norm(left[key]), 1e-30))
    for prefix in ('fast', 'stable'):
        row[prefix + '_neighbors'] = neighbor_changes(left[prefix + '_ids'], right[prefix + '_ids'])
        row[prefix + '_membership_changed_patch_indices'] = np.flatnonzero(np.any(
            np.sort(left[prefix + '_ids'], axis=-1) != np.sort(right[prefix + '_ids'], axis=-1), axis=-1)).tolist()
        row[prefix + '_top29_membership_equal'] = bool(np.array_equal(np.sort(left[prefix + '_top_patch_ids']),
                                                                    np.sort(right[prefix + '_top_patch_ids'])))
        row[prefix + '_min_k5_k6_margin_full_reuse'] = [float(x[prefix + '_k5_k6_margin'].min()) for x in (left, right)]
        row[prefix + '_zero_margin_patches_full_reuse'] = [int((x[prefix + '_k5_k6_margin'] == 0).sum()) for x in (left, right)]
        row[prefix + '_raw_scores_full_reuse'] = [float(x[prefix + '_frame_score'][0]) for x in (left, right)]
    row['distance_rounding_max_abs_full_reuse'] = [float(x['distance_rounding_max_abs']) for x in (left, right)]
    return row


def plot_report(rows, figure):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
    x = np.arange(len(rows)); labels = [f"{r['sequence']} / t={r['target_frame']}" for r in rows]
    for offset, key, label, color in [(-.18, 'fast_raw_delta', 'Actual FP32 search', '#cf774a'),
                                      (.18, 'stable_raw_delta', 'Supplemental FP64 distance', '#427da1')]:
        axes[0].bar(x + offset, [r[key] * 1e6 for r in rows], .34, label=label, color=color)
    axes[0].set(ylabel='Reuse minus full raw score (×10⁻⁶)', title='Same captured FP32 queries/bank')
    axes[0].set_ylim(0, max(abs(r['fast_raw_delta']) for r in rows) * 1e6 * 1.3)
    axes[0].text(x[0] + .18, 6, f"{rows[0]['stable_raw_delta'] * 1e6:+.3f}", ha='center', fontsize=8, color='#427da1')
    axes[0].legend(frameon=False, fontsize=8)
    for offset, key, label, color in [(-.18, 'fixed_ids_query_effect', 'Query effect with full IDs', '#618e83'),
                                      (.18, 'neighbor_selection_effect', 'Neighbor selection effect', '#cf774a')]:
        axes[1].bar(x + offset, [r[key] * 1e6 for r in rows], .34, label=label, color=color)
    axes[1].set(ylabel='Directional FP64 decomposition (×10⁻⁶)', title='Top 29 patches reselected each time')
    axes[1].set_ylim(-8, max(abs(r['neighbor_selection_effect']) for r in rows) * 1e6 * 1.3)
    for index, row in enumerate(rows):
        axes[1].text(index - .18, 6, f"{row['fixed_ids_query_effect'] * 1e6:+.3f}", ha='center', fontsize=8, color='#618e83')
    axes[1].legend(frameon=False, fontsize=8)
    for offset, prefix, label, color in [(-.18, 'fast', 'FP32 membership changes', '#cf774a'),
                                         (.18, 'stable', 'FP64 membership changes', '#427da1')]:
        bars = axes[2].bar(x + offset, [r[prefix + '_neighbors']['membership_changed_patches'] for r in rows], .34,
                           label=label, color=color)
        axes[2].bar_label(bars, padding=3)
    axes[2].set(ylabel='Changed neighbor sets / 576 patches', title='Ordered ranks tracked separately', ylim=(0, 1.5))
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set(xticks=x, xticklabels=labels); axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15); axis.set_axisbelow(True)
    fig.suptitle('DINOv3-L / online / R01 / seed 0: three original failed FP32 contexts', fontsize=13)
    fig.supxlabel('Separate untimed GPU captures; independent NumPy FP64 replay. Original all-video parity remains FAILED.\n'
                  'FP64 distances do not make all three contexts pass; no new FPS, accuracy, or all-video equivalence.', fontsize=9)
    figure.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(figure.with_suffix('.' + extension), dpi=180)
    plt.close(fig)
    return {str(figure.with_suffix('.' + ext)): file_hash(figure.with_suffix('.' + ext)) for ext in ('png', 'svg')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--reuse', type=Path, required=True)
    parser.add_argument('--pair', type=Path, default=PAIR)
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    diagnostic, pins = verify_pair_inputs(args.pair)
    roots = {'full': args.full, 'reuse': args.reuse}
    records = {k: json.loads((root / 'capture.json').read_text()) for k, root in roots.items()}
    expected_paths = [('results/stage00/manifest.json', 'manifest_sha256'),
                      ('artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', 'weights_sha256'),
                      ('artifacts/runs/dinov3-l/online/R01/seed0/phase_head.pt', 'phase_head_sha256'),
                      (str(args.bank), 'memory_sha256')]
    for implementation, record in records.items():
        meta = json.loads((args.pair / implementation / 'runtime.json').read_text())
        if (record.get('torch') != meta['torch'] or record.get('gpu') != meta['gpu']
                or any(record.get('source_sha256', {}).get(path) != meta[field] for path, field in expected_paths)
                or record.get('source_sha256', {}).get('scripts/probe_runtime_search.py')
                   != file_hash('scripts/probe_runtime_search.py')):
            raise ValueError('Capture missing actual original model/build/producer bindings')
    inventories, sources = verify_capture_pair(records, diagnostic, pins)
    if sources.get(str(args.bank)) != file_hash(args.bank):
        raise ValueError('Actual originally measured bank required')
    sources[str(Path(__file__).resolve().relative_to(Path.cwd().resolve()))] = file_hash(__file__)
    manifest_path = Path('results/stage00/manifest.json')
    manifest = json.loads(manifest_path.read_text())['sequences']
    for root in roots.values():
        sources[str(root / 'capture.json')] = file_hash(root / 'capture.json')
    validate_pins(sources)
    with np.load(args.bank, allow_pickle=False) as bank:
        prototypes = bank['prototypes'].copy(); temperature = float(bank['temperature'])
        if (prototypes.shape != (16, 128, 256) or prototypes.dtype != np.float32
                or not np.isfinite(prototypes).all() or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError('Original finite FP32 16×128×256 bank required')
    comparisons = []
    for key in sorted(inventories['full']):
        captured = {}; errors = []
        for implementation, root in roots.items():
            row = inventories[implementation][key]
            path = root / f'{key[0]}_{key[1]}.npz'
            if Path(row['array_file']).resolve() != path.resolve() or file_hash(path) != row['array_sha256']:
                raise ValueError('Changed or misplaced captured payload')
            sources[str(path)] = row['array_sha256']
            raw = next(r for r in manifest if r['device'] == 'R01' and r['partition'] == 'testing' and r['sequence'] == key[0])
            if row['frames_content_sha256'] != raw['frames_content_sha256']:
                raise ValueError('Capture raw input binding differs from original manifest')
            with np.load(path, allow_pickle=False) as payload:
                arrays = {name: payload[name].copy() for name in payload.files}
            errors.append(check_arrays(arrays, prototypes, temperature, row['phase']))
            if float(arrays['fast_frame_score'][0]) != row['feature_raw']:
                raise ValueError('GPU capture score differs from recorded actual score')
            captured[implementation] = arrays
        comparison = compare_arrays(captured['full'], captured['reuse'], prototypes, temperature)
        comparison.update(sequence=key[0], target_frame=key[1], max_independent_FP64_replay_error=max(errors))
        comparison['raw_difference_from_original_full_reuse'] = [inventories[k][key]['feature_raw_difference_from_original']
                                                                 for k in ('full', 'reuse')]
        comparison['phase_bins_full_reuse'] = [captured[k]['phase_bins'][0].tolist() for k in ('full', 'reuse')]
        comparisons.append(comparison)
    figures = plot_report(comparisons, args.figure)
    validate_pins(sources)
    report = {'status': 'passed_source_bound_capture_comparison_and_independent_FP64_replay',
              'condition': CONDITION, 'original_all_video_parity_status': 'failed', 'gates': GATES,
              'untimed_captured_contexts': len(comparisons), 'captured_implementations': ['full', 'reuse'],
              'gpu': records['full']['gpu'], 'capture_torch': records['full']['torch'],
              'cuda_matmul_tf32': False, 'cudnn_tf32': False,
              'source_sha256': sources, 'figures': figures, 'comparisons': comparisons,
              'reporting_runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
              'stable_raw_tolerance_failed_contexts': sum(not r['stable_raw_within_original_tolerance'] for r in comparisons),
              'finding': 'Compare actual FP32 and supplemental FP64 raw deltas against the unchanged raw tolerance. '
                         'Ordered neighbor-rank differences are distinct from membership changes. The directional held-ID '
                         'counterfactual separates query effects from neighbor selection effects under FP64 search arithmetic.',
              'counterfactual': 'Candidate captured FP32 query promoted to FP64 with reference stable k=5 IDs; recompute distances, '
                                'softmax, projection, patch residuals and top29 mean. Query effect = fixed-ID score minus full stable score; '
                                'selection effect = reuse stable score minus fixed-ID score. Directional, no assumption of equal top29 sets.',
              'limits': 'Three selected failed contexts only. Not FP64 encoder/PCA inference, an independent replay of primary FP32 CUDA kernels, '
                        'a parity repair, sole-cause proof, new accuracy/FPS, full-video FP64 evaluation, or general streaming equivalence. '
                        'Original measured traces/scorer/calibration/gates/parity are unchanged. Capture arrays and model/bank remain private.'}
    atomic_record(args.pair / 'search_probe_comparison.json', report)
    print(json.dumps({'status': report['status'], 'contexts': len(comparisons),
                      'original_parity': 'failed', 'maximum_replay_error': max(r['max_independent_FP64_replay_error'] for r in comparisons)}))


if __name__ == '__main__':
    main()
