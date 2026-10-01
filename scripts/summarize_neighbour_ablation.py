"""Verify completed neighbour groups and summarize paired whole-video uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ipad_jepa.alignment import align
from summarize_experiments import load_run, check_pair, statistic, bootstrap, bootstrap_draws, macro_four_devices

# Keep CPU aggregation independent of the GPU producer's torch import. Every
# condition must declare these same gates and the current producer source hash.
NEIGHBOURS = (1, 5, 10)
RAW_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-7}
SCORE_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-4}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

MODELS = ['dinov3-l', 'vjepa21-l']
MODES = ['offline', 'online']
DEVICES = ['R01', 'R02', 'R03', 'R04']


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, payload):
    path.with_suffix('.json').write_text(json.dumps(payload, indent=2) + '\n')
    if payload['results']:
        with path.with_suffix('.csv').open('w', newline='') as f:
            fields = [k for k in payload['results'][0] if k != 'device_counts']
            w = csv.DictWriter(f, fields, extrasaction='ignore', lineterminator='\n')
            w.writeheader(); w.writerows(payload['results'])


def difference(model, mode, comparison, point, draws, device=None):
    valid = np.isfinite(draws).all(axis=1)
    if valid.sum() < len(draws) * .95:
        raise ValueError('Too many degenerate paired draws')
    low, high = np.quantile(draws[valid], [.025, .975], axis=0)
    row = {'backbone': model, 'mode': mode, 'comparison': comparison,
        'auroc_delta': float(point[0]), 'auroc_delta_ci_low': float(low[0]), 'auroc_delta_ci_high': float(high[0]),
        'ap_delta': float(point[1]), 'ap_delta_ci_low': float(low[1]), 'ap_delta_ci_high': float(high[1])}
    if device is not None:
        row['device'] = device
    return row


def verify_run(folder, source, manifest, data_root, model, mode, device, seed, k):
    meta = json.loads((folder / 'metrics.json').read_text())
    normal = json.loads((folder / 'normal_fit.json').read_text())
    proof = json.loads((folder / 'source_replay_check.json').read_text())
    if (meta['status'] != 'complete_device_evaluation' or normal['neighbours'] != k or meta['neighbours'] != k or
        any(tuple(obj[key] for key in ['backbone', 'mode', 'device', 'seed']) != (model, mode, device, seed)
            for obj in [meta, normal, proof]) or proof['status'] != 'passed' or
        proof['K5_alarm_mismatches_original_mask'] != 0 or
        proof['raw_tolerance'] != RAW_TOLERANCE or proof['score_tolerance'] != SCORE_TOLERANCE or
        normal['history'] != 5 or normal['score_weights'] != [.5, .5] or
        normal['code_sha256'] != digest('src/ipad_jepa/neighbour_ablation.py') or
        normal['search_code_sha256'] != digest('src/ipad_jepa/torch_memory.py') or
        normal['temporal_code_sha256'] != digest('src/ipad_jepa/temporal.py')):
        raise ValueError('Completed condition or source replay differs')
    original = source / model / mode / device / f'seed{seed}'
    for name, value in proof['sources'].items():
        if digest(original / name) != value:
            raise ValueError('Published source changed')
    old_normal = json.loads((original / 'normal_fit.json').read_text())
    if (normal['temperature'] != old_normal['temperature'] or
        normal['source_memory_sha256'] != proof['memory_sha256'] or
        normal['source_cache_identity'] != old_normal['cache_identity'] or
        normal['source_phase_checkpoint_sha256'] != old_normal['phase_checkpoint_sha256'] or
        normal['source_normal_fit_sha256'] != digest(original / 'normal_fit.json')):
        raise ValueError('Fixed temperature, head or memory identity differs')
    expected = {r['sequence']: r for r in manifest if r['device'] == device and r.get('split') == 'calibration'}
    fit = [r for r in manifest if r['device'] == device and r.get('split') == 'fit']
    if any(r['partition'] != 'training' for r in [*expected.values(), *fit]):
        raise ValueError('Normal fit or calibration uses a test video')
    np.testing.assert_allclose(normal['cycle_length_fit_median'], np.median([r['frames'] for r in fit]), rtol=0, atol=0)
    rows = read(folder / 'normal_calibration.csv'); old_rows = read(original / 'normal_calibration.csv')
    def inventory(values):
        result = {(r['sequence'], int(r['frame'])): r for r in values}
        if len(result) != len(values):
            raise ValueError('Duplicate normal calibration target')
        return result
    before, after = inventory(old_rows), inventory(rows)
    if before.keys() != after.keys() or {r['sequence'] for r in rows} != expected.keys():
        raise ValueError('Calibration targets or normal videos differ')
    for key in before:
        for column in ['predicted_phase', 'relative_phase']:
            if float(before[key][column]) != float(after[key][column]):
                raise ValueError('Fixed predicted phase changed')
    np.testing.assert_allclose([float(r['time_raw']) for r in rows], [float(r['time_raw']) for r in old_rows],
                               rtol=0, atol=1e-12, equal_nan=True)
    for r in rows:
        if int(r['valid']) != int(19 <= int(r['frame']) <= expected[r['sequence']]['frames'] - 8):
            raise ValueError('Calibration mask differs')
    selected = [r for r in rows if r['valid'] == '1']
    pairs = np.array([[float(r['feature_raw']), float(r['time_raw'])] for r in selected])
    median = np.median(pairs, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(pairs - median), axis=0), 1e-6)
    components = np.quantile(pairs, .99, axis=0)
    values = ((pairs - median) / scale).mean(axis=1)
    params = normal['calibration']['P3']
    np.testing.assert_allclose(median, params['median'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(scale, params['mad_scale'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(components, params['component_thresholds'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(values, [float(r['P3']) for r in selected], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.quantile(values, .99), params['threshold'], rtol=0, atol=1e-9)
    if k == 5:
        np.testing.assert_allclose([float(r['feature_raw']) for r in rows], [float(r['feature_raw']) for r in old_rows], **RAW_TOLERANCE)
        np.testing.assert_allclose([float(r['P3']) for r in selected], [float(r['P3']) for r in old_rows if r['valid'] == '1'], **SCORE_TOLERANCE)
    output_hashes = {}
    label_hashes = {}
    tests = {r['sequence']: r for r in manifest if r['device'] == device and r['partition'] == 'testing'}
    if {f.stem for f in (folder / 'P3').glob('*.csv')} != tests.keys():
        raise ValueError('Test video inventory differs from the audited manifest')
    for file in sorted((folder / 'P3').glob('*.csv')):
        trace = read(file); original_rows = read(original / 'P3' / file.name)
        frames = np.array([int(r['frame']) for r in trace])
        info = tests[file.stem]
        targets = np.arange(8, info['frames'] - 7) if mode == 'offline' else np.arange(15, info['frames'])
        np.testing.assert_array_equal(frames, targets)
        label_file = data_root / info['label_file']
        if digest(label_file) != info['label_sha256']:
            raise ValueError('Audited annotation file changed')
        labels, known, _ = align(np.load(label_file, allow_pickle=False), info['frames'])
        np.testing.assert_array_equal([int(float(r['label'])) for r in trace], labels[frames])
        common = (frames >= 19) & (frames <= info['frames'] - 8)
        np.testing.assert_array_equal([int(r['valid']) for r in trace], common & known[frames])
        np.testing.assert_array_equal([int(r['inference_valid']) for r in original_rows], common)
        np.testing.assert_array_equal(frames, [int(r['frame']) for r in original_rows])
        np.testing.assert_array_equal([int(float(r['label'])) for r in trace], [int(float(r['label'])) for r in original_rows])
        np.testing.assert_array_equal([int(r['valid']) for r in trace], [int(r['valid']) for r in original_rows])
        for name in ['phase', 'time_raw']:
            np.testing.assert_allclose([float(r[name]) for r in trace], [float(r[name]) for r in original_rows], rtol=0, atol=1e-12, equal_nan=True)
        raw = np.array([[float(r['feature_raw']), float(r['time_raw'])] for r in trace])
        scores = ((raw - median) / scale).mean(axis=1)
        np.testing.assert_allclose(scores, [float(r['score']) for r in trace], rtol=0, atol=1e-9, equal_nan=True)
        ready = (frames >= 19) & np.isfinite(raw).all(axis=1)
        np.testing.assert_array_equal(ready, [int(r['inference_valid']) for r in trace])
        finite = np.isfinite(raw).all(axis=1)
        types = np.zeros(len(raw), dtype=int)
        types[finite] = (raw[finite, 0] > components[0]).astype(int) + 2 * (raw[finite, 1] > components[1]).astype(int)
        np.testing.assert_array_equal(types, [int(r['evidence_type']) for r in trace])
        streak = 0
        for r in trace:
            streak = streak + 1 if r['inference_valid'] == '1' and float(r['score']) > params['threshold'] else 0
            if int(r['alarm']) != int(streak >= 3):
                raise ValueError('Online-tail/GT-independent alarm replay differs')
        if k == 5:
            np.testing.assert_allclose(raw[:, 0], [float(r['feature_raw']) for r in original_rows], **RAW_TOLERANCE)
            valid = np.array([r['valid'] == '1' for r in trace])
            np.testing.assert_allclose(scores[valid], np.array([float(r['score']) for r in original_rows])[valid], **SCORE_TOLERANCE)
            streak = 0
            for keep, value, original_row in zip(common, scores, original_rows):
                streak = streak + 1 if keep and value > params['threshold'] else 0
                if int(streak >= 3) != int(original_row['alarm']):
                    raise ValueError('K5 original-mask alarm replay differs')
        output_hashes[file.name] = digest(file)
        label_hashes[file.stem] = info['label_sha256']
    run = load_run(folder, 'P3', meta)
    check_pair(run, load_run(original, 'P3'))
    return run, {'backbone': model, 'mode': mode, 'device': device, 'seed': seed, 'neighbours': k,
        'normal_frames': len(selected), 'threshold': params['threshold'],
        'normal_calibration_sha256': digest(folder / 'normal_calibration.csv'),
        'normal_fit_sha256': digest(folder / 'normal_fit.json'), 'metrics_sha256': digest(folder / 'metrics.json'),
        'source_replay_sha256': digest(folder / 'source_replay_check.json'), 'score_csv_sha256': output_hashes,
        'actual_annotation_sha256': label_hashes}, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/stage05/ablations/neighbours'))
    parser.add_argument('--source', type=Path, default=Path('results/stage02'))
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())['sequences']
    required = [(m, mode, d) for m in MODELS for mode in MODES for d in DEVICES]
    complete = [g for g in required if all((args.root / f'K{k}' / g[0] / g[1] / g[2] / f'seed{s}' / 'metrics.json').is_file()
                for k in NEIGHBOURS for s in [0, 1, 2])]
    if not complete or (args.require_full and len(complete) != 16):
        raise ValueError('Requested complete neighbour group inventory is missing')
    completion = args.root / 'completion.json'
    completion_sha = None
    if args.require_full or completion.exists():
        marker = json.loads(completion.read_text())
        expected_status = 'complete_neighbour_matrix' if len(complete) == 16 else 'complete_neighbour_group'
        if (marker['status'] != expected_status or marker['groups'] != len(complete) or
            marker['conditions'] != len(complete) * 3 or marker['evaluated_seed_conditions'] != len(complete) * 9 or
            marker['neighbours'] != list(NEIGHBOURS) or
            marker['code_sha256'] != digest('src/ipad_jepa/neighbour_ablation.py') or
            marker['source_summary_sha256'] != digest(args.source / 'device_summary.json') or
            marker['manifest_sha256'] != digest(args.manifest)):
            raise ValueError('Completion marker differs from the actual matrix or source')
        completion_sha = digest(completion)
    rows, deltas, checks, stored, points, paired, references, proofs = [], [], [], {}, {}, {}, {}, {}
    for model, mode, device in complete:
        for k in NEIGHBOURS:
            runs = []
            for seed in [0, 1, 2]:
                folder = args.root / f'K{k}' / model / mode / device / f'seed{seed}'
                run, check, proof = verify_run(folder, args.source, manifest, args.data_root, model, mode, device, seed, k)
                checks.append(check); proofs[model, mode, device, seed] = proof
                if device in references:
                    check_pair(references[device], run)
                else:
                    references[device] = run
                runs.append(run)
            point = np.mean([statistic(r, list(r)) for r in runs], axis=0)
            draws, rejected = bootstrap(runs)
            low, high = np.quantile(draws, [.025, .975], axis=0)
            key = (model, mode, device, f'K{k}')
            stored[key], points[key] = (runs, draws), point
            paired[key] = bootstrap_draws(runs, seed=np.random.SeedSequence([2026, int(device[1:])]))
            labels = np.concatenate([v[1] for v in runs[0].values()])
            rows.append({'backbone': model, 'mode': mode, 'device': device, 'neighbours': k,
                'score_variant': 'P3', 'seeds': 3, 'test_videos': len(runs[0]), 'frames': len(labels),
                'anomaly_frames': int(labels.sum()), 'auroc_mean': float(point[0]), 'auroc_ci_low': float(low[0]),
                'auroc_ci_high': float(high[0]), 'ap_mean': float(point[1]), 'ap_ci_low': float(low[1]),
                'ap_ci_high': float(high[1]), 'bootstrap_draws': 1000, 'bootstrap_rejected': rejected})
        for k in [1, 10]:
            a, b = (model, mode, device, 'K5'), (model, mode, device, f'K{k}')
            deltas.append(difference(model, mode, f'K{k}_minus_K5', points[b] - points[a], stored[b][1] - stored[a][1], device))
    macro = macro_four_devices(stored)
    for row in macro['results']:
        row.update(neighbours=int(row['variant'][1:]), score_variant='P3')
    for model in MODELS:
        for mode in MODES:
            if not all((model, mode, d, 'K5') in points for d in DEVICES):
                continue
            baseline = np.mean([paired[model, mode, d, 'K5'] for d in DEVICES], axis=0)
            for k in [1, 10]:
                target = np.mean([paired[model, mode, d, f'K{k}'] for d in DEVICES], axis=0)
                point = np.mean([points[model, mode, d, f'K{k}'] - points[model, mode, d, 'K5'] for d in DEVICES], axis=0)
                macro['paired_deltas'].append(difference(model, mode, f'K{k}_minus_K5', point, target - baseline))
    scope = 'Three-seed mean, original t=19..N-8 shared GT mask; normal-only per-k recalibration, fixed encoder/head/PCA/prototypes/temperature/history5'
    write(args.root / 'device_summary', {'scope': scope, 'results': rows, 'paired_deltas': deltas})
    write(args.root / 'macro_summary', macro)
    proof = {'status': 'passed', 'matrix_complete': len(complete) == 16, 'completed_groups': len(complete),
        'excluded_groups': [list(g) for g in required if g not in complete], 'thresholds_checked': len(checks),
        'source_conditions_replayed': len(proofs), 'checks': checks,
        'K5_feature_max_absolute_error': max(p['K5_feature_max_absolute_error'] for p in proofs.values()),
        'K5_score_max_absolute_error': max(p['K5_score_max_absolute_error'] for p in proofs.values()),
        'unchanged_time_max_absolute_error': max(p['unchanged_time_max_absolute_error'] for p in proofs.values()),
        'K5_alarm_mismatches_original_mask': 0, 'raw_tolerance': RAW_TOLERANCE, 'score_tolerance': SCORE_TOLERANCE,
        'device_groups': len(rows), 'macro_groups': len(macro['results']),
        'device_summary_sha256': digest(args.root / 'device_summary.json'),
        'macro_summary_sha256': digest(args.root / 'macro_summary.json'),
        'source_summary_sha256': digest(args.source / 'device_summary.json'), 'manifest_sha256': digest(args.manifest),
        'producer_code_sha256': digest('src/ipad_jepa/neighbour_ablation.py'),
        'verifier_sha256': digest(__file__), 'bootstrap_helper_sha256': digest(Path(__file__).with_name('summarize_experiments.py')),
        'bootstrap_metric_sha256': digest('src/ipad_jepa/bootstrap_metrics.py'),
        'completion_sha256': completion_sha, 'annotation_policy_sha256': digest('src/ipad_jepa/alignment.py'),
        'scope': scope + '; CSV/GT/calibration/search replay audit, not independent encoder extraction or runtime'}
    (args.root / 'validation.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
