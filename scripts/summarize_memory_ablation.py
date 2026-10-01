"""Audit normal bank geometry/GT/calibration and aggregate memory OFAT accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
from ipad_jepa.alignment import align
from summarize_experiments import load_run, check_pair, statistic, bootstrap, bootstrap_draws, macro_four_devices
from summarize_neighbour_ablation import digest, read, write, difference, RAW_TOLERANCE, SCORE_TOLERANCE

CONFIGURATIONS = {'B16_M1024': (16, 1024), 'B16_M2048': (16, 2048),
                  'B16_M4096': (16, 4096), 'B8_M2048': (8, 2048)}
BASE = 'B16_M2048'
MODELS = ['dinov3-l', 'vjepa21-l']
MODES = ['offline', 'online']
DEVICES = ['R01', 'R02', 'R03', 'R04']


def verify_run(folder, source, manifest, data_root, model, mode, device, seed, label, memory_root, local):
    meta = json.loads((folder / 'metrics.json').read_text())
    normal = json.loads((folder / 'normal_fit.json').read_text())
    proof = json.loads((folder / 'source_replay_check.json').read_text())
    if (meta['status'] != 'complete_device_evaluation' or normal['configuration'] != label or meta['configuration'] != label or
        any(tuple(obj[key] for key in ['backbone', 'mode', 'device', 'seed']) != (model, mode, device, seed)
            for obj in [meta, normal, proof]) or proof['status'] != 'passed' or
        proof['default_alarm_mismatches_original_mask'] != 0 or
        proof['raw_tolerance'] != RAW_TOLERANCE or proof['score_tolerance'] != SCORE_TOLERANCE or
        normal['history'] != 5 or normal['score_weights'] != [.5, .5] or
        normal['code_sha256'] != digest('src/ipad_jepa/memory_ablation.py') or
        normal['search_code_sha256'] != digest('src/ipad_jepa/torch_memory.py') or
        normal['temporal_code_sha256'] != digest('src/ipad_jepa/temporal.py')):
        raise ValueError('Completed condition or source replay differs')
    bins, budget = CONFIGURATIONS[label]
    if ((normal['bins'], normal['total_prototypes'], normal['prototypes_per_bin'], normal['pca_dimensions']) !=
        (bins, budget, budget // bins, 256) or normal['neighbours'] != 5 or
        (meta['bins'], meta['total_prototypes']) != (bins, budget) or
        normal['memory_code_sha256'] != digest('src/ipad_jepa/memory.py') or
        normal['candidate_code_sha256'] != digest('src/ipad_jepa/cache_data.py') or
        normal['temperature_code_sha256'] != digest('src/ipad_jepa/experiment.py') or
        proof['default_array_replay'] != 'exact mean/components/prototypes, dtype and strides' or
        proof['default_temperature_replay'] != 'exact normal calibration fifth-neighbour median and sample count' or
        normal['temperature'] < 1e-6 or not np.isfinite(normal['temperature'])):
        raise ValueError('Normal bank geometry or fitted algorithm differs')
    private = memory_root / label / model / mode / device / f'seed{seed}' / 'memory.npz'
    if digest(private) != normal['bank_sha256']:
        raise ValueError('Rebuilt private bank checksum differs')
    with np.load(private, allow_pickle=False) as bank:
        if (bank['mean'].shape != (1024,) or bank['components'].shape != (256, 1024) or
            bank['prototypes'].shape != (bins, budget // bins, 256) or
            float(bank['temperature']) != normal['temperature'] or
            float(bank['cycle_length']) != normal['cycle_length_fit_median'] or
            any(not np.isfinite(bank[n]).all() for n in ['mean', 'components', 'prototypes'])):
            raise ValueError('Actual normal bank arrays/temperature differ')
        if label == BASE:
            source_bank = local / model / mode / device / f'seed{seed}' / 'memory.npz'
            if digest(source_bank) != proof['source_memory_sha256']:
                raise ValueError('Original bank checksum differs')
            with np.load(source_bank, allow_pickle=False) as original_bank:
                for name in ['mean', 'components', 'prototypes']:
                    np.testing.assert_array_equal(bank[name], original_bank[name])
                    if bank[name].strides != original_bank[name].strides or bank[name].dtype != original_bank[name].dtype:
                        raise ValueError('Default bank layout differs from source')
                if float(bank['temperature']) != float(original_bank['temperature']):
                    raise ValueError('Default normal temperature differs')
        if bins == 16:
            large_path = memory_root / 'B16_M4096' / model / mode / device / f'seed{seed}' / 'memory.npz'
            with np.load(large_path, allow_pickle=False) as large:
                np.testing.assert_array_equal(bank['mean'], large['mean'])
                np.testing.assert_array_equal(bank['components'], large['components'])
                np.testing.assert_array_equal(bank['prototypes'], large['prototypes'][:, :budget // bins])
    original = source / model / mode / device / f'seed{seed}'
    for name, value in proof['sources'].items():
        if digest(original / name) != value:
            raise ValueError('Published source changed')
    old_normal = json.loads((original / 'normal_fit.json').read_text())
    if (normal['source_memory_sha256'] != proof['source_memory_sha256'] or
        normal['source_cache_identity'] != old_normal['cache_identity'] or
        normal['source_phase_checkpoint_sha256'] != old_normal['phase_checkpoint_sha256'] or
        normal['source_normal_fit_sha256'] != digest(original / 'normal_fit.json')):
        raise ValueError('Fixed head or source memory identity differs')
    if label == BASE and (normal['temperature'] != old_normal['temperature'] or
                          normal['temperature_samples'] != old_normal['temperature_samples']):
        raise ValueError('Default normal calibration temperature differs')
    phase_dir = local / model / mode / device / f'seed{seed}'
    if (digest(phase_dir / 'phase_training.json') != proof['phase_metadata_sha256'] or
        digest(phase_dir / 'phase_head.pt') != normal['source_phase_checkpoint_sha256']):
        raise ValueError('Fixed trained phase head provenance differs')
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
    if label == BASE:
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
        if label == BASE:
            np.testing.assert_allclose(raw[:, 0], [float(r['feature_raw']) for r in original_rows], **RAW_TOLERANCE)
            valid = np.array([r['valid'] == '1' for r in trace])
            np.testing.assert_allclose(scores[valid], np.array([float(r['score']) for r in original_rows])[valid], **SCORE_TOLERANCE)
            streak = 0
            for keep, value, original_row in zip(common, scores, original_rows):
                streak = streak + 1 if keep and value > params['threshold'] else 0
                if int(streak >= 3) != int(original_row['alarm']):
                    raise ValueError('Default original-mask alarm replay differs')
        output_hashes[file.name] = digest(file)
        label_hashes[file.stem] = info['label_sha256']
    run = load_run(folder, 'P3', meta)
    check_pair(run, load_run(original, 'P3'))
    return run, {'backbone': model, 'mode': mode, 'device': device, 'seed': seed, 'configuration': label, 'bins': CONFIGURATIONS[label][0], 'total_prototypes': CONFIGURATIONS[label][1],
        'normal_frames': len(selected), 'threshold': params['threshold'],
        'normal_calibration_sha256': digest(folder / 'normal_calibration.csv'),
        'normal_fit_sha256': digest(folder / 'normal_fit.json'), 'metrics_sha256': digest(folder / 'metrics.json'),
        'source_replay_sha256': digest(folder / 'source_replay_check.json'), 'score_csv_sha256': output_hashes,
        'actual_annotation_sha256': label_hashes, 'bank_sha256': normal['bank_sha256'],
        'bank_geometry_checked': [bins, budget // bins, 256], 'temperature': normal['temperature']}, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('root', 'results/stage05/ablations/memory'), ('source', 'results/stage02'),
                          ('memory-root', 'artifacts/ablations/memory'), ('local', 'artifacts/runs'),
                          ('manifest', 'results/stage00/manifest.json'), ('data-root', '../IPAD_dataset/IPAD_dataset')]:
        parser.add_argument('--' + name, type=Path, default=Path(default))
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())['sequences']
    required = [(m, mode, d) for m in MODELS for mode in MODES for d in DEVICES]
    complete = [g for g in required if all((args.root / label / g[0] / g[1] / g[2] / f'seed{s}' / 'metrics.json').is_file()
                for label in CONFIGURATIONS for s in [0, 1, 2])]
    if not complete or (args.require_full and len(complete) != 16):
        raise ValueError('Required completed memory group inventory is missing')
    completion = args.root / 'completion.json'
    completion_sha = None
    if args.require_full or completion.exists():
        marker = json.loads(completion.read_text())
        if (marker['status'] != ('complete_memory_matrix' if len(complete) == 16 else 'complete_memory_group') or
            marker['groups'] != len(complete) or marker['conditions'] != len(complete) * 4 or
            marker['evaluated_seed_conditions'] != len(complete) * 12 or
            marker['configurations'] != {k: list(v) for k, v in CONFIGURATIONS.items()} or
            marker['code_sha256'] != digest('src/ipad_jepa/memory_ablation.py') or
            marker['source_summary_sha256'] != digest(args.source / 'device_summary.json') or
            marker['manifest_sha256'] != digest(args.manifest)):
            raise ValueError('Completion marker differs from actual matrix or source')
        completion_sha = digest(completion)
    rows, deltas, checks, stored, points, paired, references, proofs = [], [], [], {}, {}, {}, {}, {}
    for model, mode, device in complete:
        for label, (bins, budget) in CONFIGURATIONS.items():
            runs = []
            for seed in [0, 1, 2]:
                folder = args.root / label / model / mode / device / f'seed{seed}'
                run, check, proof = verify_run(folder, args.source, manifest, args.data_root,
                    model, mode, device, seed, label, args.memory_root, args.local)
                checks.append(check); proofs[model, mode, device, seed] = proof
                if device in references:
                    check_pair(references[device], run)
                else:
                    references[device] = run
                runs.append(run)
            point = np.mean([statistic(r, list(r)) for r in runs], axis=0)
            draws, rejected = bootstrap(runs)
            low, high = np.quantile(draws, [.025, .975], axis=0)
            key = (model, mode, device, label)
            stored[key], points[key] = (runs, draws), point
            paired[key] = bootstrap_draws(runs, seed=np.random.SeedSequence([2026, int(device[1:])]))
            labels = np.concatenate([v[1] for v in runs[0].values()])
            rows.append({'backbone': model, 'mode': mode, 'device': device, 'configuration': label,
                'bins': bins, 'total_prototypes': budget, 'score_variant': 'P3', 'seeds': 3,
                'test_videos': len(runs[0]), 'frames': len(labels), 'anomaly_frames': int(labels.sum()),
                'auroc_mean': float(point[0]), 'auroc_ci_low': float(low[0]), 'auroc_ci_high': float(high[0]),
                'ap_mean': float(point[1]), 'ap_ci_low': float(low[1]), 'ap_ci_high': float(high[1]),
                'bootstrap_draws': 1000, 'bootstrap_rejected': rejected})
        for label in CONFIGURATIONS:
            if label == BASE:
                continue
            a, b = (model, mode, device, BASE), (model, mode, device, label)
            deltas.append(difference(model, mode, f'{label}_minus_{BASE}', points[b] - points[a],
                stored[b][1] - stored[a][1], device))
    macro = macro_four_devices(stored)
    for row in macro['results']:
        row.update(configuration=row['variant'], bins=CONFIGURATIONS[row['variant']][0],
                   total_prototypes=CONFIGURATIONS[row['variant']][1], score_variant='P3')
    for model in MODELS:
        for mode in MODES:
            if not all((model, mode, d, BASE) in points for d in DEVICES):
                continue
            baseline = np.mean([paired[model, mode, d, BASE] for d in DEVICES], axis=0)
            for label in CONFIGURATIONS:
                if label == BASE:
                    continue
                target = np.mean([paired[model, mode, d, label] for d in DEVICES], axis=0)
                point = np.mean([points[model, mode, d, label] - points[model, mode, d, BASE] for d in DEVICES], axis=0)
                macro['paired_deltas'].append(difference(model, mode, f'{label}_minus_{BASE}', point, target - baseline))
    scope = ('Three fixed seed metrics averaged; original t=19..N-8 shared GT mask; frozen encoder/head/clip16/history5. '
             'Budget shares normal candidates/PCA; bin comparison refits phase-stratified normal samples/PCA. '
             'Per-bank normal calibration temperature/median/MAD/q99; no test selection')
    write(args.root / 'device_summary', {'scope': scope, 'results': rows, 'paired_deltas': deltas})
    write(args.root / 'macro_summary', macro)
    proof = {'status': 'passed', 'matrix_complete': len(complete) == 16, 'completed_groups': len(complete),
        'excluded_groups': [list(g) for g in required if g not in complete], 'thresholds_checked': len(checks),
        'source_conditions_replayed': len(proofs), 'checks': checks,
        'default_feature_max_absolute_error': max(p['default_feature_max_absolute_error'] for p in proofs.values()),
        'default_score_max_absolute_error': max(p['default_score_max_absolute_error'] for p in proofs.values()),
        'unchanged_time_max_absolute_error': max(p['unchanged_time_max_absolute_error'] for p in proofs.values()),
        'default_alarm_mismatches_original_mask': 0, 'default_array_replay': 'exact actual private arrays, dtype/strides',
        'raw_tolerance': RAW_TOLERANCE, 'score_tolerance': SCORE_TOLERANCE,
        'device_groups': len(rows), 'macro_groups': len(macro['results']),
        'device_summary_sha256': digest(args.root / 'device_summary.json'),
        'macro_summary_sha256': digest(args.root / 'macro_summary.json'),
        'source_summary_sha256': digest(args.source / 'device_summary.json'), 'manifest_sha256': digest(args.manifest),
        'producer_code_sha256': digest('src/ipad_jepa/memory_ablation.py'), 'verifier_sha256': digest(__file__),
        'shared_verifier_sha256': digest(Path(__file__).with_name('summarize_neighbour_ablation.py')),
        'bootstrap_helper_sha256': digest(Path(__file__).with_name('summarize_experiments.py')),
        'bootstrap_metric_sha256': digest('src/ipad_jepa/bootstrap_metrics.py'), 'completion_sha256': completion_sha,
        'annotation_policy_sha256': digest('src/ipad_jepa/alignment.py'),
        'scope': scope + '; private bank geometry/prefix/default replay and CSV/GT/calibration/alarm audit, not encoder extraction or runtime'}
    (args.root / 'validation.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
