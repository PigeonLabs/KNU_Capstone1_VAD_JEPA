"""Frozen prototype-neighbour OFAT with paired seeds and normal-only recalibration."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ipad_jepa.cache_data import FeatureSequence
from ipad_jepa.alignment import align
from ipad_jepa.history_ablation import (SavedInference, alarm_flags, atomic_json,
    decode_inference, metric, pairs_for_history, read_csv)
from ipad_jepa.memory import PrototypeMemory
from ipad_jepa.scoring import NormalCalibration
from ipad_jepa.temporal import common_mask
from ipad_jepa.torch_memory import TorchMemory

NEIGHBOURS = (1, 5, 10)
RAW_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-7}
SCORE_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-4}
# The historical version differs only in completion-marker writes, not scientific calculations.
# Audited with git diff 73e4d116..f1907842 -- src/ipad_jepa/experiment.py.
FROZEN_EVALUATOR_REVISIONS = {
    'fe6a9ad5e1b8c14c336dd29339bee81cff0d43eefe7816eb50cc75e2578eb58b': '73e4d1164086e56db2c534371401360ff1ac7991',
    '4330a26c29da9b10e5f49f12c090018ba11903d8c67ada7a247ef45f5e92a94f': 'f19078425467b4d2b5e54a8d51736639fa42a83f',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load_scorer(bank_path, info, target_device):
    with np.load(bank_path, allow_pickle=False) as stored:
        # Preserve the fitted PCA's Fortran layout: changing strides changes FP32
        # GEMM rounding and can change a neighbour at a tied distance boundary.
        arrays = {k: stored[k].copy(order='K') for k in ['mean', 'components', 'prototypes']}
        if (arrays['mean'].shape != (1024,) or arrays['components'].shape != (256, 1024) or
            arrays['prototypes'].shape != (16, 128, 256) or
            any(not np.isfinite(a).all() for a in arrays.values())):
            raise ValueError('Invalid frozen normal bank dimensions or values')
        if (float(stored['temperature']) != info['temperature'] or
            float(stored['cycle_length']) != info['cycle_length_fit_median']):
            raise ValueError('Frozen bank calibration differs')
    memory = PrototypeMemory(temperature=info['temperature'])
    memory.pca = SimpleNamespace(mean_=arrays['mean'], components_=arrays['components'])
    memory.prototypes = arrays['prototypes']
    return TorchMemory(memory).to(target_device).eval()


def score_cached_sequences(sequence, contexts, predicted, target_device, batch=16):
    """Only patches and previously fixed phase predictions enter prototype queries."""
    if batch < 1 or set(predicted) != set(contexts):
        raise ValueError('Invalid batch or paired seed inventory')
    output = {seed: {k: [] for k in NEIGHBOURS} for seed in contexts}
    with torch.inference_mode():
        for start in range(0, len(sequence.targets), batch):
            features = torch.from_numpy(np.array(sequence.patches[start:start+batch], dtype=np.float32)).to(target_device)
            for seed, context in contexts.items():
                phase = torch.from_numpy(predicted[seed][start:start+batch].astype(np.float32)).to(target_device)
                for k in NEIGHBOURS:
                    output[seed][k].append(context['scorer'].frame_scores(features, phase, k=k).cpu().numpy())
    return {seed: {k: np.concatenate(v) for k, v in group.items()} for seed, group in output.items()}


def infer_pairs(sequence, feature, cycle):
    if feature.shape != sequence.phase.shape or not np.isfinite(feature).all():
        raise ValueError('Expected aligned finite neighbour feature scores')
    return pairs_for_history(SavedInference(sequence.frames, sequence.targets, sequence.phase, feature), cycle, 5)


def recalibrate(pairs, sequences):
    selected = [p[common_mask(s.frames)[s.targets]] for p, s in zip(pairs, sequences)]
    if not selected or any(not len(p) or not np.isfinite(p).all() for p in selected):
        raise ValueError('Missing normal calibration pairs')
    return NormalCalibration().fit(np.concatenate(selected))


def scored(sequence, pairs, cal):
    scores = cal.combine(pairs)
    ready = (sequence.targets >= 19) & np.isfinite(pairs).all(axis=1)
    alarms = alarm_flags(scores, ready, cal.threshold)
    types = np.zeros(len(scores), dtype=np.int8)
    finite = np.isfinite(pairs).all(axis=1)
    types[finite] = cal.types(pairs[finite])
    return scores, ready, alarms, types


def make_context(args, model, mode, device, seed):
    original = args.source / model / mode / device / f'seed{seed}'
    local = args.local / model / mode / device / f'seed{seed}'
    fit = json.loads((original / 'normal_fit.json').read_text())
    metrics = json.loads((original / 'metrics.json').read_text())
    phase = json.loads((local / 'phase_training.json').read_text())
    if (fit['status'] != 'normal_fit_and_calibration_complete' or metrics['status'] != 'complete_device_evaluation' or
        any(tuple(meta[k] for k in ['backbone', 'mode', 'device', 'seed']) != (model, mode, device, seed)
            for meta in [fit, metrics, phase]) or phase['status'] != 'complete' or
        (fit['bins'], fit['total_prototypes'], fit['pca_dimensions']) != (16, 2048, 256) or
        fit['phase_checkpoint_sha256'] != digest(local / 'phase_head.pt') or
        sorted(fit['normal_cache_fingerprints']) != sorted(phase['cache_fingerprints']) or
        fit['code_sha256'] not in FROZEN_EVALUATOR_REVISIONS or
        fit['memory_code_sha256'] != digest(Path(__file__).with_name('memory.py')) or
        fit['cache_identity']['clip_frames'] != 16):
        raise ValueError('Matching completed frozen source, head and memory are required')
    sources = {name: digest(original / name) for name in ['metrics.json', 'normal_fit.json', 'normal_calibration.csv']}
    return {'source': original, 'fit': fit, 'source_metrics': metrics,
        'normal_rows': read_csv(original / 'normal_calibration.csv'), 'sources': sources,
        'memory_sha256': digest(local / 'memory.npz'), 'phase_metadata_sha256': digest(local / 'phase_training.json'),
        'scorer': load_scorer(local / 'memory.npz', fit, args.target_device)}


def run_group(args, manifest, model, mode, device):
    contexts = {seed: make_context(args, model, mode, device, seed) for seed in [0, 1, 2]}
    normal_rows = [r for split in ['fit', 'calibration'] for r in manifest
                   if r['device'] == device and r.get('split') == split and r['partition'] == 'training']
    normal_sequences = [FeatureSequence(args.cache / model / mode, row) for row in normal_rows]
    fingerprints = [s.meta['fingerprint'] for s in normal_sequences]
    fits = [s for s in normal_sequences if s.row['split'] == 'fit']
    calibration = [s for s in normal_sequences if s.row['split'] == 'calibration']
    if not fits or not calibration:
        raise ValueError('Missing normal fit/calibration videos')
    for context in contexts.values():
        if (fingerprints != context['fit']['normal_cache_fingerprints'] or
            any(s.identity != context['fit']['cache_identity'] for s in normal_sequences)):
            raise ValueError('Frozen normal cache provenance differs')
        np.testing.assert_allclose(context['fit']['cycle_length_fit_median'],
                                   np.median([s.row['frames'] for s in fits]), rtol=0, atol=0)
    normal_pairs = {seed: {k: [] for k in NEIGHBOURS} for seed in contexts}
    normal_saved = {seed: [] for seed in contexts}
    proof = {seed: {'status': 'passed', 'backbone': model, 'mode': mode, 'device': device, 'seed': seed,
        'K5_feature_max_absolute_error': 0., 'K5_score_max_absolute_error': 0., 'K5_alarm_mismatches_original_mask': 0,
        'unchanged_time_max_absolute_error': 0.,
        'raw_tolerance': RAW_TOLERANCE, 'score_tolerance': SCORE_TOLERANCE,
        'memory_sha256': c['memory_sha256'], 'phase_metadata_sha256': c['phase_metadata_sha256'],
        'source_evaluator_sha256': c['fit']['code_sha256'],
        'source_evaluator_git_revision': FROZEN_EVALUATOR_REVISIONS[c['fit']['code_sha256']],
        'cache_checks': [], 'sources': c['sources'],
        'scope': 'Existing frozen PCA/prototype bank and cached phase predictions; K5 search/score/alarm replay on original common mask, no independent encoder re-extraction or isolated runtime'}
        for seed, c in contexts.items()}
    for sequence in calibration:
        predictions, saved = {}, {}
        for seed, c in contexts.items():
            rows = [r for r in c['normal_rows'] if r['sequence'] == sequence.row['sequence']]
            saved[seed] = decode_inference(rows, sequence.row['frames'], mode, 'predicted_phase')
            np.testing.assert_array_equal(sequence.targets, saved[seed].targets)
            keep = common_mask(sequence.row['frames'])[sequence.targets]
            np.testing.assert_array_equal([int(r['valid']) for r in rows], keep)
            predictions[seed] = saved[seed].phase
            if {r['sequence'] for r in c['normal_rows']} != {s.row['sequence'] for s in calibration}:
                raise ValueError('Normal calibration video inventory differs')
        features = score_cached_sequences(sequence, contexts, predictions, args.target_device)
        for seed, c in contexts.items():
            np.testing.assert_allclose(features[seed][5], saved[seed].feature, **RAW_TOLERANCE)
            proof[seed]['K5_feature_max_absolute_error'] = max(proof[seed]['K5_feature_max_absolute_error'],
                float(np.max(np.abs(features[seed][5] - saved[seed].feature))))
            normal_saved[seed].append(saved[seed])
            for k in NEIGHBOURS:
                normal_pairs[seed][k].append(infer_pairs(saved[seed], features[seed][k], c['fit']['cycle_length_fit_median']))
            old_time = np.array([float(r['time_raw']) for r in c['normal_rows'] if r['sequence'] == sequence.row['sequence']])
            actual_time = normal_pairs[seed][5][-1][:, 1]
            np.testing.assert_allclose(actual_time, old_time, rtol=0, atol=1e-12, equal_nan=True)
            finite = np.isfinite(old_time)
            proof[seed]['unchanged_time_max_absolute_error'] = max(proof[seed]['unchanged_time_max_absolute_error'],
                float(np.max(np.abs(actual_time[finite] - old_time[finite]))))
            proof[seed]['cache_checks'].append({'partition': 'training', 'sequence': sequence.row['sequence'],
                'fingerprint': sequence.meta['fingerprint'], 'metadata_sha256': digest(sequence.folder / 'meta.json')})
        print(f"neighbour normal: {model}/{mode}/{device}/{sequence.row['sequence']} (3 seeds)", flush=True)
    calibrators = {seed: {k: recalibrate(values, normal_saved[seed]) for k, values in group.items()}
                   for seed, group in normal_pairs.items()}
    for seed, c in contexts.items():
        old = c['fit']['calibration']['P3']; cal = calibrators[seed][5]
        np.testing.assert_allclose(cal.median, old['median'], **RAW_TOLERANCE)
        np.testing.assert_allclose(cal.scale, old['mad_scale'], **RAW_TOLERANCE)
        np.testing.assert_allclose(cal.threshold, old['threshold'], **SCORE_TOLERANCE)
        for index, sequence in enumerate(calibration):
            pairs = normal_pairs[seed][5][index]
            keep = common_mask(sequence.row['frames'])[sequence.targets]
            expected = [float(r['P3']) for r in c['normal_rows'] if r['sequence'] == sequence.row['sequence']]
            actual = cal.combine(pairs)
            np.testing.assert_allclose(actual[keep], np.array(expected)[keep], **SCORE_TOLERANCE)
            proof[seed]['K5_score_max_absolute_error'] = max(proof[seed]['K5_score_max_absolute_error'],
                float(np.max(np.abs(actual[keep] - np.array(expected)[keep]))))
    destinations = {seed: {k: args.out / f'K{k}' / model / mode / device / f'seed{seed}' for k in NEIGHBOURS}
                    for seed in contexts}
    for seed, group in destinations.items():
        for k, folder in group.items():
            if folder.exists():
                raise FileExistsError('Never overwrite an existing neighbour condition')
            (folder / 'P3').mkdir(parents=True)
            c = contexts[seed]; cal = calibrators[seed][k]
            atomic_json(folder / 'normal_fit.json', {'status': 'normal_fit_and_calibration_complete',
                'backbone': model, 'mode': mode, 'device': device, 'seed': seed, 'neighbours': k,
                'calibration_mask': 't=19..N-8 inclusive', 'history': 5, 'score_weights': [.5, .5],
                'cycle_length_fit_median': c['fit']['cycle_length_fit_median'],
                'temperature': c['fit']['temperature'], 'temperature_definition': 'unchanged normal median fifth-neighbour distance',
                'source_memory_sha256': c['memory_sha256'], 'source_cache_identity': c['fit']['cache_identity'],
                'source_normal_fit_sha256': c['sources']['normal_fit.json'],
                'source_phase_checkpoint_sha256': c['fit']['phase_checkpoint_sha256'],
                'calibration': {'P3': {'median': cal.median.tolist(), 'mad_scale': cal.scale.tolist(),
                    'threshold': cal.threshold, 'component_thresholds': cal.component_thresholds.tolist()}},
                'code_sha256': digest(__file__), 'search_code_sha256': digest(Path(__file__).with_name('torch_memory.py')),
                'temporal_code_sha256': digest(Path(__file__).with_name('temporal.py')),
                'unchanged_components': ['frozen_encoder', 'phase_head', 'PCA', 'prototypes', 'temperature', 'time_history5'],
                'scope': 'Neighbour-only OFAT; normal-only recalibration before testing; cached features, not real-time throughput'})
            with (folder / 'normal_calibration.csv').open('w', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(['sequence', 'frame', 'valid', 'relative_phase', 'predicted_phase', 'feature_raw', 'time_raw', 'P3'])
                for sequence, seq, pairs in zip(calibration, normal_saved[seed], normal_pairs[seed][k]):
                    keep = common_mask(seq.frames)[seq.targets]
                    writer.writerows(zip([sequence.row['sequence']] * len(seq.targets), seq.targets, keep.astype(int),
                        seq.targets / seq.frames, seq.phase, pairs[:, 0], pairs[:, 1], cal.combine(pairs)))
    results = {seed: {k: {'labels': [], 'scores': []} for k in NEIGHBOURS} for seed in contexts}
    tests = [r for r in manifest if r['device'] == device and r['partition'] == 'testing']
    for c in contexts.values():
        if {p.stem for p in (c['source'] / 'P3').glob('*.csv')} != {r['sequence'] for r in tests}:
            raise ValueError('Source test inventory differs')
    # No test score CSV or annotation is opened until every normal calibration above is fixed.
    for row in tests:
        sequence = FeatureSequence(args.cache / model / mode, row)
        predictions, saved, annotations = {}, {}, {}
        for seed, c in contexts.items():
            if sequence.identity != c['fit']['cache_identity']:
                raise ValueError('Test cache differs from frozen normal identity')
            path = c['source'] / 'P3' / f"{row['sequence']}.csv"
            rows = read_csv(path); c['sources'][f"P3/{row['sequence']}.csv"] = digest(path)
            saved[seed] = decode_inference(rows, row['frames'], mode)
            np.testing.assert_array_equal(sequence.targets, saved[seed].targets)
            predictions[seed] = saved[seed].phase
            annotations[seed] = rows
        features = score_cached_sequences(sequence, contexts, predictions, args.target_device)
        label_path = args.data_root / row['label_file']
        if digest(label_path) != row['label_sha256']:
            raise ValueError('Audited test annotation changed')
        actual_labels, known, _ = align(np.load(label_path, allow_pickle=False), row['frames'])
        for seed, c in contexts.items():
            seq = saved[seed]; rows = annotations[seed]
            np.testing.assert_allclose(features[seed][5], seq.feature, **RAW_TOLERANCE)
            proof[seed]['K5_feature_max_absolute_error'] = max(proof[seed]['K5_feature_max_absolute_error'],
                float(np.max(np.abs(features[seed][5] - seq.feature))))
            old_ready = common_mask(seq.frames)[seq.targets]
            labels = np.array([float(r['label']) for r in rows])
            if not np.isin(labels, [-1, 0, 1]).all():
                raise ValueError('Unexpected binary annotation values')
            valid = old_ready & (labels >= 0)
            np.testing.assert_array_equal([int(r['inference_valid']) for r in rows], old_ready)
            np.testing.assert_array_equal(labels, actual_labels[seq.targets])
            np.testing.assert_array_equal(valid, old_ready & known[seq.targets])
            np.testing.assert_array_equal([int(r['valid']) for r in rows], valid)
            for k in NEIGHBOURS:
                pairs = infer_pairs(seq, features[seed][k], c['fit']['cycle_length_fit_median'])
                cal = calibrators[seed][k]
                scores, ready, alarms, types = scored(seq, pairs, cal)
                if k == 5:
                    old_time = np.array([float(r['time_raw']) for r in rows])
                    np.testing.assert_allclose(pairs[:, 1], old_time, rtol=0, atol=1e-12, equal_nan=True)
                    finite = np.isfinite(old_time)
                    proof[seed]['unchanged_time_max_absolute_error'] = max(proof[seed]['unchanged_time_max_absolute_error'],
                        float(np.max(np.abs(pairs[finite, 1] - old_time[finite]))))
                    expected = np.array([float(r['score']) for r in rows])
                    np.testing.assert_allclose(scores[old_ready], expected[old_ready], **SCORE_TOLERANCE)
                    np.testing.assert_array_equal(alarm_flags(scores, old_ready, cal.threshold), [int(r['alarm']) for r in rows])
                    proof[seed]['K5_score_max_absolute_error'] = max(proof[seed]['K5_score_max_absolute_error'],
                        float(np.max(np.abs(scores[old_ready] - expected[old_ready]))))
                results[seed][k]['labels'].append(labels[valid].astype(np.int8))
                results[seed][k]['scores'].append(scores[valid])
                with (destinations[seed][k] / 'P3' / f"{row['sequence']}.csv").open('w', newline='') as stream:
                    writer = csv.writer(stream, lineterminator='\n')
                    writer.writerow(['frame', 'label', 'valid', 'inference_valid', 'phase', 'feature_raw', 'time_raw', 'score', 'alarm', 'evidence_type'])
                    writer.writerows(zip(seq.targets, labels, valid.astype(int), ready.astype(int), seq.phase,
                        pairs[:, 0], pairs[:, 1], scores, alarms.astype(int), types))
            proof[seed]['cache_checks'].append({'partition': 'testing', 'sequence': row['sequence'],
                'fingerprint': sequence.meta['fingerprint'], 'metadata_sha256': digest(sequence.folder / 'meta.json')})
        print(f"neighbour test: {model}/{mode}/{device}/{row['sequence']} (3 seeds, k1/5/10)", flush=True)
    for seed, group in destinations.items():
        for k, folder in group.items():
            atomic_json(folder / 'source_replay_check.json', proof[seed])
            r = results[seed][k]
            atomic_json(folder / 'metrics.json', {'status': 'complete_device_evaluation', 'backbone': model,
                'mode': mode, 'device': device, 'seed': seed, 'neighbours': k, 'test_videos': len(tests),
                'valid_mask': 't=19..N-8 inclusive, intersect original shared annotation uncertainty mask',
                'variants': {'P3': metric(np.concatenate(r['labels']), np.concatenate(r['scores']))},
                'note': 'Neighbour-only frozen feature search; online EOF alarms active; runtime not isolated or measured'})
    print(f"neighbour group complete: {model}/{mode}/{device}; 9 conditions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/stage02'))
    parser.add_argument('--local', type=Path, default=Path('artifacts/runs'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/features'))
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path, default=Path('results/stage05/ablations/neighbours'))
    parser.add_argument('--model', choices=['dinov3-l', 'vjepa21-l'])
    parser.add_argument('--mode', choices=['offline', 'online'])
    parser.add_argument('--device', choices=['R01', 'R02', 'R03', 'R04'])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output root required; inspect partial/live jobs before recovery')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for the accepted FP32 memory search')
    torch.backends.cuda.matmul.allow_tf32 = False
    args.target_device = 'cuda'
    supplied = [args.model is not None, args.mode is not None, args.device is not None]
    if any(supplied) and not all(supplied):
        raise ValueError('Provide model/mode/device together, or run the full matrix')
    groups = [(args.model, args.mode, args.device)] if all(supplied) else [
        (m, mode, d) for m in ['dinov3-l', 'vjepa21-l'] for mode in ['offline', 'online'] for d in ['R01', 'R02', 'R03', 'R04']]
    manifest = json.loads(args.manifest.read_text())['sequences']
    summary = json.loads((args.source / 'device_summary.json').read_text())
    available = {(r['backbone'], r['mode'], r['device']) for r in summary['results'] if r['seeds'] == 3}
    if not set(groups).issubset(available):
        raise ValueError('Required complete frozen source groups are missing')
    for model, mode, device in groups:
        run_group(args, manifest, model, mode, device)
    atomic_json(args.out / 'completion.json', {'status': 'complete_neighbour_matrix' if len(groups) == 16 else 'complete_neighbour_group',
        'groups': len(groups), 'conditions': len(groups)*3, 'evaluated_seed_conditions': len(groups)*9,
        'neighbours': list(NEIGHBOURS), 'code_sha256': digest(__file__),
        'source_summary_sha256': digest(args.source / 'device_summary.json'), 'manifest_sha256': digest(args.manifest),
        'scope': 'Frozen neighbour OFAT; source replay and metrics complete; independent aggregation/publication checks still required'})


if __name__ == '__main__':
    main()
