"""Normal-only prototype budget/bin OFAT, with rebuilt default-bank replay gates."""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from ipad_jepa.alignment import align
from ipad_jepa.cache_data import FeatureSequence, normal_candidates
from ipad_jepa.experiment import temperature_sample
from ipad_jepa.history_ablation import alarm_flags, atomic_json, decode_inference, metric, read_csv
from ipad_jepa.memory import PrototypeMemory
from ipad_jepa.neighbour_ablation import (RAW_TOLERANCE, SCORE_TOLERANCE, digest,
    infer_pairs, make_context, recalibrate, scored)
from ipad_jepa.temporal import common_mask
from ipad_jepa.torch_memory import TorchMemory

CONFIGURATIONS = {'B16_M1024': (16, 1024), 'B16_M2048': (16, 2048),
                  'B16_M4096': (16, 4096), 'B8_M2048': (8, 2048)}
BASE = 'B16_M2048'


def prefix_memory(memory, per_bin):
    """A k-center prefix preserves the same candidate sample and RNG initial centre."""
    if per_bin < 1 or per_bin > memory.per_bin:
        raise ValueError('Requested prototype prefix exceeds fitted memory')
    result = replace(memory, per_bin=per_bin, temperature=1.)
    result.pca = memory.pca
    result.prototypes = memory.prototypes[:, :per_bin].copy(order='K')
    return result


def rebuild_memories(fits, seed, original_bank):
    """Rebuild normal candidates/PCA; default arrays must equal the saved bank exactly."""
    memories, sampling = {}, {}
    for bins in [16, 8]:
        x, phase, groups, sampled = normal_candidates(fits, bins=bins, seed=seed)
        largest = PrototypeMemory(bins=bins, per_bin=256, seed=seed).fit(x, phase, groups)
        del x, phase, groups
        sampling[str(bins)] = sampled
        for label, (b, budget) in CONFIGURATIONS.items():
            if b == bins:
                memories[label] = prefix_memory(largest, budget // bins)
        if bins == 16:
            base = memories[BASE]
            with np.load(original_bank, allow_pickle=False) as saved:
                for name, array in [('mean', base.pca.mean_), ('components', base.pca.components_),
                                    ('prototypes', base.prototypes)]:
                    np.testing.assert_array_equal(array, saved[name], err_msg=f'Rebuilt default {name} differs')
                    if array.dtype != saved[name].dtype or array.strides != saved[name].strides:
                        raise ValueError(f'Rebuilt default {name} layout/dtype differs')
            print(f'memory default arrays replayed exactly: seed{seed}', flush=True)
    return memories, sampling


def search(sequence, contexts, predictions, batch=16):
    output = {seed: {label: [] for label in CONFIGURATIONS} for seed in contexts}
    with torch.inference_mode():
        for start in range(0, len(sequence.targets), batch):
            patches = torch.from_numpy(np.array(sequence.patches[start:start + batch], dtype=np.float32)).cuda()
            for seed, c in contexts.items():
                phase = torch.from_numpy(predictions[seed][start:start + batch].astype(np.float32)).cuda()
                for label, scorer in c['scorers'].items():
                    output[seed][label].append(scorer.frame_scores(patches, phase, k=5).cpu().numpy())
    return {s: {k: np.concatenate(v) for k, v in settings.items()} for s, settings in output.items()}


def write_trace(path, values, fields):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(fields)
        writer.writerows(zip(*values))


def run_group(args, manifest, model, mode, device):
    contexts = {seed: make_context(args, model, mode, device, seed) for seed in [0, 1, 2]}
    rows = [r for split in ['fit', 'calibration'] for r in manifest
            if r['device'] == device and r['partition'] == 'training' and r.get('split') == split]
    sequences = [FeatureSequence(args.cache / model / mode, row) for row in rows]
    fits = [s for s in sequences if s.row['split'] == 'fit']
    calibration = [s for s in sequences if s.row['split'] == 'calibration']
    if not fits or not calibration:
        raise ValueError('Missing normal fit/calibration sequences')
    for seed, c in contexts.items():
        if ([s.meta['fingerprint'] for s in sequences] != c['fit']['normal_cache_fingerprints'] or
            any(s.identity != c['fit']['cache_identity'] for s in sequences) or
            c['fit']['cycle_length_fit_median'] != float(np.median([s.row['frames'] for s in fits]))):
            raise ValueError('Normal cache provenance differs')
        # The original scorer is used only to validate provenance in make_context.
        # Every experimental scorer below uses an actually rebuilt normal bank.
        del c['scorer']
        bank = args.local / model / mode / device / f'seed{seed}' / 'memory.npz'
        memories, sampling = rebuild_memories(fits, seed, bank)
        c['memories'], c['candidate_sampling'] = memories, sampling
        c['scorers'] = {label: TorchMemory(memory).cuda().eval() for label, memory in memories.items()}
        c['normal_saved'] = [decode_inference(
            [r for r in c['normal_rows'] if r['sequence'] == s.row['sequence']],
            s.row['frames'], mode, 'predicted_phase') for s in calibration]
        for s, saved in zip(calibration, c['normal_saved']):
            np.testing.assert_array_equal(s.targets, saved.targets)
        if {r['sequence'] for r in c['normal_rows']} != {s.row['sequence'] for s in calibration}:
            raise ValueError('Calibration inventory differs')
        for label, scorer in c['scorers'].items():
            temperature, count = temperature_sample(calibration, [s.phase for s in c['normal_saved']], scorer, seed)
            c['memories'][label].temperature = scorer.temperature = temperature
            if label == BASE and (temperature != c['fit']['temperature'] or count != c['fit']['temperature_samples']):
                raise ValueError('Rebuilt default normal temperature differs')
            c.setdefault('temperature_samples', {})[label] = count
            folder = args.memory_out / label / model / mode / device / f'seed{seed}'
            folder.mkdir(parents=True, exist_ok=False)
            memory = c['memories'][label]
            np.savez(folder / 'memory.npz', mean=memory.pca.mean_, components=memory.pca.components_,
                prototypes=memory.prototypes, temperature=temperature, cycle_length=c['fit']['cycle_length_fit_median'])
            c.setdefault('bank_hashes', {})[label] = digest(folder / 'memory.npz')
        c['proof'] = {'status': 'passed', 'backbone': model, 'mode': mode, 'device': device, 'seed': seed,
            'default_array_replay': 'exact mean/components/prototypes, dtype and strides',
            'default_temperature_replay': 'exact normal calibration fifth-neighbour median and sample count',
            'default_feature_max_absolute_error': 0., 'default_score_max_absolute_error': 0.,
            'unchanged_time_max_absolute_error': 0., 'default_alarm_mismatches_original_mask': 0,
            'raw_tolerance': RAW_TOLERANCE, 'score_tolerance': SCORE_TOLERANCE,
            'source_memory_sha256': c['memory_sha256'], 'phase_metadata_sha256': c['phase_metadata_sha256'],
            'sources': c['sources'], 'cache_checks': [{'partition': s.row['partition'],
                'sequence': s.row['sequence'], 'fingerprint': s.meta['fingerprint'],
                'metadata_sha256': digest(s.folder / 'meta.json')} for s in sequences],
            'scope': 'Actual normal bank rebuild and cached search; default source replay; no encoder re-extraction or runtime'}
    normal_pairs = {s: {k: [] for k in CONFIGURATIONS} for s in contexts}
    for index, sequence in enumerate(calibration):
        predictions = {s: c['normal_saved'][index].phase for s, c in contexts.items()}
        features = search(sequence, contexts, predictions)
        for seed, c in contexts.items():
            saved = c['normal_saved'][index]
            np.testing.assert_allclose(features[seed][BASE], saved.feature, **RAW_TOLERANCE)
            c['proof']['default_feature_max_absolute_error'] = max(c['proof']['default_feature_max_absolute_error'],
                float(np.max(np.abs(features[seed][BASE] - saved.feature))))
            for label in CONFIGURATIONS:
                normal_pairs[seed][label].append(infer_pairs(saved, features[seed][label], c['fit']['cycle_length_fit_median']))
            before = np.array([float(r['time_raw']) for r in c['normal_rows'] if r['sequence'] == sequence.row['sequence']])
            after = normal_pairs[seed][BASE][-1][:, 1]
            np.testing.assert_allclose(after, before, rtol=0, atol=1e-12, equal_nan=True)
            finite = np.isfinite(before)
            c['proof']['unchanged_time_max_absolute_error'] = max(c['proof']['unchanged_time_max_absolute_error'],
                float(np.max(np.abs(after[finite] - before[finite]))))
        print(f'memory calibration: {model}/{mode}/{device}/{sequence.row["sequence"]}', flush=True)
    calibrators = {seed: {label: recalibrate(pairs, contexts[seed]['normal_saved']) for label, pairs in settings.items()}
                   for seed, settings in normal_pairs.items()}
    destinations = {}
    for seed, c in contexts.items():
        destinations[seed] = {}
        for label, (bins, budget) in CONFIGURATIONS.items():
            cal = calibrators[seed][label]
            if label == BASE:
                old = c['fit']['calibration']['P3']
                np.testing.assert_allclose(cal.median, old['median'], **RAW_TOLERANCE)
                np.testing.assert_allclose(cal.scale, old['mad_scale'], **RAW_TOLERANCE)
                np.testing.assert_allclose(cal.threshold, old['threshold'], **SCORE_TOLERANCE)
            folder = args.out / label / model / mode / device / f'seed{seed}'
            (folder / 'P3').mkdir(parents=True, exist_ok=False)
            destinations[seed][label] = folder
            memory = c['memories'][label]
            atomic_json(folder / 'normal_fit.json', {'status': 'normal_fit_and_calibration_complete',
                'backbone': model, 'mode': mode, 'device': device, 'seed': seed, 'configuration': label,
                'bins': bins, 'total_prototypes': budget, 'prototypes_per_bin': budget // bins,
                'pca_dimensions': 256, 'pca_samples': 50000,
                'pca_explained_variance_ratio': float(memory.pca.explained_variance_ratio_.sum()),
                'candidate_sampling': c['candidate_sampling'][str(bins)],
                'calibration_mask': 't=19..N-8 inclusive', 'history': 5, 'neighbours': 5, 'score_weights': [.5, .5],
                'cycle_length_fit_median': c['fit']['cycle_length_fit_median'],
                'temperature': memory.temperature, 'temperature_samples': c['temperature_samples'][label],
                'temperature_definition': 'Per-bank normal calibration median fifth squared-neighbour distance; floor1e-6',
                'bank_sha256': c['bank_hashes'][label], 'source_memory_sha256': c['memory_sha256'],
                'source_cache_identity': c['fit']['cache_identity'],
                'source_normal_fit_sha256': c['sources']['normal_fit.json'],
                'source_phase_checkpoint_sha256': c['fit']['phase_checkpoint_sha256'],
                'calibration': {'P3': {'median': cal.median.tolist(), 'mad_scale': cal.scale.tolist(),
                    'threshold': cal.threshold, 'component_thresholds': cal.component_thresholds.tolist()}},
                'code_sha256': digest(__file__), 'memory_code_sha256': digest(Path(__file__).with_name('memory.py')),
                'candidate_code_sha256': digest(Path(__file__).with_name('cache_data.py')),
                'temperature_code_sha256': digest(Path(__file__).with_name('experiment.py')),
                'search_code_sha256': digest(Path(__file__).with_name('torch_memory.py')),
                'temporal_code_sha256': digest(Path(__file__).with_name('temporal.py')),
                'unchanged_components': ['frozen_encoder', 'phase_head', 'clip16', 'time_history5', 'P3_weights'],
                'sampling_policy': 'Budget comparison shares candidates and PCA; bin comparison refits phase-stratified normal candidates/PCA. K-center prefixes have the same RNG initial centres as separate fits.',
                'scope': 'Normal bank geometry OFAT and per-bank normal-only recalibration before test scoring'})
            normal_columns = [[] for _ in range(8)]
            for index, sequence in enumerate(calibration):
                saved = c['normal_saved'][index]; pairs = normal_pairs[seed][label][index]
                keep = common_mask(saved.frames)[saved.targets]
                rows = [r for r in c['normal_rows'] if r['sequence'] == sequence.row['sequence']]
                np.testing.assert_array_equal([int(r['valid']) for r in rows], keep)
                values = cal.combine(pairs)
                if label == BASE:
                    old = np.array([float(r['P3']) for r in rows])
                    np.testing.assert_allclose(values[keep], old[keep], **SCORE_TOLERANCE)
                    c['proof']['default_score_max_absolute_error'] = max(c['proof']['default_score_max_absolute_error'],
                        float(np.max(np.abs(values[keep] - old[keep]))))
                for column, value in zip(normal_columns, [[sequence.row['sequence']] * len(saved.targets),
                    saved.targets, keep.astype(int), saved.targets / saved.frames, saved.phase, pairs[:, 0], pairs[:, 1], values]):
                    column.extend(value)
            write_trace(folder / 'normal_calibration.csv', normal_columns,
                ['sequence', 'frame', 'valid', 'relative_phase', 'predicted_phase', 'feature_raw', 'time_raw', 'P3'])
    results = {s: {k: {'labels': [], 'scores': []} for k in CONFIGURATIONS} for s in contexts}
    tests = [r for r in manifest if r['device'] == device and r['partition'] == 'testing']
    for c in contexts.values():
        if {p.stem for p in (c['source'] / 'P3').glob('*.csv')} != {r['sequence'] for r in tests}:
            raise ValueError('Source test inventory differs')
    # All banks and normal calibration parameters above are fixed before reading test scores or GT.
    for row in tests:
        sequence = FeatureSequence(args.cache / model / mode, row)
        saved, traces = {}, {}
        for seed, c in contexts.items():
            if sequence.identity != c['fit']['cache_identity']:
                raise ValueError('Test cache identity differs')
            path = c['source'] / 'P3' / f'{row["sequence"]}.csv'
            traces[seed] = read_csv(path); c['sources'][f'P3/{row["sequence"]}.csv'] = digest(path)
            saved[seed] = decode_inference(traces[seed], row['frames'], mode)
            np.testing.assert_array_equal(sequence.targets, saved[seed].targets)
        features = search(sequence, contexts, {s: v.phase for s, v in saved.items()})
        label_file = args.data_root / row['label_file']
        if digest(label_file) != row['label_sha256']:
            raise ValueError('Raw test annotation changed')
        labels, known, _ = align(np.load(label_file, allow_pickle=False), row['frames'])
        for seed, c in contexts.items():
            seq, trace = saved[seed], traces[seed]
            old_ready = common_mask(seq.frames)[seq.targets]
            valid = old_ready & known[seq.targets]
            np.testing.assert_array_equal([int(float(r['label'])) for r in trace], labels[seq.targets])
            np.testing.assert_array_equal([int(r['valid']) for r in trace], valid)
            np.testing.assert_array_equal([int(r['inference_valid']) for r in trace], old_ready)
            np.testing.assert_allclose(features[seed][BASE], seq.feature, **RAW_TOLERANCE)
            c['proof']['default_feature_max_absolute_error'] = max(c['proof']['default_feature_max_absolute_error'],
                float(np.max(np.abs(features[seed][BASE] - seq.feature))))
            for label in CONFIGURATIONS:
                pairs = infer_pairs(seq, features[seed][label], c['fit']['cycle_length_fit_median'])
                cal = calibrators[seed][label]
                values, ready, flags, types = scored(seq, pairs, cal)
                before = np.array([float(r['time_raw']) for r in trace])
                np.testing.assert_allclose(pairs[:, 1], before, rtol=0, atol=1e-12, equal_nan=True)
                finite = np.isfinite(before)
                c['proof']['unchanged_time_max_absolute_error'] = max(c['proof']['unchanged_time_max_absolute_error'],
                    float(np.max(np.abs(pairs[finite, 1] - before[finite]))))
                if label == BASE:
                    old = np.array([float(r['score']) for r in trace])
                    np.testing.assert_allclose(values[old_ready], old[old_ready], **SCORE_TOLERANCE)
                    np.testing.assert_array_equal(alarm_flags(values, old_ready, cal.threshold), [int(r['alarm']) for r in trace])
                    c['proof']['default_score_max_absolute_error'] = max(c['proof']['default_score_max_absolute_error'],
                        float(np.max(np.abs(values[old_ready] - old[old_ready]))))
                write_trace(destinations[seed][label] / 'P3' / f'{row["sequence"]}.csv',
                    [seq.targets, labels[seq.targets], valid.astype(int), ready.astype(int), seq.phase,
                     pairs[:, 0], pairs[:, 1], values, flags.astype(int), types],
                    ['frame', 'label', 'valid', 'inference_valid', 'phase', 'feature_raw', 'time_raw', 'score', 'alarm', 'evidence_type'])
                results[seed][label]['labels'].append(labels[seq.targets][valid])
                results[seed][label]['scores'].append(values[valid])
            c['proof']['cache_checks'].append({'partition': 'testing', 'sequence': row['sequence'],
                'fingerprint': sequence.meta['fingerprint'], 'metadata_sha256': digest(sequence.folder / 'meta.json')})
        print(f'memory test: {model}/{mode}/{device}/{row["sequence"]}', flush=True)
    for seed, c in contexts.items():
        for label, folder in destinations[seed].items():
            atomic_json(folder / 'source_replay_check.json', c['proof'])
            result = results[seed][label]
            atomic_json(folder / 'metrics.json', {'status': 'complete_device_evaluation', 'backbone': model,
                'mode': mode, 'device': device, 'seed': seed, 'configuration': label,
                'bins': CONFIGURATIONS[label][0], 'total_prototypes': CONFIGURATIONS[label][1],
                'test_videos': len(tests), 'valid_mask': 't=19..N-8 inclusive, intersect shared annotation uncertainty mask',
                'variants': {'P3': metric(np.concatenate(result['labels']), np.concatenate(result['scores']))},
                'note': 'Normal-memory OFAT; online EOF alarms remain active; no isolated runtime measurement'})
    print(f'memory group complete: {model}/{mode}/{device}; 12 seed conditions', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('source', 'results/stage02'), ('local', 'artifacts/runs'),
                          ('cache', 'artifacts/features'), ('manifest', 'results/stage00/manifest.json'),
                          ('data-root', '../IPAD_dataset/IPAD_dataset'),
                          ('out', 'results/stage05/ablations/memory'), ('memory-out', 'artifacts/ablations/memory')]:
        parser.add_argument('--' + name, type=Path, default=Path(default))
    parser.add_argument('--model', choices=['dinov3-l', 'vjepa21-l'])
    parser.add_argument('--mode', choices=['offline', 'online'])
    parser.add_argument('--device', choices=['R01', 'R02', 'R03', 'R04'])
    args = parser.parse_args()
    if args.out.exists() or args.memory_out.exists():
        raise FileExistsError('Fresh public and private output roots required; never overwrite a partial/live job')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for the accepted FP32 memory search')
    torch.backends.cuda.matmul.allow_tf32 = False
    args.target_device = 'cuda'
    supplied = [args.model is not None, args.mode is not None, args.device is not None]
    if any(supplied) and not all(supplied):
        raise ValueError('Provide model/mode/device together, or run the full matrix')
    groups = [(args.model, args.mode, args.device)] if all(supplied) else [
        (m, mode, d) for m in ['dinov3-l', 'vjepa21-l'] for mode in ['offline', 'online'] for d in ['R01', 'R02', 'R03', 'R04']]
    available = {(r['backbone'], r['mode'], r['device']) for r in
                 json.loads((args.source / 'device_summary.json').read_text())['results'] if r['seeds'] == 3}
    if not set(groups).issubset(available):
        raise ValueError('Required complete frozen source groups are missing')
    manifest = json.loads(args.manifest.read_text())['sequences']
    for model, mode, device in groups:
        run_group(args, manifest, model, mode, device)
    atomic_json(args.out / 'completion.json', {'status': 'complete_memory_matrix' if len(groups) == 16 else 'complete_memory_group',
        'groups': len(groups), 'configurations': CONFIGURATIONS, 'conditions': len(groups) * 4,
        'evaluated_seed_conditions': len(groups) * 12, 'code_sha256': digest(__file__),
        'source_summary_sha256': digest(args.source / 'device_summary.json'), 'manifest_sha256': digest(args.manifest),
        'scope': 'Normal-only bank rebuild, per-bank temperature/calibration and cached search; independent aggregation/publication checks required'})


if __name__ == '__main__':
    main()
