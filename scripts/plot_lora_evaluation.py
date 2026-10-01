"""Audit and plot a completed LoRA seed against its paired frozen evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from ipad_jepa.adapted_features import selected_protocol
from ipad_jepa.alignment import align
from ipad_jepa.cache_data import FeatureSequence
from ipad_jepa.temporal import temporal_score
from summarize_experiments import load_run, check_pair, bootstrap_draws

VARIANTS = ['P0', 'P1', 'P2', 'P3']
COLORS = {'frozen': '#245B86', 'lora': '#B97812'}
LABELS = {'frozen': 'Frozen', 'lora': 'LoRA'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def calibration_check(folder, info, manifest, data_root):
    normal = json.loads((folder / 'normal_fit.json').read_text())
    if (normal['status'] != 'normal_fit_and_calibration_complete' or
        any(normal[k] != info[k] for k in ['backbone', 'mode', 'device', 'seed'])):
        raise ValueError('Normal fit condition differs')
    calibration = {r['sequence']: r for r in manifest if r['device'] == info['device'] and r.get('split') == 'calibration'}
    rows = read_csv(folder / 'normal_calibration.csv')
    keys = [(r['sequence'], int(r['frame'])) for r in rows]
    if len(set(keys)) != len(keys) or {r['sequence'] for r in rows} != calibration.keys():
        raise ValueError('Normal calibration inventory differs')
    for key, row in calibration.items():
        first = 15 if info['mode'] == 'online' else 8
        last = row['frames'] - 1 if info['mode'] == 'online' else row['frames'] - 8
        actual = [int(r['frame']) for r in rows if r['sequence'] == key]
        np.testing.assert_array_equal(actual, np.arange(first, last + 1))
    for r in rows:
        keep = 19 <= int(r['frame']) <= calibration[r['sequence']]['frames'] - 8
        if int(r['valid']) != int(keep):
            raise ValueError('Calibration common mask differs')
        if float(r['relative_phase']) != int(r['frame']) / calibration[r['sequence']]['frames']:
            raise ValueError('Normal pseudo-phase differs')
    selected = [r for r in rows if r['valid'] == '1']
    if len(selected) != normal['calibration_valid_frames']:
        raise ValueError('Normal calibration count differs')
    pairs = np.array([[float(r['feature_raw']), float(r['time_raw'])] for r in selected])
    median = np.median(pairs, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(pairs - median), axis=0), 1e-6)
    components = np.quantile(pairs, .99, axis=0)
    for v in ['P2', 'P3']:
        c = normal['calibration'][v]
        np.testing.assert_allclose(median, c['median'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(scale, c['mad_scale'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(components, c['component_thresholds'], rtol=0, atol=1e-9)
        z = (pairs - median) / scale
        expected = z.mean(axis=1) if v == 'P3' else z[:, 0]
        np.testing.assert_allclose(expected, [float(r[v]) for r in selected], rtol=0, atol=1e-9)
    sources = {name: digest(folder / name) for name in ['metrics.json', 'normal_fit.json', 'normal_calibration.csv']}
    test_rows = {r['sequence']: r for r in manifest if r['device'] == info['device'] and r['partition'] == 'testing'}
    for v in VARIANTS:
        c = normal['calibration'][v]
        np.testing.assert_allclose(np.quantile([float(r[v]) for r in selected], .99), c['threshold'], rtol=0, atol=1e-9)
        if {p.stem for p in (folder / v).glob('*.csv')} != test_rows.keys():
            raise ValueError('Test video inventory differs')
        for key, row in test_rows.items():
            source = folder / v / f'{key}.csv'
            trace = read_csv(source)
            frames = np.array([int(r['frame']) for r in trace])
            first = 15 if info['mode'] == 'online' else 8
            last = row['frames'] - 1 if info['mode'] == 'online' else row['frames'] - 8
            np.testing.assert_array_equal(frames, np.arange(first, last + 1))
            label_path = data_root / row['label_file']
            if digest(label_path) != row['label_sha256']:
                raise ValueError('Test annotation hash differs')
            labels, known, _ = align(np.load(label_path, allow_pickle=False), row['frames'])
            inference = (frames >= 19) & (frames <= row['frames'] - 8)
            np.testing.assert_array_equal([int(r['inference_valid']) for r in trace], inference)
            np.testing.assert_array_equal([int(r['valid']) for r in trace], inference & known[frames])
            np.testing.assert_array_equal([int(r['label']) for r in trace], labels[frames])
            raw = np.array([[float(r['feature_raw']), float(r['time_raw'])] for r in trace])
            dense = np.full(row['frames'], np.nan)
            dense[frames] = [float(r['phase']) for r in trace]
            np.testing.assert_allclose(temporal_score(dense, normal['cycle_length_fit_median'])[frames], raw[:, 1], rtol=0, atol=1e-9, equal_nan=True)
            z = (raw - np.array(c['median'])) / np.array(c['mad_scale'])
            scores = z.mean(axis=1) if v == 'P3' else z[:, 0]
            np.testing.assert_allclose(scores, [float(r['score']) for r in trace], rtol=0, atol=1e-9, equal_nan=True)
            types = np.zeros(len(trace), dtype=int)
            finite = np.isfinite(raw).all(axis=1)
            exceeds = raw[finite] > np.array(c['component_thresholds'])
            types[finite] = exceeds[:, 0].astype(int) + 2 * exceeds[:, 1].astype(int)
            np.testing.assert_array_equal(types, [int(r['evidence_type']) for r in trace])
            streak = 0
            for r in trace:
                streak = streak + 1 if r['inference_valid'] == '1' and float(r['score']) > c['threshold'] else 0
                if int(r['alarm']) != int(streak >= 3):
                    raise ValueError('Cached alarm differs from normal q99 and three consecutive exceedances')
            sources[str(source.relative_to(folder))] = digest(source)
    return normal, sources


def provenance(args, info, normal, manifest):
    training, checkpoint = selected_protocol(args.training)
    keys = ['backbone', 'mode', 'device', 'seed']
    if any(training[k] != info[k] for k in keys):
        raise ValueError('Selected training condition differs')
    selected_hash = digest(args.training / 'selected_adapter.pt')
    phase_path = args.local / 'phase_training.json'
    phase = json.loads(phase_path.read_text())
    if (phase['status'] != 'complete' or phase['selected_adapter_sha256'] != selected_hash or
        any(phase[k] != info[k] for k in keys) or phase['selected_epoch'] != training['selected_epoch'] or
        phase['cache_fingerprints'] != normal['normal_cache_fingerprints']):
        raise ValueError('Joint head/adapted normal provenance differs')
    head_path = args.local / 'phase_head.pt'
    head = torch.load(head_path, map_location='cpu', weights_only=True)
    if head.keys() != checkpoint['head'].keys() or any(not torch.equal(head[k], checkpoint['head'][k]) for k in head):
        raise ValueError('Joint selected head was replaced')
    if normal['phase_checkpoint_sha256'] != digest(head_path) or normal['phase_selected_epoch'] != training['selected_epoch']:
        raise ValueError('Normal memory was fitted with another head')
    adapter_identity = hashlib.sha256(json.dumps({'base': training['teacher_identity']['adapter_sha256'],
        'source': training['adaptation_sha256'], 'selected': selected_hash}, sort_keys=True).encode()).hexdigest()
    if normal['cache_identity']['adapter_sha256'] != adapter_identity:
        raise ValueError('Frozen or unrelated adapter cache used')
    if any(v != training['teacher_identity'][k] for k, v in normal['cache_identity'].items() if k != 'adapter_sha256'):
        raise ValueError('Adapted cache differs from fixed teacher base identity')
    code_root = Path('src/ipad_jepa')
    if (normal['code_sha256'] != digest(code_root / 'experiment.py') or
        normal['memory_code_sha256'] != digest(code_root / 'memory.py')):
        raise ValueError('Normal evaluator or memory source changed')
    # The evaluator concatenates fit first, then calibration; manifest order interleaves them.
    selected_rows = [r for split in ['fit', 'calibration', 'test'] for r in manifest
        if r['device'] == info['device'] and r.get('split', 'test') == split and
        r['partition'] == ('testing' if split == 'test' else 'training')]
    fingerprints, cache_checks = [], []
    for row in selected_rows:
        seq = FeatureSequence(args.cache, row)
        spec = seq.meta['spec']
        if (seq.identity != normal['cache_identity'] or spec['adaptation'] != 'qv_lora_last4' or
            spec['adaptation_seed'] != info['seed'] or spec['selected_adapter_sha256'] != selected_hash or
            spec['selected_epoch'] != training['selected_epoch'] or
            spec['adaptation_source_sha256'] != digest(code_root / 'adaptation.py') or
            spec['adapted_reader_sha256'] != digest(code_root / 'adapted_features.py')):
            raise ValueError('Adapted cache identity or condition differs')
        if row.get('split') in ['fit', 'calibration']:
            fingerprints.append(seq.meta['fingerprint'])
        cache_checks.append({'partition': row['partition'], 'sequence': row['sequence'],
            'split': row.get('split', 'test'), 'fingerprint': seq.meta['fingerprint'],
            'metadata_sha256': digest(seq.folder / 'meta.json'), 'clips': len(seq.targets)})
    if fingerprints != normal['normal_cache_fingerprints']:
        raise ValueError('Normal fit/calibration cache inventory differs')
    fit = [r for r in manifest if r['device'] == info['device'] and r.get('split') == 'fit']
    np.testing.assert_allclose(np.median([r['frames'] for r in fit]), normal['cycle_length_fit_median'], rtol=0, atol=0)
    fit_ids = {r['sequence'] for r in fit}
    sampling = normal['candidate_sampling']
    if (len(sampling) != 16 or [r['phase_bin'] for r in sampling] != list(range(16)) or
        any(set(r['per_video']) != fit_ids or sum(r['per_video'].values()) != r['tokens'] or
            not 128 <= r['tokens'] <= 10000 for r in sampling)):
        raise ValueError('Prototype candidates differ from normal fit inventory or budget')
    if set(training['teacher_fingerprints']) != set(phase['teacher_cache_fingerprints']):
        raise ValueError('Fixed teacher provenance changed')
    frozen_normal = json.loads((args.frozen_results / 'normal_fit.json').read_text())
    if set(training['teacher_fingerprints']) != set(frozen_normal['normal_cache_fingerprints']):
        raise ValueError('Teacher differs from the paired frozen normal caches')
    for name in ['lora_training.json', 'lora_training.csv']:
        if (args.results / name).read_bytes() != (args.training / name).read_bytes():
            raise ValueError('Public normal training export differs')
    bank = args.local / 'memory.npz'
    frozen_bank = args.frozen_local / 'memory.npz'
    with np.load(bank, allow_pickle=False) as a, np.load(frozen_bank, allow_pickle=False) as b:
        if a['prototypes'].shape != (16, 128, 256) or a['components'].shape != (256, 1024):
            raise ValueError('Rebuilt normal bank dimensions differ')
        if np.array_equal(a['prototypes'], b['prototypes']) or np.array_equal(a['components'], b['components']):
            raise ValueError('Frozen bank or PCA was reused')
        np.testing.assert_allclose(a['temperature'], normal['temperature'], rtol=0, atol=0)
    return {'status': 'passed', 'selected_epoch': training['selected_epoch'],
        'selected_adapter_sha256': selected_hash, 'joint_head_sha256': digest(head_path),
        'rebuilt_memory_sha256': digest(bank), 'frozen_memory_sha256': digest(frozen_bank),
        'phase_metadata_sha256': digest(phase_path), 'cache_checks': cache_checks,
        'scope': 'Selected normal-CE adapter/joint head equality, audited cache metadata/dimensions, rebuilt PCA/prototype distinction and CSV/calibration/GT replay; no independent raw-feature re-extraction or runtime measurement'}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg']:
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches='tight', facecolor='white')
    svg = path.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['results', 'frozen-results', 'training', 'cache', 'local', 'frozen-local', 'data-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage04'))
    parser.add_argument('--sequence', default='03')
    args = parser.parse_args()
    info = json.loads((args.results / 'metrics.json').read_text())
    frozen_info = json.loads((args.frozen_results / 'metrics.json').read_text())
    if info['status'] != 'complete_device_evaluation' or frozen_info['status'] != 'complete_device_evaluation':
        raise ValueError('Both evaluations must be complete')
    if any(info[k] != frozen_info[k] for k in ['backbone', 'mode', 'device', 'seed', 'test_videos']):
        raise ValueError('Paired frozen and LoRA conditions differ')
    manifest = json.loads(args.manifest.read_text())['sequences']
    normal, sources = calibration_check(args.results, info, manifest, args.data_root)
    frozen_normal, frozen_sources = calibration_check(args.frozen_results, frozen_info, manifest, args.data_root)
    proof = provenance(args, info, normal, manifest)
    runs = {kind: {v: load_run(path, v, meta) for v in VARIANTS}
        for kind, path, meta in [('lora', args.results, info), ('frozen', args.frozen_results, frozen_info)]}
    reference = runs['lora']['P3']
    for group in runs.values():
        for run in group.values():
            check_pair(reference, run)
    draws = {(kind, v): bootstrap_draws([run], 1000) for kind, group in runs.items() for v, run in group.items()}
    valid = np.logical_and.reduce([np.isfinite(value).all(axis=1) for value in draws.values()])
    if valid.sum() < 950:
        raise ValueError('Too many degenerate paired draws')
    summary = {'status': 'complete_single_seed_lora_comparison', **{k: info[k] for k in ['backbone', 'mode', 'device', 'seed', 'test_videos']},
        'seeds': 1, 'bootstrap': '1000 paired whole-video resamples, seed2026; one fixed frozen/LoRA trained seed; percentile95%',
        'bootstrap_rejected': int((~valid).sum()), 'results': {}, 'lora_minus_frozen': {},
        'sources': {'lora': sources, 'frozen': frozen_sources}, 'provenance': proof,
        'manifest_sha256': digest(args.manifest), 'plot_code_sha256': digest(__file__),
        'bootstrap_code_sha256': digest(Path(__file__).with_name('summarize_experiments.py')),
        'bootstrap_metric_sha256': digest(Path('src/ipad_jepa/bootstrap_metrics.py')),
        'verification': {'normal_q99_thresholds': 8, 'normal_P2_P3_median_MAD_component_q99': 'passed',
            'calibration_full_target_inventory': 'passed', 'test_full_target_GT_score_time_alarm_replay': 'passed',
            'all_eight_variant_pairings': 'passed'},
        'limitation': 'Single seed and device; no 3-seed mean, training-seed uncertainty, Macro4, real anomaly-type accuracy or runtime claim'}
    for kind in runs:
        summary['results'][kind] = {}
        meta = info if kind == 'lora' else frozen_info
        for v in VARIANTS:
            low, high = np.quantile(draws[kind, v][valid], [.025, .975], axis=0)
            summary['results'][kind][v] = {**meta['variants'][v], 'auroc_ci_low': float(low[0]),
                'auroc_ci_high': float(high[0]), 'ap_ci_low': float(low[1]), 'ap_ci_high': float(high[1])}
    for v in VARIANTS:
        low, high = np.quantile((draws['lora', v] - draws['frozen', v])[valid], [.025, .975], axis=0)
        summary['lora_minus_frozen'][v] = {'auroc_delta': info['variants'][v]['frame_auroc'] - frozen_info['variants'][v]['frame_auroc'],
            'auroc_ci_low': float(low[0]), 'auroc_ci_high': float(high[0]),
            'ap_delta': info['variants'][v]['frame_ap'] - frozen_info['variants'][v]['frame_ap'],
            'ap_ci_low': float(low[1]), 'ap_ci_high': float(high[1])}
    for name in ['lora_training.json', 'lora_training.csv']:
        summary['sources']['lora'][name] = digest(args.results / name)
    (args.results / 'single_seed_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.results / 'provenance_check.json').write_text(json.dumps(proof, indent=2) + '\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for index, kind in enumerate(['frozen', 'lora']):
        result = summary['results'][kind]['P3']
        x = np.arange(2) + (index - .5) * .3
        points = 100 * np.array([result['frame_auroc'], result['frame_ap']])
        low = 100 * np.array([result['auroc_ci_low'], result['ap_ci_low']])
        high = 100 * np.array([result['auroc_ci_high'], result['ap_ci_high']])
        axes[0, 0].bar(x, points, width=.28, color=COLORS[kind], label=LABELS[kind])
        axes[0, 0].vlines(x, low, high, color='#222222')
        axes[0, 0].hlines(low, x-.035, x+.035, color='#222222')
        axes[0, 0].hlines(high, x-.035, x+.035, color='#222222')
        for position, point in zip(x, points):
            axes[0, 0].text(position, point * .5, f'{point:.1f}', ha='center', color='white' if index == 0 else '#111111')
        run = runs[kind]['P3']
        labels = np.concatenate([v[1] for v in run.values()])
        scores = np.concatenate([v[2] for v in run.values()])
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        style = '-' if index == 0 else '--'
        axes[1, 0].plot(fpr, tpr, color=COLORS[kind], linestyle=style, label=LABELS[kind])
        axes[1, 1].plot(recall, precision, color=COLORS[kind], linestyle=style, label=LABELS[kind])
    axes[0, 0].set(xticks=[0, 1], xticklabels=['AUROC', 'AP'], ylabel='P3 score (%)', ylim=(0, 100))
    delta = summary['lora_minus_frozen']['P3']
    points = 100 * np.array([delta['auroc_delta'], delta['ap_delta']])
    low = 100 * np.array([delta['auroc_ci_low'], delta['ap_ci_low']])
    high = 100 * np.array([delta['auroc_ci_high'], delta['ap_ci_high']])
    axes[0, 1].hlines([0, 1], low, high, color=COLORS['lora'], linewidth=2)
    axes[0, 1].scatter(points, [0, 1], color=COLORS['lora'], marker='D')
    axes[0, 1].axvline(0, color='#555555', linestyle=':')
    axes[0, 1].set(yticks=[0, 1], yticklabels=['AUROC', 'AP'], xlabel='LoRA minus frozen (percentage points)', ylim=(-.6, 1.6))
    axes[1, 0].plot([0, 1], [0, 1], ':', color='#555555')
    axes[1, 0].set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].axhline(labels.mean(), linestyle=':', color='#555555')
    axes[1, 1].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    for axis in [axes[0, 0], axes[1, 0], axes[1, 1]]:
        axis.legend(frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.spines[['top', 'right']].set_visible(False)
    counts = info['variants']['P3']
    stem = f"{info['backbone']}_{info['device']}_{info['mode']}_seed{info['seed']}_lora"
    fig.suptitle(f"{info['backbone']} / {info['device']} / {info['mode']} / seed {info['seed']} only\nP3: {info['test_videos']} videos, {counts['frames']} frames ({counts['anomaly_frames']} anomalous)", fontsize=12)
    fig.text(.02, -.04, '95% CI: paired whole-video bootstrap for a single fixed seed. Both normal-only heads/calibrations are fitted separately.\nNo test score min-max or checkpoint selection; no training-seed uncertainty, Macro4 or measured runtime.', fontsize=9)
    save(fig, args.out / (stem + '_evaluation'))
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.7), sharex=True, layout='constrained')
    for axis, kind, folder, cal in zip(axes, ['frozen', 'lora'], [args.frozen_results, args.results], [frozen_normal, normal]):
        trace = read_csv(folder / 'P3' / f'{args.sequence}.csv')
        x = np.array([int(r['frame']) for r in trace]); y = np.array([float(r['score']) for r in trace])
        gt = np.array([int(r['label']) for r in trace]); alarm = np.array([int(r['alarm']) for r in trace], dtype=bool)
        threshold = cal['calibration']['P3']['threshold']
        axis.plot(x, y, color=COLORS[kind], linewidth=1, label=LABELS[kind] + ' P3')
        axis.axhline(threshold, linestyle='--', color='#333333', linewidth=1, label='Own fixed normal q99')
        axis.fill_between(x, 0, 1, where=gt == 1, transform=axis.get_xaxis_transform(), step='mid', color='#C98DA1', alpha=.22, label='GT anomaly')
        finite = y[np.isfinite(y)]
        band = min(finite.min(), threshold) - .05 * max(finite.max() - min(finite.min(), threshold), 1.)
        axis.scatter(x[alarm], np.full(alarm.sum(), band), marker='^', s=9, color='#222222', label='Cached alarm (lower band)')
        axis.set(ylabel='Own normal median/MAD P3 score')
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False, fontsize=8, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.01))
    axes[-1].set_xlabel('Target frame')
    fig.suptitle(f"{info['backbone']} / {info['device']} / {info['mode']} / seed {info['seed']} only / fixed test video {args.sequence}", fontsize=12)
    fig.text(.02, -.035, 'Own normal calibration per condition. Cached alarm mask: target19..N-8, GT uncertainty never changes streak.\nFrame indices are not measured wall-clock alarm delays; online runtime EOF handling is evaluated separately.', fontsize=8)
    save(fig, args.out / (stem + f'_sequence{args.sequence}'))
    print(json.dumps({'status': summary['status'], 'provenance': {k: v for k, v in proof.items() if k != 'cache_checks'},
        'results': summary['results'], 'lora_minus_frozen': summary['lora_minus_frozen']}, indent=2))


if __name__ == '__main__':
    main()
