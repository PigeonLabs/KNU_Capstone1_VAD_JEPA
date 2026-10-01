"""Audit and plot one completed controlled native device, with three fixed seeds."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score

from ipad_jepa.alignment import align
from summarize_experiments import load_run, check_pair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/stage01/controlled'))
    parser.add_argument('--device', default='R01', choices=['R01', 'R02', 'R03', 'R04'])
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage01'))
    args = parser.parse_args()
    manifest = [r for r in json.loads(args.manifest.read_text())['sequences']
                if r['device'] == args.device]
    test = {r['sequence']: r for r in manifest if r['partition'] == 'testing'}
    cal = {r['sequence']: r for r in manifest if r.get('split') == 'calibration'}
    fit = [r for r in manifest if r.get('split') == 'fit']
    sources = {str(args.manifest): digest(args.manifest)}
    annotations = {}
    for name, row in test.items():
        path = args.data_root / row['label_file']
        if digest(path) != row['label_sha256']:
            raise ValueError('Raw annotation differs from audited manifest')
        annotations[name] = align(np.load(path, allow_pickle=False), row['frames'])[:2]
    runs = {variant: [] for variant in ['B0', 'B1']}
    threshold_checks = []
    for seed in [0, 1, 2]:
        training = args.root / 'training' / args.device / f'seed{seed}'
        trained = json.loads((training / 'training.json').read_text())
        if (trained['status'] != 'complete_training' or trained['epochs_completed'] != 50 or
            trained['scope'] != 'controlled' or trained['device'] != args.device or
            trained['seed'] != seed or trained['max_steps'] is not None or
            trained['code_sha256'] != digest('src/ipad_jepa/ipad_baseline.py') or
            trained['normal_videos'] != [r['sequence'] for r in fit] or
            trained['normal_source_hashes'] != [r['frames_content_sha256'] for r in fit]):
            raise ValueError('Complete controlled training source differs')
        curve = read_csv(training / 'training.csv')
        if ([int(r['epoch']) for r in curve] != list(range(1, 51)) or
            int(curve[-1]['steps_total']) != trained['steps']):
            raise ValueError('Native training curve differs from completion marker')
        folder = args.root / 'IPAD-native-repaired' / 'offline' / args.device / f'seed{seed}'
        meta = json.loads((folder / 'metrics.json').read_text())
        normal = json.loads((folder / 'normal_fit.json').read_text())
        if (meta['status'] != 'complete_device_evaluation' or meta['seed'] != seed or
            meta['device'] != args.device or meta['scope'] != 'controlled' or
            meta['code_sha256'] != digest('src/ipad_jepa/ipad_evaluate.py') or
            normal['status'] != 'normal_calibration_complete' or
            normal['training_scope'] != 'controlled' or normal['calibration_overlap_with_training']):
            raise ValueError('Native evaluation/calibration source differs')
        calibration = read_csv(folder / 'normal_calibration.csv')
        if {r['sequence'] for r in calibration} != set(cal):
            raise ValueError('Normal calibration inventory differs')
        for name, row in cal.items():
            selected = [r for r in calibration if r['sequence'] == name]
            expected = np.arange(8, row['frames'] - 7)
            np.testing.assert_array_equal([int(r['frame']) for r in selected], expected)
            np.testing.assert_array_equal([int(r['valid']) for r in selected], expected >= 19)
        for variant in runs:
            values = np.array([float(r[variant + '_raw']) for r in calibration if r['valid'] == '1'])
            median = float(np.median(values))
            scale = max(float(1.4826 * np.median(np.abs(values - median))), 1e-6)
            threshold = float(np.quantile((values - median) / scale, .99))
            saved = normal['variants'][variant]
            np.testing.assert_allclose([median, scale, threshold],
                [saved['median'], saved['mad_scale'], saved['threshold']], rtol=0, atol=1e-9)
            threshold_checks.append({'seed': seed, 'variant': variant, 'normal_frames': len(values),
                                     'threshold': threshold, 'checkpoint_sha256': normal['checkpoint_sha256']})
            paths = sorted((folder / variant).glob('*.csv'), key=lambda p: int(p.stem))
            if {p.stem for p in paths} != set(test):
                raise ValueError('Test video inventory differs')
            for path in paths:
                trace = read_csv(path)
                target = np.array([int(r['frame']) for r in trace])
                n = test[path.stem]['frames']
                np.testing.assert_array_equal(target, np.arange(8, n - 7))
                labels, known = annotations[path.stem]
                np.testing.assert_array_equal([int(r['label']) for r in trace], labels[target])
                np.testing.assert_array_equal([int(r['valid']) for r in trace], (target >= 19) & known[target])
                np.testing.assert_array_equal([int(r['inference_valid']) for r in trace], target >= 19)
                raw = np.array([float(r['raw']) for r in trace])
                scores = np.array([float(r['score']) for r in trace])
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite native raw scores')
                np.testing.assert_allclose(scores, (raw - median) / scale, rtol=0, atol=1e-9)
                streak = 0
                for row in trace:
                    streak = streak + 1 if row['inference_valid'] == '1' and float(row['score']) > threshold else 0
                    if int(row['alarm']) != int(streak >= 3):
                        raise ValueError('Native alarm differs from normal threshold/streak')
                sources[str(path)] = digest(path)
            run = load_run(folder, variant, meta)
            if runs['B0']:
                check_pair(runs['B0'][0], run)
            runs[variant].append(run)
        for path in [training / 'training.json', training / 'training.csv',
                     folder / 'metrics.json', folder / 'normal_fit.json', folder / 'normal_calibration.csv']:
            sources[str(path)] = digest(path)
    # Independent sklearn reference: bootstrap the videos directly, bypassing VideoBootstrapMetric.
    keys = list(runs['B0'][0])
    points, draws = {}, {}
    for variant, seeds in runs.items():
        rng = np.random.default_rng(2026)
        def statistic(run, selected):
            labels = np.concatenate([run[k][1] for k in selected])
            scores = np.concatenate([run[k][2] for k in selected])
            if len(np.unique(labels)) < 2:
                return np.array([np.nan, np.nan])
            return np.array([roc_auc_score(labels, scores), average_precision_score(labels, scores)])
        points[variant] = np.mean([statistic(run, keys) for run in seeds], axis=0)
        draws[variant] = np.array([np.mean([statistic(run, selected) for run in seeds], axis=0)
            for selected in [rng.choice(keys, len(keys), replace=True) for _ in range(1000)]])
    valid = np.logical_and.reduce([np.isfinite(v).all(axis=1) for v in draws.values()])
    if valid.sum() < 950:
        raise ValueError('Too many degenerate video draws')
    summary_path = args.root / 'device_summary.json'
    summary = json.loads(summary_path.read_text())
    rows = [r for r in summary['results'] if r['backbone'] == 'IPAD-native-repaired' and
            r['mode'] == 'offline' and r['device'] == args.device]
    if len(rows) != 2 or any(r['seeds'] != 3 for r in rows):
        raise ValueError('Expected exactly two three-seed summary variants')
    errors = []
    for row in rows:
        variant = row['variant']
        low, high = np.quantile(draws[variant][valid], [.025, .975], axis=0)
        for i, metric in enumerate(['auroc', 'ap']):
            for actual, suffix in [(points[variant][i], '_mean'), (low[i], '_ci_low'), (high[i], '_ci_high')]:
                errors.append(abs(float(actual) - row[metric + suffix]))
    delta = points['B1'] - points['B0']
    low, high = np.quantile((draws['B1'] - draws['B0'])[valid], [.025, .975], axis=0)
    paired = next(r for r in summary['paired_deltas'] if r['device'] == args.device and
                  r['comparison'] == 'B1_minus_B0' and r['backbone'] == 'IPAD-native-repaired')
    for i, metric in enumerate(['auroc', 'ap']):
        for actual, suffix in [(delta[i], '_delta'), (low[i], '_delta_ci_low'), (high[i], '_delta_ci_high')]:
            errors.append(abs(float(actual) - paired[metric + suffix]))
    if max(errors) > 1e-12 or any(r['bootstrap_rejected'] != int((~valid).sum()) for r in rows):
        raise ValueError('Independent sklearn bootstrap differs from published summary')
    sources[str(summary_path)] = digest(summary_path)
    for path in ['scripts/summarize_experiments.py', 'src/ipad_jepa/bootstrap_metrics.py',
                 'src/ipad_jepa/ipad_baseline.py', 'src/ipad_jepa/ipad_evaluate.py',
                 Path(__file__).resolve().relative_to(Path.cwd().resolve())]:
        sources[str(Path(path))] = digest(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout='constrained')
    for index, row in enumerate(rows):
        variant = row['variant']; color = ['#245B86', '#C17A1A'][index]
        for axis, metric in zip(axes[:2], ['auroc', 'ap']):
            axis.vlines(index, row[metric + '_ci_low'] * 100, row[metric + '_ci_high'] * 100, color=color)
            axis.scatter(index, row[metric + '_mean'] * 100, color=color, marker=['s', 'o'][index], s=55)
            axis.set(xticks=[0, 1], xticklabels=['B0 pixel', 'B1 latent'], ylabel=metric.upper() + ' (%)',
                     ylim=(0, 100), xlim=(-.6, 1.6))
    for i, metric in enumerate(['auroc', 'ap']):
        axes[2].hlines(i, paired[metric + '_delta_ci_low'] * 100,
                      paired[metric + '_delta_ci_high'] * 100, color='#245B86')
        axes[2].scatter(paired[metric + '_delta'] * 100, i, color='#245B86')
    axes[2].axvline(0, color='#777777', linestyle=':')
    axes[2].set(yticks=[0, 1], yticklabels=['AUROC', 'AP'], ylim=(1.5, -.5),
                xlabel='B1 minus B0 (pp)', title='Paired difference')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'IPAD native repaired / {args.device} controlled / three-seed mean', fontsize=13)
    fig.text(.02, -.08, f'{len(keys)} test videos; common t=19..N-8. 95% paired whole-video bootstrap CI, 1000 draws, fixed seeds 0/1/2.\n'
             'Both variants use decoder training. Not Macro4, training-seed uncertainty, paper-exact reproduction or measured runtime.\n'
             'Source: controlled/device_summary.json; independently checked against sklearn.', fontsize=8, va='top')
    args.out.mkdir(parents=True, exist_ok=True)
    figures = {}
    for suffix in ['.png', '.svg']:
        path = args.out / f'IPAD_{args.device}_controlled_three_seed{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        if suffix == '.svg':
            path.write_text('\n'.join(r.rstrip() for r in path.read_text().splitlines()) + '\n')
        figures[str(path)] = digest(path)
    plt.close(fig)
    record = {'status': 'passed', 'device': args.device, 'scope': 'controlled', 'seeds': [0, 1, 2],
              'normal_thresholds_checked': len(threshold_checks), 'threshold_checks': threshold_checks,
              'actual_annotations_checked': len(test), 'annotation_sha256': {k: r['label_sha256'] for k, r in test.items()},
              'sklearn_bootstrap_float_values_compared': len(errors), 'max_absolute_error': max(errors),
              'absolute_tolerance': 1e-12, 'bootstrap_draws': 1000, 'bootstrap_rejected': int((~valid).sum()),
              'sources': sources, 'figures': figures,
              'reporting_runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                                    'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
              'limitation': 'CSV/source audit and independent metric aggregation; no independent encoder extraction or private checkpoint hash recomputation. Three fixed seeds, one device, no runtime.'}
    (args.root / f'{args.device}_three_seed_validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ['sources', 'figures', 'threshold_checks', 'annotation_sha256']}, indent=2))


if __name__ == '__main__':
    main()
