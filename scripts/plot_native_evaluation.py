"""Inspect and plot one completed native baseline seed; never imply a 3-seed result."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from summarize_experiments import load_run, check_pair, bootstrap_draws


COLORS = {'B0': '#245B86', 'B1': '#B97812'}
LABELS = {'B0': 'B0 pixel -PSNR', 'B1': 'B1 latent memory residual'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['.png', '.svg']:
        fig.savefig(path.with_suffix(extension), dpi=180, bbox_inches='tight', facecolor='white')
    svg = path.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage01'))
    parser.add_argument('--sequence', default='03')
    args = parser.parse_args()
    info = json.loads((args.results/'metrics.json').read_text())
    if info['status'] != 'complete_device_evaluation' or info['backbone'] != 'IPAD-native-repaired':
        raise ValueError('Completed native device evaluation required')
    normal = json.loads((args.results/'normal_fit.json').read_text())
    if (normal['status'] != 'normal_calibration_complete' or normal['training_scope'] != info['scope'] or
        normal['calibration_overlap_with_training'] != (info['scope'] == 'reference')):
        raise ValueError('Native normal calibration scope differs')
    manifest = json.loads(args.manifest.read_text())['sequences']
    calibration = {row['sequence']: row for row in manifest
                   if row['device'] == info['device'] and row.get('split') == 'calibration'}
    with (args.results/'normal_calibration.csv').open() as stream:
        cal_rows = list(csv.DictReader(stream))
    if set(row['sequence'] for row in cal_rows) != set(calibration):
        raise ValueError('Native calibration video split differs')
    for row in cal_rows:
        target = int(row['frame']); n = calibration[row['sequence']]['frames']
        if int(row['valid']) != int(19 <= target <= n-8):
            raise ValueError('Native calibration target mask differs')
    runs = {variant: load_run(args.results, variant, info) for variant in COLORS}
    check_pair(runs['B0'], runs['B1'])
    draws = {variant: bootstrap_draws([run], 1000) for variant, run in runs.items()}
    valid = np.logical_and.reduce([np.isfinite(value).all(axis=1) for value in draws.values()])
    if valid.sum() < 950:
        raise ValueError('Too many degenerate native video bootstrap draws')
    summary = {'status': 'complete_single_seed_summary', 'device': info['device'], 'scope': info['scope'],
               'seed': info['seed'], 'seeds': 1, 'test_videos': info['test_videos'],
               'bootstrap': '1000 whole-video resamples, seed2026, paired B0/B1; percentile95%; one fixed trained seed',
               'bootstrap_rejected': int((~valid).sum()), 'variants': {}, 'sources': {}}
    for variant, run in runs.items():
        parameters = normal['variants'][variant]
        values = np.array([float(row[variant+'_raw']) for row in cal_rows if row['valid'] == '1'])
        median = float(np.median(values))
        scale = max(float(1.4826*np.median(np.abs(values-median))), 1e-6)
        threshold = float(np.quantile((values-median)/scale, .99))
        np.testing.assert_allclose([median, scale, threshold],
                                  [parameters['median'], parameters['mad_scale'], parameters['threshold']],
                                  rtol=0, atol=1e-9)
        low, high = np.quantile(draws[variant][valid], [.025, .975], axis=0)
        summary['variants'][variant] = {**info['variants'][variant],
            'auroc_ci_low': float(low[0]), 'auroc_ci_high': float(high[0]),
            'ap_ci_low': float(low[1]), 'ap_ci_high': float(high[1])}
        for source in (args.results/variant).glob('*.csv'):
            with source.open() as stream:
                trace = list(csv.DictReader(stream))
            raw = np.array([float(row['raw']) for row in trace])
            actual_scores = np.array([float(row['score']) for row in trace])
            np.testing.assert_allclose(actual_scores, (raw-median)/scale, rtol=0, atol=1e-9)
            streak = 0
            for row in trace:
                streak = streak+1 if row['inference_valid'] == '1' and float(row['score']) > threshold else 0
                if int(row['alarm']) != int(streak >= 3):
                    raise ValueError('Native alarm flags differ from fixed normal threshold/streak')
            summary['sources'][str(source.relative_to(args.results))] = digest(source)
    delta = draws['B1'][valid]-draws['B0'][valid]
    low, high = np.quantile(delta, [.025, .975], axis=0)
    summary['B1_minus_B0'] = {'auroc_delta': info['variants']['B1']['frame_auroc']-info['variants']['B0']['frame_auroc'],
        'auroc_ci_low': float(low[0]), 'auroc_ci_high': float(high[0]),
        'ap_delta': info['variants']['B1']['frame_ap']-info['variants']['B0']['frame_ap'],
        'ap_ci_low': float(low[1]), 'ap_ci_high': float(high[1])}
    for name in ['metrics.json', 'normal_fit.json', 'normal_calibration.csv']:
        summary['sources'][name] = digest(args.results/name)
    summary['plot_code_sha256'] = digest(Path(__file__))
    summary['bootstrap_code_sha256'] = digest(Path(__file__).with_name('summarize_experiments.py'))
    summary['limitation'] = 'Single seed; no training-seed uncertainty, macro4, runtime or paper-exact reproduction claim'
    (args.results/'single_seed_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    for index, (variant, run) in enumerate(runs.items()):
        result = summary['variants'][variant]; x = np.arange(2)+(index-.5)*.3
        points = 100*np.array([result['frame_auroc'], result['frame_ap']])
        low = 100*np.array([result['auroc_ci_low'], result['ap_ci_low']])
        high = 100*np.array([result['auroc_ci_high'], result['ap_ci_high']])
        axes[0].bar(x, points, width=.28, color=COLORS[variant], label=LABELS[variant],
                    hatch='//' if index else None, edgecolor='#333333', linewidth=.6)
        axes[0].vlines(x, low, high, color='#222222', linewidth=1)
        axes[0].hlines(low, x-.035, x+.035, color='#222222', linewidth=1)
        axes[0].hlines(high, x-.035, x+.035, color='#222222', linewidth=1)
        for position, point in zip(x, points):
            axes[0].text(position, point*.5, f'{point:.1f}', ha='center', va='center', fontsize=9,
                         color='white' if index == 0 else '#111111', fontweight='bold',
                         bbox={'facecolor': COLORS[variant], 'edgecolor': 'none', 'pad': 1})
        labels = np.concatenate([value[1] for value in run.values()])
        scores = np.concatenate([value[2] for value in run.values()])
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[1].plot(fpr, tpr, color=COLORS[variant], linestyle='-' if index == 0 else '--', label=LABELS[variant])
        axes[2].plot(recall, precision, color=COLORS[variant], linestyle='-' if index == 0 else '--', label=LABELS[variant])
    axes[0].set(xticks=[0, 1], xticklabels=['AUROC', 'AP'], ylabel='Score (%)', ylim=(0, 100))
    axes[0].legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02))
    axes[1].plot([0, 1], [0, 1], ':', color='#555555'); axes[1].set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1))
    axes[2].axhline(float(labels.mean()), linestyle=':', color='#555555'); axes[2].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    frames = info['variants']['B0']['frames']; anomalies = info['variants']['B0']['anomaly_frames']
    fig.suptitle(f"IPAD native repaired / {info['device']} / {info['scope']} / seed {info['seed']} only / {info['test_videos']} videos, {frames} frames ({anomalies} anomalous)", fontsize=12)
    fig.text(.02, -.035, 'Bars: one trained seed; 95% paired whole-video bootstrap CI. Curves: all valid frames. Normal-only calibration; no test min-max.', fontsize=8)
    fig.text(.02, -.075, 'Both B0/B1 use decoder training; independent B1 inference omits the decoder. Runtime not measured. Source: single_seed_summary.json and score CSVs.', fontsize=8)
    stem = f"IPAD_{info['device']}_{info['scope']}_seed{info['seed']}"
    save(fig, args.out/(stem+'_evaluation'))
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.7), sharex=True, layout='constrained')
    for axis, variant in zip(axes, COLORS):
        with (args.results/variant/f'{args.sequence}.csv').open() as stream:
            trace = list(csv.DictReader(stream))
        x = np.array([int(row['frame']) for row in trace]); y = np.array([float(row['score']) for row in trace])
        gt = np.array([int(row['label']) for row in trace]); alarms = np.array([row['alarm'] == '1' for row in trace])
        threshold = normal['variants'][variant]['threshold']
        axis.plot(x, y, color=COLORS[variant], linewidth=1, label=LABELS[variant])
        axis.axhline(threshold, linestyle='--', color='#333333', linewidth=1, label='Fixed normal q99')
        axis.fill_between(x, 0, 1, where=gt == 1, transform=axis.get_xaxis_transform(), step='mid', color='#C98DA1', alpha=.22, label='GT anomaly')
        band = min(y.min(), threshold)-.05*max(y.max()-min(y.min(), threshold), 1.)
        axis.scatter(x[alarms], np.full(alarms.sum(), band), marker='^', s=9, color='#222222', label='Alarm flag (lower band)')
        axis.set(ylabel='Normal median/MAD z score'); axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel('Target frame')
    fig.suptitle(f"IPAD {info['device']} / {info['scope']} / seed {info['seed']} only / fixed test video {args.sequence}", fontsize=12)
    fig.text(.02, -.035, 'Alarm: 3 consecutive threshold exceedances after target19. GT is used after scoring. Frame indices are not measured wall-clock alarm delays.', fontsize=8)
    save(fig, args.out/(stem+f'_sequence{args.sequence}'))
    print(json.dumps({'status': summary['status'], 'variants': summary['variants'], 'B1_minus_B0': summary['B1_minus_B0']}, indent=2))


if __name__ == '__main__':
    main()
