"""Plot verified neighbour OFAT accuracy and paired differences, with explicit scope."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'dinov3-l': 'DINOv3-L', 'vjepa21-l': 'V-JEPA 2.1-L'}
MODES = ['offline', 'online']
KS = [1, 5, 10]
COLORS = ['#245B86', '#C17A1A']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(rows, **keys):
    matches = [r for r in rows if all(r[k] == value for k, value in keys.items())]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate plotted neighbour condition')
    return matches[0]


def save(fig, path, note):
    fig.text(.015, -.08, note, fontsize=9, va='top')
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg']:
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches='tight', facecolor='white')
    svg = path.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    return {p.name: digest(p) for p in [path.with_suffix('.png'), svg]}


def accuracy(axes, rows, title, series):
    for axis, metric in zip(axes, ['auroc', 'ap']):
        for index, (label, conditions) in enumerate(series):
            values = [choose(rows, neighbours=k, **conditions) for k in KS]
            if any(r['seeds'] != 3 for r in values):
                raise ValueError('Expected three completed training seeds')
            positions = np.array(KS) + (index - (len(series) - 1) / 2) * .18
            means = np.array([r[metric + '_mean'] * 100 for r in values])
            low = np.array([r[metric + '_ci_low'] * 100 for r in values])
            high = np.array([r[metric + '_ci_high'] * 100 for r in values])
            color = COLORS[index]
            axis.plot(positions, means, color=color, marker=['s', 'o'][index],
                      linestyle=['--', '-'][index], label=label)
            axis.vlines(positions, low, high, color=color, linewidth=1.2)
            axis.hlines(low, positions - .12, positions + .12, color=color)
            axis.hlines(high, positions - .12, positions + .12, color=color)
        axis.set(xlabel='Prototype neighbours k', xticks=KS, xlim=(.3, 10.7),
                 ylabel=metric.upper() + ' (%)', ylim=(0, 100), title=title)
        axis.spines[['top', 'right']].set_visible(False)


def differences(rows, items, title, path, note):
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.8, len(items) * .53 + 1.7)),
                             sharey=True, layout='constrained')
    for axis, metric in zip(axes, ['auroc', 'ap']):
        axis.axvline(0, color='#777777', linestyle=':', linewidth=1)
        for index, (label, conditions, color, marker) in enumerate(items):
            row = choose(rows, **conditions)
            value, low, high = [row[metric + suffix] * 100
                               for suffix in ['_delta', '_delta_ci_low', '_delta_ci_high']]
            axis.scatter(value, index, color=color, marker=marker, zorder=3)
            axis.hlines(index, low, high, color=color, linewidth=1.5)
            axis.vlines([low, high], index - .08, index + .08, color=color)
        axis.set(xlabel=metric.upper() + ' difference (pp)', yticks=np.arange(len(items)),
                 yticklabels=[i[0] for i in items], ylim=(len(items) - .5, -.5))
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, fontsize=13)
    return save(fig, path, note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/stage05/ablations/neighbours'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage05'))
    args = parser.parse_args()
    proof = json.loads((args.root / 'validation.json').read_text())
    for name in ['device_summary', 'macro_summary']:
        if proof[name + '_sha256'] != digest(args.root / (name + '.json')):
            raise ValueError('Verified summaries changed')
    if (proof['status'] != 'passed' or proof['thresholds_checked'] != proof['completed_groups'] * 9 or
        proof['source_conditions_replayed'] != proof['completed_groups'] * 3 or
        proof['K5_alarm_mismatches_original_mask'] != 0 or
        proof['producer_code_sha256'] != digest('src/ipad_jepa/neighbour_ablation.py') or
        proof['verifier_sha256'] != digest(Path(__file__).with_name('summarize_neighbour_ablation.py'))):
        raise ValueError('Completed source-verified neighbour groups required')
    hashes = {}
    source_root = '/'.join(args.root.parts[args.root.parts.index('neighbours'):]) if 'neighbours' in args.root.parts else str(args.root)
    note = ('Common normal calibration and evaluated targets: t=19..N-8; three-seed mean, 95% whole-video bootstrap CI.\n'
            'Encoder, head, PCA, bank, temperature and history5 fixed; per-k normal-only recalibration. Cached search; no runtime measurement.')
    if proof['matrix_complete']:
        if proof['completed_groups'] != 16 or proof['macro_groups'] != 12:
            raise ValueError('Complete equal-weight Macro4 required')
        summary = json.loads((args.root / 'macro_summary.json').read_text())
        if any(r['devices'] != 4 for r in summary['results']):
            raise ValueError('Incomplete Macro4')
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout='constrained')
        for index, model in enumerate(NAMES):
            accuracy(axes[index], summary['results'], NAMES[model],
                     [(mode, {'backbone': model, 'mode': mode}) for mode in MODES])
        axes[0, 0].legend(frameon=False, loc='upper right')
        fig.suptitle('Frozen encoders / P3 neighbour sensitivity / equal-weight four-device accuracy', fontsize=13)
        macro_note = note + f'\nSource: {source_root}/macro_summary.json; equal weight per device.'
        hashes.update(save(fig, args.out / 'neighbours_macro4', macro_note))
        items = [(f'{NAMES[m]} / {mode} / K{k}-K5',
                  {'backbone': m, 'mode': mode, 'comparison': f'K{k}_minus_K5'},
                  COLORS[i], 's' if mode == 'offline' else 'o')
                 for i, m in enumerate(NAMES) for mode in MODES for k in [1, 10]]
        hashes.update(differences(summary['paired_deltas'], items,
            'Neighbour count minus K5 / paired whole-video 95% CI / equal-weight Macro4',
            args.out / 'neighbours_paired_differences', macro_note))
    else:
        summary = json.loads((args.root / 'device_summary.json').read_text())
        for model in NAMES:
            for mode in MODES:
                devices = sorted({r['device'] for r in summary['results']
                                  if r['backbone'] == model and r['mode'] == mode})
                if not devices:
                    continue
                fig, axes = plt.subplots(len(devices), 2, figsize=(11, max(3.8, 3.3 * len(devices))),
                                         squeeze=False, layout='constrained')
                for index, device in enumerate(devices):
                    accuracy(axes[index], summary['results'], f'{NAMES[model]} / {mode} / {device}',
                             [(mode, {'backbone': model, 'mode': mode, 'device': device})])
                fig.suptitle('P3 neighbour sensitivity / completed devices / partial matrix', fontsize=13)
                device_note = note + f'\nSource: {source_root}/device_summary.json; ' + ', '.join(devices) + ' only; not Macro4.'
                prefix = f'neighbours_{model}_{mode}'
                hashes.update(save(fig, args.out / prefix, device_note))
                items = [(f'{d} / K{k}-K5', {'backbone': model, 'mode': mode,
                          'device': d, 'comparison': f'K{k}_minus_K5'}, COLORS[i], ['s', 'o'][i])
                         for d in devices for i, k in enumerate([1, 10])]
                hashes.update(differences(summary['paired_deltas'], items,
                    f'{NAMES[model]} / {mode} / paired 95% CI / partial matrix',
                    args.out / (prefix + '_paired_differences'), device_note))
    receipt = {'status': 'rendered_from_verified_summaries', 'matrix_complete': proof['matrix_complete'],
               'reporting_runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
               'completed_groups': proof['completed_groups'], 'validation_sha256': digest(args.root / 'validation.json'),
               'device_summary_sha256': proof['device_summary_sha256'],
               'macro_summary_sha256': proof['macro_summary_sha256'], 'plot_code_sha256': digest(__file__),
               'figures': hashes, 'scope': 'Accuracy and uncertainty plots; visual inspection recorded separately'}
    (args.root / 'figure_sources.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
