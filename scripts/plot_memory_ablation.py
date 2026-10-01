"""Render audited budget and phase-bin comparisons separately, with paired CIs."""
import argparse
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_neighbour_ablation import NAMES, COLORS, MODES, choose, save, differences, digest

BASE = 'B16_M2048'
LABELS = ['B16_M1024', 'B16_M4096', 'B8_M2048']
SHORT = {'B16_M1024': '1024 prototypes', 'B16_M4096': '4096 prototypes', 'B8_M2048': '8 bins'}


def accuracy(axes, rows, series, settings, ticks, xlabel, title):
    for axis, metric in zip(axes, ['auroc', 'ap']):
        for index, (legend, keys) in enumerate(series):
            selected = [choose(rows, configuration=label, **keys) for label in settings]
            if any(r['seeds'] != 3 for r in selected):
                raise ValueError('Completed three-seed means required')
            x = np.arange(len(settings)) + (index - (len(series) - 1) / 2) * .08
            mean = np.array([r[metric + '_mean'] * 100 for r in selected])
            low = np.array([r[metric + '_ci_low'] * 100 for r in selected])
            high = np.array([r[metric + '_ci_high'] * 100 for r in selected])
            axis.plot(x, mean, color=COLORS[index], marker=['s', 'o'][index],
                      linestyle=['--', '-'][index], label=legend)
            axis.vlines(x, low, high, color=COLORS[index])
            axis.hlines(low, x - .04, x + .04, color=COLORS[index])
            axis.hlines(high, x - .04, x + .04, color=COLORS[index])
        axis.set(xticks=np.arange(len(settings)), xticklabels=ticks, xlabel=xlabel,
                 ylabel=metric.upper() + ' (%)', ylim=(0, 100), xlim=(-.35, len(settings) - .65), title=title)
        axis.spines[['top', 'right']].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/stage05/ablations/memory'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage05'))
    args = parser.parse_args()
    proof = json.loads((args.root / 'validation.json').read_text())
    if (proof['status'] != 'passed' or proof['thresholds_checked'] != proof['completed_groups'] * 12 or
        proof['source_conditions_replayed'] != proof['completed_groups'] * 3 or
        proof['default_alarm_mismatches_original_mask'] != 0 or
        proof['producer_code_sha256'] != digest('src/ipad_jepa/memory_ablation.py') or
        proof['verifier_sha256'] != digest(Path(__file__).with_name('summarize_memory_ablation.py'))):
        raise ValueError('Current audited completed memory results required')
    for name in ['device_summary', 'macro_summary']:
        if proof[name + '_sha256'] != digest(args.root / (name + '.json')):
            raise ValueError('Audited summary changed')
    complete = proof['matrix_complete']
    summary = json.loads((args.root / ('macro_summary.json' if complete else 'device_summary.json')).read_text())
    rows = summary['results']
    if complete and (proof['completed_groups'] != 16 or len(rows) != 16 or any(r['devices'] != 4 for r in rows)):
        raise ValueError('Complete equal-weight Macro4 required')
    groups = [(model, None) for model in NAMES] if complete else [
        (model, mode) for model in NAMES for mode in MODES
        if any(r['backbone'] == model and r['mode'] == mode for r in rows)]
    hashes = {}
    source_root = '/'.join(args.root.parts[args.root.parts.index('memory'):]) if 'memory' in args.root.parts else str(args.root)
    source_note = f'Source: {source_root}/' + ('macro_summary.json; equal weight per device.' if complete else 'device_summary.json; partial matrix, not Macro4.')
    note = ('Common targets t=19..N-8; three fixed seed metrics averaged; 95% whole-video bootstrap CI.\n'
            'Frozen encoder/head, clip16, k5, history5. Normal-only bank/PCA, per-bank temperature and q99; cached search, no runtime.\n')
    for model, mode in groups:
        devices = [None] if complete else sorted({r['device'] for r in rows if r['backbone'] == model and r['mode'] == mode})
        series_by_device = [(device, [(m, {'backbone': model, 'mode': m}) for m in MODES]) for device in devices] if complete else [
            (device, [(mode, {'backbone': model, 'mode': mode, 'device': device})]) for device in devices]
        prefix = 'memory_' + model + ('_macro4' if complete else '_' + mode)
        for name, settings, ticks, xlabel, policy in [
            ('budget', ['B16_M1024', BASE, 'B16_M4096'], [1024, 2048, 4096], 'Total prototypes / 16 bins', 'Budget: shared normal candidates and PCA.'),
            ('bins', ['B8_M2048', BASE], [8, 16], 'Phase bins / 2048 total prototypes', 'Bins: normal phase strata and PCA refitted; three-bin search span also changes.')]:
            fig, axes = plt.subplots(len(devices), 2, figsize=(11, max(3.8, 3.3 * len(devices))),
                                     squeeze=False, layout='constrained')
            for index, (device, series) in enumerate(series_by_device):
                title = NAMES[model] + (' / equal-weight Macro4' if complete else f' / {mode} / {device}')
                accuracy(axes[index], rows, series, settings, ticks, xlabel, title)
            axes[0, 0].legend(frameon=False, loc='upper right')
            fig.suptitle('P3 memory sensitivity / ' + ('complete matrix' if complete else 'partial matrix'), fontsize=13)
            hashes.update(save(fig, args.out / (prefix + '_' + name), note + policy + '\n' + source_note))
        items = []
        for device in devices:
            for i, m in enumerate(MODES if complete else [mode]):
                for label in LABELS:
                    keys = {'backbone': model, 'mode': m, 'comparison': f'{label}_minus_{BASE}'}
                    if device is not None:
                        keys['device'] = device
                    legend = (m if complete else device) + ' / ' + SHORT[label] + ' - default'
                    items.append((legend, keys, COLORS[i], ['s', 'o'][i]))
        hashes.update(differences(summary['paired_deltas'], items,
            NAMES[model] + ' / memory setting minus default / paired 95% CI',
            args.out / (prefix + '_paired_differences'), note + 'Default: 16 bins / 2048 prototypes.\n' + source_note))
    receipt = {'status': 'rendered_from_verified_summaries', 'matrix_complete': complete,
        'completed_groups': proof['completed_groups'], 'validation_sha256': digest(args.root / 'validation.json'),
        'device_summary_sha256': proof['device_summary_sha256'], 'macro_summary_sha256': proof['macro_summary_sha256'],
        'plot_code_sha256': digest(__file__),
        'shared_plot_code_sha256': digest(Path(__file__).with_name('plot_neighbour_ablation.py')),
        'reporting_runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'figures': hashes, 'scope': 'Separate budget/bin plots and paired uncertainty; final visual inspection recorded separately'}
    (args.root / 'figure_sources.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
