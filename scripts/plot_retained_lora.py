"""Render mixed current/retained summaries after fresh condition and statistic replay."""
import argparse
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lora_reporting import MODELS, MODES, DEVICES, SEEDS
from plot_lora_matrix import group_rows, verify_plotted_statistics
from plot_neighbour_ablation import digest, save, NAMES
from summarize_retained_lora_matrix import route_condition


def verify_mixed_sources(args, validation):
    if validation['status'] != 'passed_mixed_current_retained_evidence' or validation['kind'] != args.kind:
        raise ValueError('Explicit matching mixed current/retained validation required')
    if validation['manifest_sha256'] != digest(args.manifest):
        raise ValueError('Actual audited manifest changed')
    for path, expected in validation['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Mixed evidence verifier changed')
    checks = validation['condition_checks']
    keys = [tuple(row['condition']) for row in checks]
    required = {(m, mode, d, s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS}
    if len(keys) != len(set(keys)) or not set(keys) <= required or len(keys) != validation['completed_conditions_or_pairs']:
        raise ValueError('Actual mixed condition inventory differs')
    groups = sum(all((m, mode, d, s) in keys for s in SEEDS) for m in MODELS for mode in MODES for d in DEVICES)
    if (validation['complete_three_seed_groups'] != groups or validation['matrix_complete'] != (len(keys) == 48)
            or validation['normal_thresholds_replayed'] != len(keys) * 8):
        raise ValueError('Mixed group/full/threshold declarations differ from actual condition coverage')
    expected_weights = {1.} if args.kind == 'lora' else {0., 1.}
    retained = 0
    Path('artifacts/tmp').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mixed-plot-replay-', dir='artifacts/tmp') as temporary:
        for row in checks:
            if len(row['treatments']) != len(expected_weights) or {p['teacher_weight'] for p in row['treatments']} != expected_weights:
                raise ValueError('Both actual teacher treatments required; no frozen substitution')
            for entry in row['treatments']:
                old_path = args.root / entry['condition_audit']
                if digest(old_path) != entry['condition_audit_sha256']:
                    raise ValueError('Mixed condition witness changed')
                _, fresh, _ = route_condition(tuple(row['condition']), entry['teacher_weight'],
                                              args.manifest, args.data_root, Path(temporary))
                new_path = Path(temporary) / fresh['condition_audit']
                if (fresh['evidence'] != entry['evidence'] or json.loads(new_path.read_text()) != json.loads(old_path.read_text())):
                    raise ValueError('Current tensor/metadata/trace evidence differs from saved mixed witness')
                retained += entry['evidence'] == 'archived_payloads_current_retained_replay'
    if retained != validation['retained_treatments_replayed']:
        raise ValueError('Historical payload evidence count differs')
    for name in ['device_summary', 'macro_summary']:
        if validation[name + '_sha256'] != digest(args.root / (name + '.json')):
            raise ValueError('Mixed summary changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['lora', 'teacher'], default='lora')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--primary-root', type=Path, default=Path('results/stage04'))
    parser.add_argument('--zero-root', type=Path, default=Path('results/stage05/ablations/teacher_weight/T0'))
    parser.add_argument('--frozen-root', type=Path, default=Path('results/stage02'))
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    args = parser.parse_args()
    validation = json.loads((args.root / 'validation.json').read_text())
    verify_mixed_sources(args, validation)
    full = validation['matrix_complete']
    summary = json.loads((args.root / ('macro_summary.json' if full else 'device_summary.json')).read_text())
    groups = group_rows(summary, args.kind, full)
    verify_plotted_statistics(args, summary, groups, full)
    selected = [('Frozen P3', 'F_P3'), ('LoRA P0', 'L_P0'), ('LoRA P3', 'L_P3')] if args.kind == 'lora' else [
        ('Teacher 1 P3', 'T1_P3'), ('Teacher 0 P3', 'T0_P3')]
    comparisons = [('LoRA P3 - Frozen P3', 'LoRA_minus_frozen_P3'), ('LoRA P3 - LoRA P0', 'LoRA_P3_minus_P0')] if args.kind == 'lora' else [
        ('Teacher 0 - Teacher 1', 'T0_minus_T1_P3')]
    figures = {}
    for (model, mode, device), rows in sorted(groups.items()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
        for axis, metric in zip(axes[:2], ['auroc', 'ap']):
            for x, (label, name), colour in zip(range(len(selected)), selected, ['#245B86', '#B97812', '#728348']):
                point, lo, hi = [rows[name][metric + suffix] * 100 for suffix in ['_mean', '_ci_low', '_ci_high']]
                axis.bar(x, point, color=colour, width=.55)
                axis.vlines(x, lo, hi, color='#333333')
                axis.hlines([lo, hi], x - .08, x + .08, color='#333333')
                axis.text(x, min(100, hi + 2), f'{point:.2f}', ha='center', fontsize=9)
            axis.set(xticks=range(len(selected)), xticklabels=[r[0] for r in selected], ylim=(0, 105), ylabel=metric.upper() + ' (%)')
        axis = axes[2]
        labels = []
        for index, (label, name) in enumerate(comparisons):
            pair = [r for r in summary['paired_deltas'] if r['backbone'] == model and r['mode'] == mode
                    and r['comparison'] == name and (full or r.get('device') == device)]
            if len(pair) != 1:
                raise ValueError('Exact current paired comparison required')
            for j, metric, colour in zip([0, 1], ['auroc', 'ap'], ['#245B86', '#B97812']):
                point, lo, hi = [pair[0][metric + suffix] * 100 for suffix in ['_delta', '_delta_ci_low', '_delta_ci_high']]
                y = index * 2 + j
                axis.plot(point, y, 'o', color=colour)
                axis.hlines(y, lo, hi, color=colour)
                labels.append(label + '\n' + metric.upper())
        axis.axvline(0, color='#666666', linestyle='--')
        axis.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels) - .5, -.5), xlabel='Paired difference (pp)')
        for axis in axes:
            axis.spines[['top', 'right']].set_visible(False)
        scope = 'equal-weight Macro4' if full else device + ' only / partial matrix'
        fig.suptitle(f'{NAMES[model]} / {mode} / {args.kind}\n{scope}', fontsize=13)
        note = ('Mean of exact seeds0/1/2 metrics; 1000 paired whole-video draws; percentile 95% CI. No runtime claim.\n'
                'Retired payload finite checks/hashes are archived history; metadata/selected tensors/bank/calibration/GT/score/alarm are replayed now.\n'
                'Source: ' + str(args.root / ('macro_summary.json' if full else 'device_summary.json')))
        stem = 'retained_' + args.kind + '_' + model + '_' + mode + ('_macro4' if full else '_' + device)
        figures.update(save(fig, args.out / stem, note))
    receipt = {'status': 'rendered_from_fresh_mixed_condition_and_statistic_replay', 'kind': args.kind,
               'matrix_complete': full, 'validation_sha256': digest(args.root / 'validation.json'),
               'plot_code_sha256': digest(__file__), 'figures': figures,
               'limits': 'Historical payload checks remain explicitly labelled; actual complete three-seed groups only. Final PNG review required.'}
    (args.root / 'figure_sources.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
