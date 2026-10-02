"""Render matched DINOv3/V-JEPA LoRA groups with replayed paired statistics.

Only complete, independently audited three-seed groups are eligible. The mixed
source verifier explicitly preserves the distinction between current arrays and
archived payload evidence. This plot never changes the scientific producer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lora_reporting import MODELS, MODES, DEVICES, SEEDS
from plot_lora_matrix import group_rows, verify_plotted_statistics
from plot_neighbour_ablation import digest, save, NAMES
from plot_retained_lora import verify_mixed_sources
from summarize_experiments import bootstrap_draws, check_pair, load_run, statistic

SELECTED = ('F_P3', 'L_P0', 'L_P3')


def matched_pairs(groups, full):
    pairs = [(mode, None if full else device) for mode in MODES
             for device in ((None,) if full else DEVICES)
             if all((model, mode, None if full else device) in groups for model in MODELS)]
    if not pairs:
        raise ValueError('Both backbones require completed three-seed groups in the same mode/device')
    return pairs


def paired_statistics(reference, candidate, full):
    """Candidate-minus-reference with identical original-video draws per device."""
    required = set(DEVICES) if full else set(reference)
    if (set(reference) != set(candidate) or set(reference) != required
            or not full and (len(required) != 1 or not required <= set(DEVICES))):
        raise ValueError('Matched device strata; Macro4 requires all four devices')
    points, draws = [], []
    for device in sorted(required):
        a, b = reference[device], candidate[device]
        if len(a) != 3 or len(b) != 3:
            raise ValueError('Exactly seeds0/1/2 required on both sides')
        for run in [*a[1:], *b]:
            check_pair(a[0], run)
        points.append(np.mean([statistic(run, list(run)) for run in b], axis=0)
                      - np.mean([statistic(run, list(run)) for run in a], axis=0))
        seed = np.random.SeedSequence([2026, int(device[1:])]) if full else 2026
        draws.append(bootstrap_draws(b, seed=seed) - bootstrap_draws(a, seed=seed))
    values = np.mean(draws, axis=0)
    keep = np.isfinite(values).all(axis=1)
    if keep.sum() < 950:
        raise ValueError('Too many degenerate paired original-video draws')
    low, high = np.quantile(values[keep], [.025, .975], axis=0)
    return np.mean(points, axis=0), low, high, int((~keep).sum())


def verify_backbone_statistics(args, summary, groups, full):
    verified = {}
    for mode, device in matched_pairs(groups, full):
        devices = DEVICES if full else (device,)
        for name in SELECTED:
            root = args.frozen_root if name.startswith('F_') else args.primary_root
            runs = {}
            for model in MODELS:
                runs[model] = {}
                for current in devices:
                    seeds = []
                    for seed in SEEDS:
                        folder = root / model / mode / current / f'seed{seed}'
                        meta = json.loads((folder / 'metrics.json').read_text())
                        if (meta.get('status') != 'complete_device_evaluation'
                                or tuple(meta.get(k) for k in ('backbone', 'mode', 'device', 'seed'))
                                   != (model, mode, current, seed)):
                            raise ValueError('Current paired CSV condition differs')
                        seeds.append(load_run(folder, name.split('_')[1], meta))
                    runs[model][current] = seeds
            point, low, high, rejected = paired_statistics(runs[MODELS[0]], runs[MODELS[1]], full)
            matches = [r for r in summary['paired_deltas']
                       if r['backbone'] == 'vjepa21-l_minus_dinov3-l' and r['mode'] == mode
                       and r['comparison'] == name + '_backbone' and (full or r.get('device') == device)]
            if len(matches) != 1:
                raise ValueError('Exactly one audited candidate-minus-reference backbone difference required')
            row = matches[0]
            for index, metric in enumerate(('auroc', 'ap')):
                np.testing.assert_allclose([row[metric + suffix] for suffix in ('_delta', '_delta_ci_low', '_delta_ci_high')],
                                           [point[index], low[index], high[index]], rtol=0, atol=1e-9)
            verified[mode, device, name] = dict(row, bootstrap_rejected=rejected)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/stage04/retained_matrix_audit'))
    parser.add_argument('--primary-root', type=Path, default=Path('results/stage04'))
    parser.add_argument('--frozen-root', type=Path, default=Path('results/stage02'))
    parser.add_argument('--zero-root', type=Path, default=Path('results/stage05/ablations/teacher_weight/T0'))
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage04/retained'))
    args = parser.parse_args(); args.kind = 'lora'
    validation = json.loads((args.root / 'validation.json').read_text())
    full = validation['matrix_complete']
    summary_path = args.root / ('macro_summary.json' if full else 'device_summary.json')
    summary = json.loads(summary_path.read_text())
    groups = group_rows(summary, 'lora', full)
    pairs = matched_pairs(groups, full)
    verify_mixed_sources(args, validation)
    verify_plotted_statistics(args, summary, groups, full)
    comparisons = verify_backbone_statistics(args, summary, groups, full)
    figures = {}
    labels = ['Frozen P3', 'LoRA P0', 'LoRA P3']
    for mode, device in pairs:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), layout='constrained')
        for axis, metric in zip(axes[:2], ('auroc', 'ap')):
            for model, offset, color in zip(MODELS, (-.17, .17), ('#245B86', '#B97812')):
                rows = groups[model, mode, device]
                positions = np.arange(3) + offset
                points = [rows[name][metric + '_mean'] * 100 for name in SELECTED]
                low = [rows[name][metric + '_ci_low'] * 100 for name in SELECTED]
                high = [rows[name][metric + '_ci_high'] * 100 for name in SELECTED]
                axis.bar(positions, points, .30, color=color, label=NAMES[model])
                axis.vlines(positions, low, high, color='#25313c', linewidth=1.2)
                axis.hlines(low, positions - .04, positions + .04, color='#25313c')
                axis.hlines(high, positions - .04, positions + .04, color='#25313c')
                for x, value, hi in zip(positions, points, high):
                    axis.text(x, min(101, hi + 1.5), f'{value:.2f}', ha='center', fontsize=8)
            axis.set(xticks=np.arange(3), xticklabels=labels, ylim=(0, 117), ylabel=metric.upper() + ' (%)')
            axis.legend(frameon=False, fontsize=8, loc='upper left')
        ticklabels = []
        for index, name in enumerate(SELECTED):
            pair = comparisons[mode, device, name]
            for j, metric, color in zip((0, 1), ('auroc', 'ap'), ('#245B86', '#B97812')):
                y = index * 2 + j
                point, low, high = [pair[metric + suffix] * 100 for suffix in ('_delta', '_delta_ci_low', '_delta_ci_high')]
                axes[2].plot(point, y, 'o', color=color)
                axes[2].hlines(y, low, high, color=color, linewidth=2)
                ticklabels.append(labels[index] + ' / ' + metric.upper())
        axes[2].axvline(0, color='#7f8c94', linestyle='--', linewidth=1)
        axes[2].set(yticks=range(6), yticklabels=ticklabels, ylim=(5.5, -.5),
                    xlabel='V-JEPA minus DINOv3 (pp)', title='Paired 95% video-bootstrap CI')
        axes[2].tick_params(axis='y', labelsize=8)
        for axis in axes:
            axis.spines[['top', 'right']].set_visible(False)
        scope = 'equal-weight Macro4' if full else device + ' only / partial matrix'
        fig.suptitle(f'DINOv3-L vs V-JEPA 2.1-L / {mode} / {scope}', fontsize=13)
        note = ('Mean of exact seeds0/1/2 metrics; 1000 paired whole-video draws; seeds themselves are not resampled.\n'
                'Same evaluated targets/GT; separate normal-only heads, adapted encoders and rebuilt banks. No runtime claim.\n'
                'Current/retained evidence freshly replayed; retired array checks are historical. Source: ' + str(summary_path))
        figures.update(save(fig, args.out / ('lora_backbone_' + mode + ('_macro4' if full else '_' + device)), note))
    receipt = {'status': 'rendered_from_fresh_mixed_sources_and_paired_backbone_statistic_replay',
               'matrix_complete': full, 'matched_backbone_groups': len(pairs),
               'validation_sha256': digest(args.root / 'validation.json'), 'summary_sha256': digest(summary_path),
               'plot_code_sha256': digest(__file__), 'figures': figures,
               'source_sha256': {path: digest(path) for path in [
                   'scripts/plot_lora_backbone_comparison.py', 'scripts/plot_retained_lora.py',
                   'scripts/plot_lora_matrix.py', 'scripts/plot_neighbour_ablation.py',
                   'scripts/summarize_experiments.py', 'scripts/lora_reporting.py']},
               'paired_statistic_replay': list(comparisons.values()),
               'limits': 'Audited complete three-seed pairs only; partial device results are not Macro4. '
                         'No new training, encoder replay, runtime measurement, or seed resampling; final PNG inspection required.'}
    (args.root / 'backbone_figure_sources.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
