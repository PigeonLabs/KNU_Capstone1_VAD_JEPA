"""Compare controlled native IPAD and feature P3 on identical three-seed targets.

This is a whole-protocol comparison. The native B1 encoder uses decoder training;
neither B1 inference nor the feature P3 methods reconstruct pixels. Native and
feature methods have different preprocessing, learning and memory protocols.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_neighbour_ablation import digest, save
from plot_retained_lora import verify_mixed_sources
from summarize_experiments import check_pair, load_run, statistic

METHODS = ('native_B0', 'native_B1', 'dinov3_F_P3', 'dinov3_L_P3',
           'vjepa21_F_P3', 'vjepa21_L_P3')
LABELS = ('IPAD B0 pixel', 'IPAD B1 latent', 'DINOv3 frozen P3', 'DINOv3 LoRA P3',
          'V-JEPA frozen P3', 'V-JEPA LoRA P3')


def check_sources(sources):
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError('Audited source changed: ' + str(path))


def replay_native(args):
    """Run the unchanged native validator in a private copy; preserve public plots."""
    witness_path = args.native_root / f'{args.device}_three_seed_validation.json'
    witness = json.loads(witness_path.read_text())
    if (witness['status'] != 'passed' or witness['scope'] != 'controlled'
            or witness['device'] != args.device or witness['seeds'] != [0, 1, 2]):
        raise ValueError('Completed controlled native three-seed witness required')
    check_sources(witness['sources'])
    with tempfile.TemporaryDirectory(prefix='native-feature-replay-', dir='artifacts/tmp') as tmp:
        copied = Path(tmp) / 'native'
        for name in witness['sources']:
            source = Path(name)
            if source.is_relative_to(args.native_root):
                destination = copied / source.relative_to(args.native_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        subprocess.run([sys.executable, 'scripts/plot_native_three_seed.py',
                        '--root', str(copied), '--device', args.device,
                        '--manifest', str(args.manifest), '--data-root', str(args.data_root),
                        '--out', str(Path(tmp) / 'figures')], check=True,
                       stdout=subprocess.DEVNULL)
        fresh = json.loads((copied / f'{args.device}_three_seed_validation.json').read_text())
        fields = ('status', 'device', 'scope', 'seeds', 'normal_thresholds_checked',
                  'threshold_checks', 'actual_annotations_checked', 'annotation_sha256',
                  'sklearn_bootstrap_float_values_compared', 'max_absolute_error',
                  'absolute_tolerance', 'bootstrap_draws', 'bootstrap_rejected')
        for key in fields:
            if fresh[key] != witness[key]:
                raise ValueError('Fresh native scientific replay differs: ' + key)
    return {key: fresh[key] for key in fields}, {
        **witness['sources'], str(witness_path): digest(witness_path)}


def matched_statistics(methods, count=1000, seed=2026):
    """Direct sklearn metrics per seed, shared whole-video draws across all methods."""
    if set(methods) != set(METHODS):
        raise ValueError('Exactly native B0/B1 and both frozen/LoRA P3 methods required')
    reference = methods['native_B0']
    if any(set(runs) != {0, 1, 2} for runs in methods.values()):
        raise ValueError('Exactly seeds0/1/2 required per method')
    for runs in methods.values():
        for run in runs.values():
            check_pair(reference[0], run)
    keys = list(reference[0])
    if not keys:
        raise ValueError('No evaluated videos')
    selected = np.random.default_rng(seed).choice(keys, (count, len(keys)), replace=True)
    points, draws = {}, {}
    for name in METHODS:
        runs = methods[name]
        points[name] = np.mean([statistic(runs[s], keys) for s in (0, 1, 2)], axis=0)
        draws[name] = np.array([np.mean([statistic(runs[s], chosen) for s in (0, 1, 2)], axis=0)
                               for chosen in selected])
    keep = np.logical_and.reduce([np.isfinite(v).all(axis=1) for v in draws.values()])
    if keep.sum() < .95 * count or any(not np.isfinite(p).all() for p in points.values()):
        raise ValueError('Too many degenerate paired video draws')
    def row(point, values, delta=False):
        low, high = np.quantile(values[keep], [.025, .975], axis=0)
        suffix = '_delta' if delta else '_mean'
        result = {}
        for i, metric in enumerate(('auroc', 'ap')):
            result[metric + suffix] = float(point[i])
            result[metric + ('_delta_ci_low' if delta else '_ci_low')] = float(low[i])
            result[metric + ('_delta_ci_high' if delta else '_ci_high')] = float(high[i])
        return result
    rows = [dict(method=name, seeds=[0, 1, 2], **row(points[name], draws[name])) for name in METHODS]
    pairs = [dict(candidate=name, reference=baseline,
                  **row(points[name] - points[baseline], draws[name] - draws[baseline], True))
             for name in METHODS[2:] for baseline in METHODS[:2]]
    return {'results': rows, 'paired_deltas': pairs, 'bootstrap_draws': count,
            'bootstrap_seed': seed, 'bootstrap_rejected': int((~keep).sum()),
            'test_videos': len(keys), 'common_frames': sum(len(v[1]) for v in reference[0].values()),
            'anomaly_frames': int(sum(v[1].sum() for v in reference[0].values()))}


def load_methods(args):
    methods, sources = {}, {}
    for name in METHODS:
        native = name.startswith('native_')
        model = 'IPAD-native-repaired' if native else ('dinov3-l' if name.startswith('dinov3') else 'vjepa21-l')
        variant = name.split('_')[-1]
        root = args.native_root if native else (args.frozen_root if '_F_' in name else args.primary_root)
        methods[name] = {}
        for seed in (0, 1, 2):
            folder = root / model / 'offline' / args.device / f'seed{seed}'
            meta = json.loads((folder / 'metrics.json').read_text())
            if (meta['status'] != 'complete_device_evaluation'
                    or tuple(meta[k] for k in ('backbone', 'mode', 'device', 'seed'))
                       != (model, 'offline', args.device, seed)):
                raise ValueError('Completed matching method condition required')
            methods[name][seed] = load_run(folder, variant, meta)
            for path in [folder / 'metrics.json', folder / 'normal_fit.json',
                         folder / 'normal_calibration.csv', *sorted((folder / variant).glob('*.csv'))]:
                sources[str(path)] = digest(path)
    return methods, sources


def render(report, out, device):
    rows = report['results']
    colors = ['#58666f', '#9c7040', '#80a8c3', '#245b86', '#d9b674', '#b97812']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, layout='constrained')
    for axis, metric in zip(axes, ('auroc', 'ap')):
        for index, (r, color) in enumerate(zip(rows, colors)):
            point, low, high = [r[metric + s] * 100 for s in ('_mean', '_ci_low', '_ci_high')]
            axis.hlines(index, low, high, color=color, linewidth=2)
            axis.plot(point, index, 'o', color=color)
            axis.text(103, index, f'{point:.2f}', va='center', fontsize=9)
        axis.set(yticks=range(6), yticklabels=LABELS, ylim=(5.6, -.6),
                 xlim=(0, 114), xticks=range(0, 101, 20), xlabel=metric.upper() + ' (%)')
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Controlled IPAD vs feature P3 / {device} offline / three-seed means')
    note = ('Same evaluated GT targets; 1000 paired whole-video draws; seeds0/1/2 fixed, not resampled.\n'
            'Whole protocols differ in preprocessing, encoder training and memory; no decoder-only causal claim.\n'
            'One device; no Macro4, encoder re-extraction or runtime claim. Source: native_feature_comparison.json')
    figures = save(fig, out / f'native_feature_{device}_accuracy', note)
    selected = [r for r in report['paired_deltas'] if '_L_' in r['candidate']]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, layout='constrained')
    labels = []
    for r in selected:
        model = 'DINOv3' if r['candidate'].startswith('dinov3') else 'V-JEPA'
        labels.append(model + ' LoRA P3 minus ' + r['reference'].replace('native_', 'IPAD '))
    for axis, metric in zip(axes, ('auroc', 'ap')):
        axis.axvline(0, color='#777777', linestyle=':')
        for index, r in enumerate(selected):
            point, low, high = [r[metric + s] * 100 for s in ('_delta', '_delta_ci_low', '_delta_ci_high')]
            color = '#245b86' if r['candidate'].startswith('dinov3') else '#b97812'
            axis.hlines(index, low, high, color=color, linewidth=2)
            axis.plot(point, index, 'o', color=color)
        axis.set(yticks=range(4), yticklabels=labels, ylim=(3.6, -.6), xlabel=metric.upper() + ' difference (pp)')
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Feature LoRA P3 minus controlled IPAD / {device} offline / paired 95% CI')
    figures.update(save(fig, out / f'native_feature_{device}_differences', note))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('R01', 'R02', 'R03', 'R04'), default='R01')
    parser.add_argument('--native-root', type=Path, default=Path('results/stage01/controlled'))
    parser.add_argument('--root', type=Path, default=Path('results/stage04/retained_matrix_audit'))
    parser.add_argument('--primary-root', type=Path, default=Path('results/stage04'))
    parser.add_argument('--frozen-root', type=Path, default=Path('results/stage02'))
    parser.add_argument('--zero-root', type=Path, default=Path('results/stage05/ablations/teacher_weight/T0'))
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--figures', type=Path, default=Path('docs/figures/comparisons'))
    args = parser.parse_args(); args.kind = 'lora'
    if args.out is None:
        args.out = Path('results/comparisons/native_feature') / args.device
    Path('artifacts/tmp').mkdir(parents=True, exist_ok=True)
    native, sources = replay_native(args)
    validation = json.loads((args.root / 'validation.json').read_text())
    required = {(m, 'offline', args.device, s) for m in ('dinov3-l', 'vjepa21-l') for s in (0, 1, 2)}
    actual = {tuple(row['condition']) for row in validation['condition_checks']}
    if not required <= actual:
        raise ValueError('Both feature backbones require all three independently audited LoRA seeds')
    verify_mixed_sources(args, validation)
    methods, traces = load_methods(args)
    sources.update(traces)
    sources.update(validation['source_sha256'])
    for path in [args.root / 'validation.json', args.manifest,
                 Path('scripts/compare_native_feature_methods.py'), Path('scripts/plot_retained_lora.py'),
                 Path('scripts/summarize_retained_lora_matrix.py')]:
        sources[str(path)] = digest(path)
    for row in validation['condition_checks']:
        for entry in row['treatments']:
            path = args.root / entry['condition_audit']
            sources[str(path)] = digest(path)
    report = dict(status='passed_fresh_native_and_mixed_feature_replay_with_direct_sklearn_comparison',
                  device=args.device, mode='offline', matrix_complete=False,
                  multiple_comparison_adjustment='none; pointwise 95% confidence intervals',
                  native_replay=native, **matched_statistics(methods), source_sha256=sources,
                  limits='Whole-protocol comparison of six methods on identical targets, each a mean of three fixed-seed metrics. '
                         'Shared whole-video percentile 95% CI; seeds are not resampled. '
                         'Native B1 uses decoder training; feature P3 has different preprocessing/training/memory. '
                         'No decoder-only causal effect, equivalence, paper-exact reproduction, Macro4 or runtime claim. '
                         'Private native checkpoint hashes and encoder extraction are not independently replayed. '
                         'Retired DINO seed0 feature payload checks are archived history; current retained artifacts replayed.')
    check_sources(sources)
    report['figures'] = render(report, args.figures, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'native_feature_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'native_replay')}, indent=2))


if __name__ == '__main__':
    main()
