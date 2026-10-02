"""Audit and visualize individual clip8/16 seed pairs before a three-seed group exists."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ipad_jepa.cache_retirement import atomic_record
from summarize_clip_ablation import audit_clip8_source, audit_condition, digest
from summarize_experiments import check_pair, statistic


def partial_seeds(seeds):
    if (not seeds or len(seeds) > 2 or len(set(seeds)) != len(seeds)
            or any(seed not in (0, 1, 2) for seed in seeds)):
        raise ValueError('One or two distinct fixed seeds required; complete groups use summarize_clip_ablation.py')
    return sorted(seeds)


def pair_points(runs, seed):
    check_pair(runs[16], runs[8])
    rows = []
    for frames in (16, 8):
        run = runs[frames]
        value = statistic(run, list(run))
        if not np.isfinite(value).all():
            raise ValueError('Finite paired individual metrics required')
        rows.append({'seed': seed, 'clip_frames': frames, 'score_variant': 'P3',
                     'test_videos': len(run), 'frames': sum(len(item[1]) for item in run.values()),
                     'anomaly_frames': int(sum(item[1].sum() for item in run.values())),
                     'auroc': float(value[0]), 'ap': float(value[1])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('dinov3-l', 'vjepa21-l'), required=True)
    parser.add_argument('--mode', choices=('offline', 'online'), required=True)
    parser.add_argument('--device', choices=('R01', 'R02', 'R03', 'R04'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--root', type=Path, default=Path('results/stage05/ablations/clip_frames'))
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    seeds = partial_seeds(args.seeds)
    out = args.root / 'individual_pairs' / args.model / args.mode / args.device
    out.mkdir(parents=True, exist_ok=True)
    (out / 'validation.json').unlink(missing_ok=True)
    code_paths = [Path(__file__).resolve().relative_to(Path.cwd().resolve()), args.manifest,
                  Path('scripts/summarize_clip_ablation.py'), Path('scripts/summarize_experiments.py'),
                  Path('src/ipad_jepa/alignment.py'), Path('src/ipad_jepa/temporal.py')]
    sources = {str(path): digest(path) for path in code_paths}
    manifest = json.loads(args.manifest.read_text())['sequences']
    checks, points = [], []
    for seed in seeds:
        relative = Path(args.model) / args.mode / args.device / f'seed{seed}'
        folder8 = args.root / 'T8' / relative
        local = Path('artifacts/runs_clip8') / relative
        source = audit_clip8_source(folder8, local, Path('artifacts/features_clip8') / args.model / args.mode,
                                   manifest, args.model, args.mode, args.device, seed, args.manifest)
        with np.load(local / 'memory.npz', allow_pickle=False) as bank:
            if (bank['mean'].shape != (1024,) or bank['components'].shape != (256, 1024)
                    or bank['prototypes'].shape != (16, 128, 256)
                    or not all(np.isfinite(bank[name]).all() for name in ('mean', 'components', 'prototypes'))):
                raise ValueError('Actual refitted clip8 bank geometry or values differ')
            normal = json.loads((folder8 / 'normal_fit.json').read_text())
            np.testing.assert_allclose([float(bank['temperature']), float(bank['cycle_length'])],
                                       [normal['temperature'], normal['cycle_length_fit_median']], rtol=0, atol=0)
        runs = {}
        for frames, folder in [(16, Path('results/stage02') / relative), (8, folder8)]:
            runs[frames], check = audit_condition(folder, manifest, args.data_root, args.model, args.mode,
                                                  args.device, seed, frames, source if frames == 8 else None)
            checks.append(check)
            for path in [folder / 'metrics.json', folder / 'normal_fit.json', folder / 'normal_calibration.csv',
                         *sorted((folder / 'P3').glob('*.csv'))]:
                sources[str(path)] = digest(path)
        for path in [folder8 / 'clip8_source_proof.json', folder8 / 'phase_training.json', folder8 / 'phase_training.csv']:
            sources[str(path)] = digest(path)
        for name, value in source[0]['source_sha256'].items():
            name = 'scripts/run_clip8_matrix.py' if Path(name).is_absolute() else name
            if digest(name) != value:
                raise ValueError('Producer source changed')
            sources[name] = value
        points.extend(pair_points(runs, seed))
        print(f'Individual seed pair independently verified: {relative}', flush=True)
    scope = ('P3 individual fixed-seed metrics on shared t=19..N-8/annotation mask; per-context normal '
             'head/PCA/memory/calibration refitted. No seed mean, bootstrap CI, Macro4 or runtime claim.')
    summary = {'scope': scope, 'backbone': args.model, 'mode': args.mode, 'device': args.device,
               'seeds': seeds, 'missing_seeds': [seed for seed in (0, 1, 2) if seed not in seeds],
               'matrix_complete': False, 'results': points}
    atomic_record(out / 'points.json', summary)
    with (out / 'points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(points)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout='constrained')
    for axis, metric in zip(axes, ('auroc', 'ap')):
        for offset, frames, color in [(-.19, 16, '#3979ae'), (.19, 8, '#db8b37')]:
            values = [next(row[metric] * 100 for row in points if row['seed'] == seed and row['clip_frames'] == frames) for seed in seeds]
            bars = axis.bar(np.arange(len(seeds)) + offset, values, width=.36, color=color, label=f'{frames} frames')
            axis.bar_label(bars, fmt='%.2f', padding=4, fontsize=9)
        axis.set(xticks=np.arange(len(seeds)), xticklabels=[f'Seed {seed}' for seed in seeds],
                 ylim=(0, 100), ylabel=metric.upper() + ' (%)')
        axis.grid(axis='y', alpha=.18); axis.set_axisbelow(True)
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle(f'{args.model} / {args.mode} / {args.device}\nIndividual seed pairs; remaining seed pending', fontsize=13)
    count = points[0]
    fig.supxlabel(f"P3; {count['test_videos']} videos / {count['frames']:,} shared frames / {count['anomaly_frames']:,} anomaly frames.\n"
                  'Separate normal head/memory/calibration; no three-seed mean, CI, Macro4 or FPS.', fontsize=9)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(args.figure.with_suffix('.' + extension), dpi=180)
    plt.close(fig)
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('Audited partial-pair inputs changed')
    validation = {'status': 'passed_individual_clip_seed_pairs', 'matrix_complete': False,
                  'completed_seed_pairs': len(seeds), 'seeds': seeds, 'missing_seeds': summary['missing_seeds'],
                  'normal_thresholds_replayed': len(checks), 'checks': checks, 'source_sha256': sources,
                  'points_sha256': digest(out / 'points.json'), 'csv_sha256': digest(out / 'points.csv'),
                  'scope': scope,
                  'limits': 'Checks actual clip8 cache metadata/targets/shapes, selected head/bank hashes and geometry, full20 normal CE selection, calibration/GT/P3/time/alarm traces. Does not independently rerun encoders/PCA/GPU feature distances or runtime.'}
    atomic_record(out / 'validation.json', validation)
    atomic_record(out / 'figure_sources.json', {'status': 'rendered_audited_individual_seed_pairs',
                  'validation_sha256': digest(out / 'validation.json'), 'points_sha256': validation['points_sha256'],
                  'plot_code_sha256': digest(__file__), 'figures': {str(args.figure.with_suffix('.' + extension)):
                  digest(args.figure.with_suffix('.' + extension)) for extension in ('png', 'svg')}, 'scope': scope})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
