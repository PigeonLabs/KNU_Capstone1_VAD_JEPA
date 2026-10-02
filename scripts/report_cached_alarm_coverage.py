"""Audit cached P3 alarm coverage; frame targets carry no measured emission times."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ipad_jepa.alignment import align
from lora_reporting import MODELS, MODES, DEVICES, SEEDS
from plot_lora_evaluation import calibration_check, read_csv
from plot_neighbour_ablation import digest, save, NAMES
from plot_retained_lora import verify_mixed_sources
from summarize_experiments import load_run


def cached_events(labels, frames, alarms):
    """Count observed GT segments and episode starts before any unknown-GT filtering."""
    labels, frames, alarms = map(np.asarray, (labels, frames, alarms))
    if (labels.ndim != 1 or not np.isin(labels, [-1, 0, 1]).all() or frames.ndim != 1
            or alarms.shape != frames.shape or not np.isin(alarms, [0, 1]).all()
            or not np.issubdtype(frames.dtype, np.integer) or np.any(np.diff(frames) <= 0)
            or np.any(frames < 0) or np.any(frames >= len(labels))):
        raise ValueError('Invalid cached frame/GT/alarm inventory')
    alarms = alarms.astype(bool)
    starts = alarms.copy()
    if len(starts) > 1:
        starts[1:] &= (~alarms[:-1]) | (np.diff(frames) != 1)
    normal = labels[frames] == 0
    unknown = labels[frames] == -1
    events, position = [], 0
    while position < len(labels):
        if labels[position] != 1:
            position += 1
            continue
        start = position
        while position < len(labels) and labels[position] == 1:
            position += 1
        end = position - 1
        selected = np.flatnonzero((frames >= start) & (frames <= end))
        hits = selected[alarms[selected]]
        events.append({'start_frame': int(start), 'end_frame': int(end),
                       'onset_uncertain': bool(start == 0 or labels[start - 1] == -1),
                       'offset_uncertain': bool(end == len(labels) - 1 or labels[end + 1] == -1),
                       'covered_frames': int(len(selected)), 'detected': bool(len(hits)),
                       'first_alarm_target': int(frames[hits[0]]) if len(hits) else None})
    covered = [r for r in events if r['covered_frames']]
    return {'events': events, 'covered_events': len(covered),
            'events_outside_coverage': len(events) - len(covered),
            'detected_events': sum(r['detected'] for r in covered),
            'missed_events': sum(not r['detected'] for r in covered),
            'uncertain_boundary_covered_events': sum(r['onset_uncertain'] or r['offset_uncertain'] for r in covered),
            'normal_alarm_episode_starts': int(np.sum(starts & normal)),
            'unknown_alarm_episode_starts': int(np.sum(starts & unknown)),
            'normal_alarm_frames': int(np.sum(alarms & normal)),
            'normal_targets': int(np.sum(normal)), 'unknown_targets': int(np.sum(unknown)),
            'inference_targets': int(len(frames))}


def aggregate(videos):
    names = ('covered_events', 'events_outside_coverage', 'detected_events', 'missed_events',
             'uncertain_boundary_covered_events', 'normal_alarm_episode_starts',
             'unknown_alarm_episode_starts', 'normal_alarm_frames', 'normal_targets',
             'unknown_targets', 'inference_targets')
    result = {name: sum(v[name] for v in videos.values()) for name in names}
    result['observed_event_recall'] = (result['detected_events'] / result['covered_events']
                                       if result['covered_events'] else None)
    result['normal_alarm_frame_rate'] = (result['normal_alarm_frames'] / result['normal_targets']
                                        if result['normal_targets'] else None)
    return result


def group_conditions(conditions):
    grouped = {}
    for row in conditions:
        key = tuple(row[k] for k in ('treatment', 'backbone', 'mode', 'device'))
        seeds = grouped.setdefault(key, {})
        if row['seed'] in seeds:
            raise ValueError('Duplicate seed condition')
        seeds[row['seed']] = row
    summaries = []
    for key, seeds in sorted(grouped.items()):
        if set(seeds) != {0, 1, 2}:
            continue
        for name in ('covered_events', 'normal_targets', 'unknown_targets', 'inference_targets'):
            if len({row['aggregate'][name] for row in seeds.values()}) != 1:
                raise ValueError('Seed coverage inventories differ')
        values = {name: [seeds[s]['aggregate'][name] for s in SEEDS]
                  for name in seeds[0]['aggregate']}
        summary = dict(zip(('treatment', 'backbone', 'mode', 'device'), key), seeds=[0, 1, 2])
        for name, data in values.items():
            if any(v is None for v in data):
                summary[name + '_mean'] = None
            else:
                summary[name + '_mean'] = float(np.mean(data))
                summary[name + '_seed_values'] = data
        summaries.append(summary)
    return summaries


def audit_condition(folder, treatment, manifest, data_root):
    meta = json.loads((folder / 'metrics.json').read_text())
    key = tuple(meta[k] for k in ('backbone', 'mode', 'device', 'seed'))
    if meta['status'] != 'complete_device_evaluation' or key not in {
            (m, mode, d, s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS}:
        raise ValueError('Complete primary method condition required')
    # Existing immutable audit checks every normal/test target, all P0-P3 raw/time/
    # normalized scores, actual annotations, normal q99 and original streak alarms.
    _, hashes = calibration_check(folder, meta, manifest, data_root)
    load_run(folder, 'P3', meta)
    sources = {str(folder / name): value for name, value in hashes.items()}
    videos = {}
    for row in manifest:
        if row['device'] != meta['device'] or row['partition'] != 'testing':
            continue
        labels, _, _ = align(np.load(data_root / row['label_file'], allow_pickle=False), row['frames'])
        trace = read_csv(folder / 'P3' / f"{row['sequence']}.csv")
        selected = [r for r in trace if r['inference_valid'] == '1']
        frames = np.array([int(r['frame']) for r in selected], dtype=np.int64)
        alarms = np.array([int(r['alarm']) for r in selected])
        np.testing.assert_array_equal(frames, np.arange(19, row['frames'] - 7))
        videos[row['sequence']] = cached_events(labels, frames, alarms)
    return dict(treatment=treatment, **{k: meta[k] for k in ('backbone', 'mode', 'device', 'seed')},
                threshold_fixed_on_normal=True, test_videos=len(videos), videos=videos,
                aggregate=aggregate(videos)), sources


def render(groups, out):
    figures = {}
    for treatment in ('frozen', 'lora'):
        rows = [r for r in groups if r['treatment'] == treatment]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, len(rows) * .43 + 1.5)),
                                 sharey=True, layout='constrained')
        labels = [f"{NAMES[r['backbone']]} / {r['mode']} / {r['device']}" for r in rows]
        for axis, metric in zip(axes, ('observed_event_recall', 'normal_alarm_frame_rate')):
            for i, row in enumerate(rows):
                values = row.get(metric + '_seed_values')
                if values is None:
                    axis.text(0, i, 'No denominator', va='center')
                    continue
                color = '#245b86' if row['backbone'] == 'dinov3-l' else '#b97812'
                axis.scatter(np.array(values) * 100, np.full(3, i), marker='x', s=30, color=color)
                axis.plot(row[metric + '_mean'] * 100, i, 'o', color=color)
            axis.set(yticks=range(len(rows)), yticklabels=labels, ylim=(len(rows) - .5, -.5),
                     xlabel=('Observed GT segment recall (%)' if metric == 'observed_event_recall'
                             else 'Known-normal alarm frame rate (%)'))
            axis.spines[['top', 'right']].set_visible(False)
        axes[0].set(xlim=(-3, 103))
        axes[1].set_xlim(left=-1)
        fig.suptitle(f'Cached P3 / {treatment} / normal-only q99 + three-target streak')
        note = ('Circles: mean of seeds0/1/2; crosses: individual seeds, not confidence intervals.\n'
                'Shared cached target19..N-8; unknown GT does not reset alarms. Observed GT segments may have uncertain boundaries.\n'
                'No measured emission time, FPS, wall-clock delay or operational online EOF coverage. Source: cached_alarm_coverage.json')
        figures.update(save(fig, out / ('cached_P3_' + treatment + '_alarm_coverage'), note))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, default=Path('results/stage02'))
    parser.add_argument('--primary-root', type=Path, default=Path('results/stage04'))
    parser.add_argument('--root', type=Path, default=Path('results/stage04/retained_matrix_audit'))
    parser.add_argument('--zero-root', type=Path, default=Path('results/stage05/ablations/teacher_weight/T0'))
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path, default=Path('results/stage05/cached_alarm_coverage'))
    parser.add_argument('--figures', type=Path, default=Path('docs/figures/stage05'))
    args = parser.parse_args(); args.kind = 'lora'
    manifest = json.loads(args.manifest.read_text())['sequences']
    validation = json.loads((args.root / 'validation.json').read_text())
    verify_mixed_sources(args, validation)
    required = {(m, mode, d, s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS}
    lora_required = {tuple(r['condition']) for r in validation['condition_checks']}
    sources = {str(args.manifest): digest(args.manifest), str(args.root / 'validation.json'): digest(args.root / 'validation.json'),
               **validation['source_sha256']}
    conditions = []
    for treatment, root, expected in [('frozen', args.frozen_root, required), ('lora', args.primary_root, lora_required)]:
        actual = set()
        for folder in sorted(root.glob('*/*/*/seed*')):
            path = folder / 'metrics.json'
            if not path.exists():
                continue
            meta = json.loads(path.read_text())
            if meta['status'] != 'complete_device_evaluation':
                continue
            key = tuple(meta[k] for k in ('backbone', 'mode', 'device', 'seed'))
            if key in actual or key not in expected:
                raise ValueError('Duplicate or unaudited completed primary condition')
            row, hashes = audit_condition(folder, treatment, manifest, args.data_root)
            actual.add(key); conditions.append(row); sources.update(hashes)
        if actual != expected:
            raise ValueError('Completed method condition inventory differs')
    annotations = {str(args.data_root / row['label_file']): row['label_sha256']
                   for row in manifest if row['partition'] == 'testing'}
    for name in ['scripts/report_cached_alarm_coverage.py', 'scripts/plot_lora_evaluation.py',
                 'scripts/plot_retained_lora.py', 'scripts/summarize_experiments.py',
                 'scripts/lora_reporting.py', 'src/ipad_jepa/alignment.py', 'src/ipad_jepa/temporal.py']:
        sources[name] = digest(name)
    for path, expected in {**sources, **annotations}.items():
        if digest(path) != expected:
            raise ValueError('Cached coverage source changed: ' + path)
    groups = group_conditions(conditions)
    report = {'status': 'passed_fresh_normal_threshold_score_GT_alarm_replay_and_cached_event_accounting',
              'variant': 'P3', 'frozen_conditions': 48, 'lora_conditions': len(lora_required),
              'lora_matrix_complete': len(lora_required) == 48,
              'conditions': conditions, 'three_seed_groups': groups, 'source_sha256': sources,
              'annotation_sha256': annotations,
              'limits': 'Shared cached target19..N-8 only; no measured emission times, delay, FPS, camera input or operational EOF coverage. '
                        'GT is read after inference; episode starts calculated before unknown-GT filtering. '
                        'R02 unresolved alignment uses existing bounded-offset consensus; observed segments touching unknown boundaries flagged. '
                        'Means are fixed three-seed metrics; individual seed values are not confidence intervals. '
                        'Normal q99 thresholds and three-target streaks are unchanged. No new training, feature extraction or PCA/search replay. '
                        'Retired DINO LoRA seed0 payload checks remain historical; retained/current condition replay performed.'}
    report['figures'] = render(groups, args.figures)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cached_alarm_coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('conditions', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
