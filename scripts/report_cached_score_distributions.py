"""Explain fixed-q99 crossings from an immutable audited single-seed P3 pair."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ipad_jepa.experiment import alarms
from plot_neighbour_ablation import digest, save, NAMES
from report_single_cached_alarm_pair import verify_pair
from summarize_experiments import load_run, check_pair


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def describe(scores, threshold):
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Nonempty finite score population required')
    return {'targets': len(scores), 'score_min': float(np.min(scores)),
            'score_median': float(np.median(scores)),
            'score_q90': float(np.quantile(scores, .90)),
            'score_q99_descriptive_only': float(np.quantile(scores, .99)),
            'score_max': float(np.max(scores)),
            'strict_threshold_crossings': int(np.sum(scores > threshold))}


def collect(folder, audited_row):
    meta = json.loads((folder / 'metrics.json').read_text())
    fit = json.loads((folder / 'normal_fit.json').read_text())
    run = load_run(folder, 'P3', meta)
    threshold = fit['calibration']['P3']['threshold']
    calibration = read_csv(folder / 'normal_calibration.csv')
    calibration_scores = np.array([float(r['P3']) for r in calibration if r['valid'] == '1'])
    if len(calibration_scores) != fit['calibration_valid_frames']:
        raise ValueError('Valid score-calibration inventory differs')
    np.testing.assert_allclose(np.quantile(calibration_scores, .99), threshold, atol=1e-9, rtol=0)
    populations = {'normal_calibration': calibration_scores, 'known_normal_test': [],
                   'known_anomaly_test': []}
    counts = {key: {'targets': 0, 'strict_threshold_crossings': 0, 'streak_alarm_frames': 0}
              for key in ('known_normal_test', 'known_anomaly_test')}
    videos = {}
    inference_targets = unknown_targets = 0
    for video in run:
        trace = read_csv(folder / 'P3' / (video + '.csv'))
        scores = np.array([float(r['score']) for r in trace])
        inference = np.array([r['inference_valid'] == '1' for r in trace])
        valid = np.array([r['valid'] == '1' for r in trace])
        labels = np.array([int(r['label']) for r in trace])
        stored_alarm = np.array([int(r['alarm']) for r in trace], dtype=bool)
        if np.any(valid & ~inference) or not np.isfinite(scores[inference]).all():
            raise ValueError('Invalid inference/GT score population')
        # Replay on every original target before masking unknown GT. Strict > q99,
        # reset at invalid inference, and the original three-target streak.
        np.testing.assert_array_equal(alarms(scores, threshold, inference, 3), stored_alarm)
        np.testing.assert_array_equal(np.array([int(r['frame']) for r in trace])[valid], run[video][0])
        np.testing.assert_array_equal(labels[valid], run[video][1])
        np.testing.assert_array_equal(scores[valid], run[video][2])
        per_video = {}
        for key, label in [('known_normal_test', 0), ('known_anomaly_test', 1)]:
            keep = valid & (labels == label)
            values = scores[keep]
            populations[key].append(values)
            values_count = {'targets': int(keep.sum()),
                            'strict_threshold_crossings': int(np.sum(values > threshold)),
                            'streak_alarm_frames': int(stored_alarm[keep].sum())}
            if values_count['streak_alarm_frames'] > values_count['strict_threshold_crossings']:
                raise ValueError('Streak alarm without a threshold crossing')
            per_video[key] = values_count
            for name, value in values_count.items():
                counts[key][name] += value
        expected = audited_row['videos'][video]
        if (per_video['known_normal_test']['targets'] != expected['normal_targets'] or
                per_video['known_normal_test']['streak_alarm_frames'] != expected['normal_alarm_frames'] or
                int((inference & ~valid).sum()) != expected['unknown_targets'] or
                int(inference.sum()) != expected['inference_targets']):
            raise ValueError('Prior independently audited video accounting differs')
        inference_targets += int(inference.sum())
        unknown_targets += int((inference & ~valid).sum())
        videos[video] = per_video
    for key in counts:
        populations[key] = np.concatenate(populations[key])
        for name, value in describe(populations[key], threshold).items():
            if name in counts[key] and counts[key][name] != value:
                raise ValueError('Population count differs')
            counts[key][name] = value
        counts[key]['strict_crossing_rate'] = counts[key]['strict_threshold_crossings'] / counts[key]['targets']
        counts[key]['streak_alarm_frame_rate'] = counts[key]['streak_alarm_frames'] / counts[key]['targets']
    aggregate = audited_row['aggregate']
    if (counts['known_normal_test']['targets'] != aggregate['normal_targets'] or
            counts['known_normal_test']['streak_alarm_frames'] != aggregate['normal_alarm_frames'] or
            sum(r['targets'] for r in counts.values()) != meta['variants']['P3']['frames'] or
            counts['known_anomaly_test']['targets'] != meta['variants']['P3']['anomaly_frames'] or
            inference_targets != aggregate['inference_targets'] or
            unknown_targets != aggregate['unknown_targets'] or
            inference_targets != sum(r['targets'] for r in counts.values()) + unknown_targets or
            len(videos) != audited_row['test_videos']):
        raise ValueError('Aggregate denominator differs')
    summary = {'treatment': audited_row['treatment'], 'normal_only_q99_threshold': threshold,
               'calibration_total_trace_rows': len(calibration),
               'normal_calibration': describe(calibration_scores, threshold),
               'inference_targets': inference_targets, 'unknown_targets': unknown_targets,
               'test_videos': len(videos),
               **counts, 'per_video': videos}
    return summary, populations, run


def population_note(summaries):
    """Use replayed populations, refusing a caption that hides unequal coverage."""
    if len(summaries) != 2 or [r['treatment'] for r in summaries] != ['frozen', 'lora']:
        raise ValueError('One ordered frozen/LoRA population pair required')
    coverage = []
    for row in summaries:
        normal = row['known_normal_test']['targets']
        anomaly = row['known_anomaly_test']['targets']
        unknown = row['unknown_targets']
        inference = row['inference_targets']
        if min(normal, anomaly, row['test_videos']) <= 0 or unknown < 0 or inference != normal + anomaly + unknown:
            raise ValueError('Caption coverage denominator does not reconcile')
        if not 0 < row['normal_calibration']['targets'] <= row['calibration_total_trace_rows']:
            raise ValueError('Caption calibration denominator does not reconcile')
        coverage.append((normal, anomaly, unknown, inference, row['test_videos']))
    if coverage[0] != coverage[1]:
        raise ValueError('Caption pair coverage differs')
    normal, anomaly, unknown, inference, videos = coverage[0]
    valid = ' / '.join(f"{r['normal_calibration']['targets']:,}" for r in summaries)
    trace = ' / '.join(f"{r['calibration_total_trace_rows']:,}" for r in summaries)
    return (f'Score calibration (Frozen / LoRA): {valid} valid targets of {trace} trace rows; differs from phase-training counts.\n'
            f'Test: {videos} videos; {normal + anomaly:,} known-GT targets ({normal:,} normal / {anomaly:,} anomaly); '
            f'{unknown:,} unknown of {inference:,} inference targets.\n'
            'Unknown GT does not reset the streak. Frame alarm shares are not observed-event recall.\n'
            'Bottom panels share the percentage axis; ECDF panels show each method on its own score scale.\n'
            'Descriptive single-seed view; no threshold retuning, CI, causal attribution, measured FPS/delay or online EOF claim.')


def render(condition, summaries, populations, path):
    note = population_note(summaries)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), layout='constrained')
    colors = ['#245b86', '#36836c', '#bd463c']
    names = ['Normal calibration', 'Known-normal test', 'Known-anomaly test']
    for column, (summary, data) in enumerate(zip(summaries, populations)):
        axis = axes[0, column]
        for key, label, color in zip(data, names, colors):
            values = np.sort(data[key])
            axis.step(values, np.arange(1, len(values) + 1) / len(values),
                      where='post', color=color, label=f'{label} (n={len(values):,})')
        threshold = summary['normal_only_q99_threshold']
        axis.axvline(threshold, color='#333333', linestyle='--', linewidth=1.2,
                     label=f'Own normal q99 = {threshold:.4f}')
        axis.set_xscale('symlog', linthresh=1)
        axis.set(xlabel='P3 score (symlog; linear within [-1, 1])',
                 ylabel='Empirical cumulative fraction', ylim=(0, 1.04),
                 title=('Frozen P3' if column == 0 else 'LoRA P3') + ' / own score scale')
        axis.legend(frameon=True, facecolor='white', framealpha=.95, edgecolor='none',
                    fontsize=9, loc='lower right')
        axis = axes[1, column]
        for offset, key, label, color in [(-.18, 'strict_threshold_crossings', 'Score > own q99', '#245b86'),
                                          (.18, 'streak_alarm_frames', 'Original three-target alarm', '#b97812')]:
            for x, category in enumerate(['known_normal_test', 'known_anomaly_test']):
                row = summary[category]; count = row[key]; rate = count / row['targets'] * 100
                axis.bar(x + offset, rate, width=.34, color=color, label=label if x == 0 else None)
                axis.annotate(f'{count}/{row["targets"]:,}\n{rate:.3f}%',
                              (x + offset, rate), xytext=(0, 5), textcoords='offset points',
                              ha='center', va='bottom', fontsize=9)
        axis.set(xticks=[0, 1], xticklabels=['Known-normal test', 'Known-anomaly test'],
                 ylabel='Share of known-GT class targets (%)',
                 title='Fixed threshold crossings and stored alarm frames')
        axis.legend(frameon=False, fontsize=9, loc='upper left')
        for axis in axes[:, column]:
            axis.spines[['top', 'right']].set_visible(False)
    # Frame fractions have common units across treatments. Keep the common
    # zero-based axis; only the separately labeled ECDF score scales differ.
    largest_rate = max(row[category][count] / row[category]['targets'] * 100
                       for row in summaries
                       for category in ['known_normal_test', 'known_anomaly_test']
                       for count in ['strict_threshold_crossings', 'streak_alarm_frames'])
    for axis in axes[1]:
        axis.set_ylim(0, max(1, largest_rate * 1.5))
    model, mode, device, seed = condition
    fig.suptitle(f'{NAMES[model]} / {device} / {mode} / seed {seed} only\n'
                 'Immutable cached P3 distributions; original normal q99 and alarms replayed')
    return save(fig, path, note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or any(args.figure.with_suffix(e).exists() for e in ('.png', '.svg')):
        raise ValueError('Fresh report and figure destinations required')
    audited = json.loads(args.audit_report.read_text())
    if audited['status'] != 'passed_single_seed_pair_normal_GT_score_alarm_and_selected_lora_replay':
        raise ValueError('Existing completed source-audited pair required')
    model, mode, device, seed = audited['condition']
    if (model not in NAMES or mode not in ['offline', 'online'] or
            device not in ['R01', 'R02', 'R03', 'R04'] or
            type(seed) is not int or seed not in [0, 1, 2] or audited['seeds'] != 1):
        raise ValueError('One supported audited primary condition required')
    verify_pair(audited['conditions'])
    if any([r['backbone'], r['mode'], r['device'], r['seed']] != audited['condition']
           for r in audited['conditions']):
        raise ValueError('Audited condition and pair identity differ')
    sources = {**audited['source_sha256'], str(args.audit_report): digest(args.audit_report)}
    for name in [__file__, 'src/ipad_jepa/experiment.py', 'scripts/report_single_cached_alarm_pair.py',
                 'scripts/summarize_experiments.py', 'scripts/plot_neighbour_ablation.py']:
        relative = str(Path(name).resolve().relative_to(Path.cwd()))
        sources[relative] = digest(name)
    for name, expected in {**sources, **audited['annotation_sha256']}.items():
        if digest(name) != expected:
            raise ValueError('Audited input changed: ' + name)
    relative = Path(model) / mode / device / f'seed{seed}'
    summaries, populations, runs = [], [], []
    for root, row in zip(['results/stage02', 'results/stage04'], audited['conditions']):
        summary, population, run = collect(Path(root) / relative, row)
        summaries.append(summary); populations.append(population); runs.append(run)
    check_pair(*runs)
    note = population_note(summaries)
    figures = render(audited['condition'], summaries, populations, args.figure)
    for name, expected in {**sources, **audited['annotation_sha256']}.items():
        if digest(name) != expected:
            raise ValueError('Input changed during descriptive replay: ' + name)
    report = {'status': 'passed_original_q99_alarm_and_known_GT_distribution_replay',
              'condition': audited['condition'], 'variant': 'P3', 'seeds': 1,
              'conditions': summaries, 'source_sha256': sources,
              'annotation_sha256': audited['annotation_sha256'], 'figures': figures,
              'population_note': note,
              'limits': 'Descriptive reuse of immutable independently audited scores. Own normal q99 and strict > '
                        'three-target streak unchanged; unknown GT filtered only after alarm replay. No reencoding, '
                        'new PCA/prototype replay, alternative threshold, causal attribution, confidence intervals, '
                        'three-seed mean, measured runtime or operational online EOF coverage.'}
    args.out.mkdir(parents=True)
    (args.out / 'score_distributions.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'condition': report['condition'],
                      'conditions': [{k: v for k, v in r.items() if k != 'per_video'} for r in summaries]}, indent=2))


if __name__ == '__main__':
    main()
