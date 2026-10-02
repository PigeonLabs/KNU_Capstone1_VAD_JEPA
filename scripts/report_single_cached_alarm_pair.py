"""Compare one audited frozen/LoRA P3 alarm pair on shared cached targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lora_reporting import MODELS, MODES, DEVICES, SEEDS
from plot_neighbour_ablation import digest, save, NAMES
from report_cached_alarm_coverage import audit_condition, aggregate
from summarize_retained_lora_matrix import route_condition


def verify_pair(rows):
    """A single-seed pair must have the same actual GT segments and coverage."""
    if len(rows) != 2 or [r['treatment'] for r in rows] != ['frozen', 'lora']:
        raise ValueError('Exactly one ordered frozen/LoRA pair required')
    identity = ('backbone', 'mode', 'device', 'seed')
    if any(rows[0][key] != rows[1][key] for key in identity):
        raise ValueError('Paired condition identity differs')
    coverage = ('covered_events', 'events_outside_coverage',
                'uncertain_boundary_covered_events', 'normal_targets',
                'unknown_targets', 'inference_targets')
    event_fields = ('start_frame', 'end_frame', 'onset_uncertain',
                    'offset_uncertain', 'covered_frames')
    for row in rows:
        if (not row['threshold_fixed_on_normal'] or row['test_videos'] != len(row['videos'])
                or row['aggregate'] != aggregate(row['videos'])):
            raise ValueError('Incomplete or inconsistent pair accounting')
    if rows[0]['videos'].keys() != rows[1]['videos'].keys():
        raise ValueError('Paired video inventory differs')
    for video in rows[0]['videos']:
        a, b = [r['videos'][video] for r in rows]
        if any(a[key] != b[key] for key in coverage):
            raise ValueError('Paired target coverage differs')
        events = [[{key: e[key] for key in event_fields} for e in v['events']] for v in (a, b)]
        if events[0] != events[1]:
            raise ValueError('Paired observed GT boundaries differ')


def render(rows, out):
    verify_pair(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    fields = ('observed_event_recall', 'normal_alarm_frame_rate', 'normal_alarm_episode_starts')
    labels = ('Observed GT segment recall (%)', 'Known-normal alarm frame rate (%)',
              'Normal alarm episode starts')
    for axis, field, label in zip(axes, fields, labels):
        for x, row, color in zip(range(2), rows, ('#245b86', '#b97812')):
            a = row['aggregate']; point = a[field]
            if point is None:
                axis.text(x, 0, 'No denominator', ha='center')
                continue
            height = point * 100 if field != fields[-1] else point
            axis.bar(x, height, width=.55, color=color)
            if field == fields[0]:
                text = f"{a['detected_events']}/{a['covered_events']}\n{height:.2f}%"
            elif field == fields[1]:
                text = f"{a['normal_alarm_frames']}/{a['normal_targets']}\n{height:.2f}%"
            else:
                text = str(point)
            axis.annotate(text, (x, height), xytext=(0, 5), textcoords='offset points',
                          ha='center', va='bottom', fontsize=10)
        axis.set(xticks=[0, 1], xticklabels=['Frozen P3', 'LoRA P3'], ylabel=label)
        axis.set_ylim(0, 100 if field == fields[0] else max(1., axis.get_ylim()[1] * 1.24))
        axis.spines[['top', 'right']].set_visible(False)
    meta = rows[0]
    fig.suptitle(f"{NAMES[meta['backbone']]} / {meta['device']} / {meta['mode']} / seed {meta['seed']} only\n"
                 'Cached P3: own fixed normal q99 + three-target streak')
    a = rows[0]['aggregate']
    note = (f"Single fixed seed; no three-seed mean or confidence intervals. Shared cached target19..N-8; "
            f"{a['uncertain_boundary_covered_events']} covered GT segments have uncertain boundaries.\n"
            'Unknown GT does not reset alarms; an alarm continuing into normal is not a new episode start.\n'
            'No measured emission times, FPS, wall-clock delay or operational online EOF coverage.')
    return save(fig, out, note)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--device', choices=DEVICES, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    args = p.parse_args()
    condition = (args.model, args.mode, args.device, args.seed)
    manifest = json.loads(args.manifest.read_text())['sequences']
    _, audit, _ = route_condition(condition, 1., args.manifest, args.data_root, args.out / 'audit')
    audit_path = args.out / 'audit' / audit['condition_audit']
    sources = {str(args.manifest): digest(args.manifest), str(audit_path): digest(audit_path)}
    rows = []
    relative = Path(args.model) / args.mode / args.device / f'seed{args.seed}'
    for treatment, root in [('frozen', Path('results/stage02')), ('lora', Path('results/stage04'))]:
        row, hashes = audit_condition(root / relative, treatment, manifest, args.data_root)
        rows.append(row); sources.update(hashes)
    verify_pair(rows)
    for name in ('scripts/report_single_cached_alarm_pair.py', 'scripts/report_cached_alarm_coverage.py',
                 'scripts/plot_lora_evaluation.py', 'scripts/summarize_retained_lora_matrix.py',
                 'scripts/summarize_lora_matrix.py', 'src/ipad_jepa/alignment.py', 'src/ipad_jepa/temporal.py'):
        sources[name] = digest(name)
    annotations = {str(args.data_root / r['label_file']): r['label_sha256']
                   for r in manifest if r['device'] == args.device and r['partition'] == 'testing'}
    figures = render(rows, args.figure)
    for name, expected in {**sources, **annotations}.items():
        if digest(name) != expected:
            raise ValueError('Source changed during pair replay: ' + name)
    report = dict(status='passed_single_seed_pair_normal_GT_score_alarm_and_selected_lora_replay',
                  condition=list(condition), variant='P3', seeds=1, conditions=rows,
                  independent_lora_audit=audit, source_sha256=sources,
                  annotation_sha256=annotations, figures=figures,
                  limits='One fixed trained seed, shared cached target19..N-8. Normal q99 and three-target '
                         'streak unchanged. Observed GT boundaries can be uncertain; continuing alarms count '
                         'for event coverage. No confidence intervals, three-seed mean, threshold retuning, '
                         'training-seed uncertainty, measured FPS/delay or operational EOF coverage. '
                         'Selected tensors/bank/calibration/GT/score/alarm replayed; no pixel reencoding or '
                         'independent PCA/prototype search replay.')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cached_alarm_pair.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'condition': report['condition'],
                      'aggregates': {r['treatment']: r['aggregate'] for r in rows}}, indent=2))


if __name__ == '__main__':
    main()
