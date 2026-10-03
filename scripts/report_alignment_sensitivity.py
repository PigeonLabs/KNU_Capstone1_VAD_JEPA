"""Replay existing metrics under bounded annotation offsets; no model execution."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ipad_jepa.alignment import POLICY, align
from ipad_jepa.temporal import common_mask
from plot_neighbour_ablation import digest, save
from summarize_experiments import check_pair, load_run


def metric(labels, scores):
    labels, scores = np.concatenate(labels), np.concatenate(scores)
    if not np.all(np.isfinite(scores)) or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Finite scores and both binary classes required')
    return dict(frame_auroc=float(roc_auc_score(labels, scores)),
                frame_ap=float(average_precision_score(labels, scores)),
                frames=len(labels), anomaly_frames=int(labels.sum()))


def verify(value, expected):
    if (value['frames'], value['anomaly_frames']) != (expected['frames'], expected['anomaly_frames']):
        raise ValueError('Metric inventory differs')
    np.testing.assert_allclose([value[k] for k in ('frame_auroc', 'frame_ap')],
                               [expected[k] for k in ('frame_auroc', 'frame_ap')], atol=1e-9, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['dinov3-l', 'vjepa21-l'], required=True)
    parser.add_argument('--mode', choices=['offline', 'online'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--device', choices=['R02'], default='R02')
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--audit-report', type=Path, required=True,
                        help='Completed independent cached-alarm pair audit for these exact conditions')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    target = [args.model, args.mode, args.device, args.seed]
    report_path = args.out / 'alignment_sensitivity.json'
    if report_path.exists() or args.figure.with_suffix('.png').exists():
        raise ValueError('Fresh output namespace required')
    audit = json.loads(args.audit_report.read_text())
    if audit['status'] != 'passed_single_seed_pair_normal_GT_score_alarm_and_selected_lora_replay' or audit['condition'] != target:
        raise ValueError('Matching completed independent source audit required')
    sources = {**audit['source_sha256'], str(args.audit_report): digest(args.audit_report),
               str(args.manifest): digest(args.manifest), str(Path(__file__).relative_to(Path.cwd())): digest(__file__)}
    if not all(digest(n) == h for n, h in sources.items()):
        raise ValueError('Previously audited source changed')
    annotations = dict(audit['annotation_sha256'])
    if not all(digest(n) == h for n, h in annotations.items()):
        raise ValueError('Audited original annotation changed')
    rows = [r for r in json.loads(args.manifest.read_text())['sequences']
            if r['device'] == args.device and r['partition'] == 'testing']
    if len(rows) != 15:
        raise ValueError('All 15 R02 test videos required')
    labels_by_video = {}
    for row in rows:
        path = args.data_root / row['label_file']
        if digest(path) != row['label_sha256'] or str(path) not in annotations:
            raise ValueError('Annotation differs from original manifest/audit')
        labels_by_video[row['sequence']] = align(np.load(path, allow_pickle=False), row['frames'])
    outputs, paired = [], {}
    for treatment, stage in [('frozen', 'stage02'), ('lora', 'stage04')]:
        folder = Path('results') / stage / args.model / args.mode / args.device / f'seed{args.seed}'
        meta_path = folder / 'metrics.json'
        sources[str(meta_path)] = digest(meta_path)
        meta = json.loads(meta_path.read_text())
        if meta['status'] != 'complete_device_evaluation' or [meta[k] for k in ('backbone', 'mode', 'device', 'seed')] != target or meta['alignment_policy'] != POLICY:
            raise ValueError('Completed condition/policy mismatch')
        for variant in ['P0', 'P1', 'P2', 'P3']:
            primary = load_run(folder, variant, meta)
            if variant in paired:
                check_pair(paired[variant], primary)
            paired[variant] = primary
            labels = {o: [] for o in (-1, 0, 1)}
            scores = {o: [] for o in (-1, 0, 1)}
            for row in rows:
                trace_path = folder / variant / f"{row['sequence']}.csv"
                if str(trace_path) not in audit['source_sha256']:
                    raise ValueError('Score trace absent from independent audit')
                with trace_path.open() as stream:
                    trace = list(csv.DictReader(stream))
                frame = np.array([int(r['frame']) for r in trace])
                inferred = np.array([r['inference_valid'] == '1' for r in trace])
                np.testing.assert_array_equal(frame[inferred], np.arange(19, row['frames'] - 7))
                np.testing.assert_array_equal(inferred, common_mask(row['frames'])[frame])
                consensus, known, candidates = labels_by_video[row['sequence']]
                np.testing.assert_array_equal([int(r['label']) for r in trace], consensus[frame])
                np.testing.assert_array_equal([r['valid'] == '1' for r in trace], inferred & known[frame])
                score = np.array([float(r['score']) for r in trace])
                for offset in (-1, 0, 1):
                    alternate = candidates.get(offset, candidates[0])[frame]
                    keep = inferred & (alternate >= 0)
                    labels[offset].append(alternate[keep]); scores[offset].append(score[keep])
            values = {}
            for offset in (-1, 0, 1):
                values[str(offset)] = metric(labels[offset], scores[offset])
                verify(values[str(offset)], meta['alignment_sensitivity'][variant][str(offset)])
            outputs.append(dict(treatment=treatment, variant=variant,
                                consensus=meta['variants'][variant], forced_offsets=values))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), layout='constrained')
    for axis, name, title in zip(axes, ['frame_auroc', 'frame_ap'], ['AUROC', 'AP']):
        for treatment, color, marker in [('frozen', '#245b86', 's'), ('lora', '#b97812', 'o')]:
            row = next(r for r in outputs if r['treatment'] == treatment and r['variant'] == 'P3')
            values = [row['forced_offsets'][str(o)][name] * 100 for o in (-1, 0, 1)]
            axis.plot([0, 1, 2], values, color=color, marker=marker, label=treatment.title())
            axis.scatter(3.4, row['consensus'][name] * 100, color=color, marker=marker)
            for x, value in zip([0, 1, 2, 3.4], values + [row['consensus'][name] * 100]):
                axis.annotate(f'{value:.3f}', (x, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color=color)
        axis.axvline(2.7, linestyle=':', color='#aaa')
        axis.set(xticks=[0, 1, 2, 3.4], xticklabels=['-1', '0', '+1', 'Consensus'],
                 xlim=(-.4, 3.9), ylabel=title + ' (%)', xlabel='Assumed label index offset on R02/12,13,14', title='P3 / ' + title)
        axis.margins(y=.22); axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f'{args.model} / {args.mode} / R02 seed{args.seed} / bounded-offset sensitivity', fontsize=13)
    note = ('Forced offsets: 9,228 targets; anomaly counts -1/0/+1 = 2,959/2,958/2,957. Consensus: 9,210 targets, 2,949 anomalies; 18 unknown excluded.\n'
            'Offsets shift only length-mismatched annotations; matched videos retain index alignment. Consensus is a separate denominator, not a fourth offset.\n'
            'Same completed scores, no model execution or retuning. Single fixed seed, no CI; exact alignment and offsets beyond +/-1 remain unproven.')
    figures = save(fig, args.figure, note)
    if not all(digest(n) == h for n, h in {**sources, **annotations}.items()):
        raise ValueError('Sources changed during replay')
    output = dict(status='passed_original_label_and_score_replay_for_all_P0_P3_offsets', condition=target,
                  alignment_policy=POLICY, exact_alignment_confirmed=False, seeds=1,
                  results=outputs, source_sha256=sources, annotation_sha256=annotations,
                  figures=figures, scope=note)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(status=output['status'], conditions=len(outputs), offset_metrics=len(outputs)*3,
                          source_pins=len(sources), annotation_pins=len(annotations)), indent=2), flush=True)


if __name__ == '__main__':
    main()
