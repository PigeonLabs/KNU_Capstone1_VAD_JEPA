"""CPU-only actual full-offline Macro4 exports; the global 48-condition matrix remains incomplete."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_retained_lora import verify_mixed_sources
from plot_lora_matrix import group_rows, verify_plotted_statistics
from plot_neighbour_ablation import save, NAMES
from summarize_experiments import load_run, macro_four_devices

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
MODELS = ('dinov3-l', 'vjepa21-l')
SELECTED = (('Frozen P3', 'F_P3'), ('LoRA P0', 'L_P0'), ('LoRA P3', 'L_P3'))
COLORS = ('#245B86', '#B97812', '#728348')


def pair(summary, backbone, comparison):
    rows = [r for r in summary['paired_deltas'] if r['backbone'] == backbone
            and r['mode'] == 'offline' and r['comparison'] == comparison]
    assert len(rows) == 1
    return rows[0]


def interval(axis, x, row, metric, color, marker='o'):
    point, lo, hi = [row[metric + suffix] * 100
                     for suffix in ('_delta', '_delta_ci_low', '_delta_ci_high')]
    axis.plot(point, x, marker, color=color)
    axis.hlines(x, lo, hi, color=color, linewidth=1.4)
    axis.vlines([lo, hi], x-.07, x+.07, color=color)


def bar(axis, x, row, metric, color, hatch=None):
    point, lo, hi = [row[metric + suffix] * 100
                     for suffix in ('_mean', '_ci_low', '_ci_high')]
    axis.bar(x, point, width=.32, color=color, hatch=hatch,
             edgecolor='#333333', linewidth=.5)
    axis.vlines(x, lo, hi, color='#333333')
    axis.hlines([lo, hi], x-.055, x+.055, color='#333333')
    axis.text(x, min(100, hi+2), f'{point:.2f}', ha='center', fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Output directory for three PNG/SVG pairs')
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    args = parser.parse_args()
    args.kind = 'lora'
    args.manifest = Path('results/stage00/manifest.json')
    args.primary_root = Path('results/stage04')
    args.frozen_root = Path('results/stage02')
    args.zero_root = Path('results/stage05/ablations/teacher_weight/T0')
    # This future semantic record must be bound to actual seed2 and prior M23 evidence.
    # Preparing this renderer does not create that evidence or launch any experiment.
    semantic = read('artifacts/tmp/R04_completed24_reporting_semantic_read_only_preparation.json')
    assert all(sha(n) == h for n, h in semantic['reviewed_original_producer_sha256'].items())
    assert all(sha(n) == h for n, h in semantic['preserved_M23_paths_sha256'].items())
    validation = read(args.root/'validation.json')
    assert not validation['matrix_complete']
    assert (validation['completed_conditions_or_pairs'], validation['complete_three_seed_groups'],
            validation['normal_thresholds_replayed'], validation['retained_treatments_replayed'],
            len(validation['pending_conditions_or_pairs'])) == (24, 8, 192, 1, 24)
    assert sorted(r['condition'] for r in validation['condition_checks']) == semantic['exact_expected_ready_conditions']
    prior = Path('results/stage04/retained_matrix_audit_R01_R02_R03_R04_partial23')
    summary, old = read(args.root/'device_summary.json'), read(prior/'device_summary.json')
    assert len(summary['results']) == 64 and len(summary['paired_deltas']) == 72
    assert all(r in summary['results'] for r in old['results'])
    assert all(r in summary['paired_deltas'] for r in old['paired_deltas'])
    macro, old_macro = read(args.root/'macro_summary.json'), read(prior/'macro_summary.json')
    assert len(macro['results']) == 16 and len(macro['paired_deltas']) == 18
    assert macro['incomplete_conditions'] == []
    assert macro['scope'] == old_macro['scope'] and macro['bootstrap'] == old_macro['bootstrap']
    assert all(r in macro['results'] for r in old_macro['results'])
    assert all(r in macro['paired_deltas'] for r in old_macro['paired_deltas'])
    groups = group_rows(macro, 'lora', True)
    assert set(groups) == {(m, 'offline', None) for m in MODELS}
    stems = [args.out/(m+'_offline_Macro4') for m in MODELS]
    stems.append(args.out/'offline_Macro4_backbone_comparison')
    assert all(not Path(str(stem)+ext).exists() for stem in stems
               for ext in ('.png', '.svg', '_sources.json'))
    verify_mixed_sources(args, validation)
    verify_plotted_statistics(args, macro, groups, True)
    # Replay the original Macro4 implementation for all 16 model/variant groups.
    # This preserves its common valid-draw mask and cross-backbone paired intervals.
    stored = {}
    for model in MODELS:
        for device in ('R01', 'R02', 'R03', 'R04'):
            for treatment, root in (('F', args.frozen_root), ('L', args.primary_root)):
                for variant in ('P0', 'P1', 'P2', 'P3'):
                    runs = []
                    for seed in (0, 1, 2):
                        folder = root/model/'offline'/device/f'seed{seed}'
                        metadata = read(folder/'metrics.json')
                        assert metadata['status'] == 'complete_device_evaluation'
                        assert (metadata['backbone'], metadata['mode'], metadata['device'], metadata['seed']) == (model, 'offline', device, seed)
                        runs.append(load_run(folder, variant, metadata))
                    stored[model, 'offline', device, treatment+'_'+variant] = (runs, None)
    replay = macro_four_devices(stored)
    assert replay['results'] == macro['results'] and replay['incomplete_conditions'] == []
    assert len(replay['paired_deltas']) == 8
    assert all(r in macro['paired_deltas'] for r in replay['paired_deltas'])
    note = ('Equal mean of four device means, each a mean of three fixed seed metrics; no pooled frames/scores.\n'
            '1000 original-video draws independently within each device; paired across conditions; percentile 95% CI; no seed resampling.\n'
            'Both offline Macro4 groups complete; global matrix 24/48, online LoRA incomplete. No measured runtime or decoder-only causal claim.\n'
            'Retired payload checks are historical; current metadata/tensors/traces replayed. Source: '+str(args.root/'macro_summary.json'))
    charts = []
    for model, stem in zip(MODELS, stems[:2]):
        rows = groups[model, 'offline', None]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout='constrained')
        for axis, metric in zip(axes[:2], ('auroc', 'ap')):
            for x, (_, key), color in zip(range(3), SELECTED, COLORS):
                bar(axis, x, rows[key], metric, color)
            axis.set(xticks=range(3), xticklabels=[r[0] for r in SELECTED],
                     ylim=(0, 105), ylabel=metric.upper()+' (%)')
        labels = []
        for index, (label, key) in enumerate((('LoRA P3 - Frozen P3', 'LoRA_minus_frozen_P3'),
                                              ('LoRA P3 - LoRA P0', 'LoRA_P3_minus_P0'))):
            row = pair(macro, model, key)
            for j, metric, color, marker in zip((0, 1), ('auroc', 'ap'), COLORS[:2], ('o', 's')):
                interval(axes[2], index*2+j, row, metric, color, marker)
                labels.append(label+'\n'+metric.upper())
        axes[2].axvline(0, color='#666666', linestyle='--')
        axes[2].set(yticks=range(4), yticklabels=labels, ylim=(3.5, -.5), xlabel='Paired difference (pp)')
        for axis in axes:
            axis.spines[['top', 'right']].set_visible(False)
        fig.suptitle(NAMES[model]+' / offline / equal-weight Macro4\nAll four devices, exact seeds 0/1/2; global matrix 24/48', fontsize=13)
        charts.append((stem, save(fig, stem, note), [model, 'offline']))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout='constrained')
    for axis, metric in zip(axes[:2], ('auroc', 'ap')):
        for index, (_, key) in enumerate(SELECTED):
            for j, model in enumerate(MODELS):
                bar(axis, index+(j-.5)*.38, groups[model, 'offline', None][key], metric,
                    COLORS[j], hatch=None if j == 0 else '//')
        axis.set(xticks=range(3), xticklabels=[r[0] for r in SELECTED], ylim=(0, 105), ylabel=metric.upper()+' (%)')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=COLORS[j], hatch=None if j == 0 else '//',
                                 label=NAMES[m]) for j, m in enumerate(MODELS)], loc='upper left')
    labels = []
    for index, (label, key) in enumerate(SELECTED):
        row = pair(macro, 'vjepa21-l_minus_dinov3-l', key+'_backbone')
        for j, metric, color, marker in zip((0, 1), ('auroc', 'ap'), COLORS[:2], ('o', 's')):
            interval(axes[2], index*2+j, row, metric, color, marker)
            labels.append(label+'\n'+metric.upper())
    axes[2].axvline(0, color='#666666', linestyle='--')
    axes[2].set(yticks=range(6), yticklabels=labels, ylim=(5.5, -.5), xlabel='V-JEPA 2.1-L minus DINOv3-L (pp)')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Offline backbone comparison / equal-weight Macro4\nAll four devices, exact seeds 0/1/2; global matrix 24/48', fontsize=13)
    charts.append((stems[2], save(fig, stems[2], note), [list(MODELS), 'offline']))
    for stem, figures, model_mode in charts:
        receipt = {'status': 'rendered_actual_complete_offline_Macro4_with_global_matrix_incomplete',
                   'global_matrix_complete': False, 'global_completed_conditions': 24,
                   'plotted_complete_model_mode': model_mode, 'devices': 4, 'seeds': 3,
                   'validation_sha256': sha(args.root/'validation.json'),
                   'macro_summary_sha256': sha(args.root/'macro_summary.json'),
                   'semantic_sha256': sha('artifacts/tmp/R04_completed24_reporting_semantic_read_only_preparation.json'),
                   'plot_code_sha256': sha(__file__),
                   'original_verifier_sha256': {n: sha(n) for n in ('scripts/plot_retained_lora.py', 'scripts/plot_lora_matrix.py', 'scripts/summarize_experiments.py')},
                   'original_macro_cross_backbone_pairs_replayed': 8,
                   'figures': figures, 'root_PNG_review_required': True}
        Path(str(stem)+'_sources.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
