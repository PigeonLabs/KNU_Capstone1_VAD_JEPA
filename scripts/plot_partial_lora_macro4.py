"""Future CPU-only Macro4 figure; global matrix remains explicitly incomplete."""
from pathlib import Path
import argparse, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_retained_lora import verify_mixed_sources
from plot_lora_matrix import group_rows, verify_plotted_statistics
from plot_neighbour_ablation import save

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    args = parser.parse_args()
    args.kind = 'lora'
    args.manifest = Path('results/stage00/manifest.json')
    args.primary_root = Path('results/stage04')
    args.frozen_root = Path('results/stage02')
    args.zero_root = Path('results/stage05/ablations/teacher_weight/T0')
    expected = read('artifacts/tmp/R04_completed21_reporting_semantic_read_only_preparation.json')
    assert all(sha(n) == h for n, h in expected['reviewed_original_producer_sha256'].items())
    assert all(sha(n) == h for n, h in expected['preserved_M20_paths_sha256'].items())
    validation = read(args.root / 'validation.json')
    assert not validation['matrix_complete']
    assert (validation['completed_conditions_or_pairs'], validation['complete_three_seed_groups'],
            validation['normal_thresholds_replayed'], validation['retained_treatments_replayed'],
            len(validation['pending_conditions_or_pairs'])) == (21, 7, 168, 1, 27)
    assert sorted(r['condition'] for r in validation['condition_checks']) == expected['exact_expected_ready_conditions']
    summary = read(args.root / 'device_summary.json')
    prior = read('results/stage04/retained_matrix_audit_R01_R02_R03_R04_partial20/device_summary.json')
    assert len(summary['results']) == 56 and len(summary['paired_deltas']) == 59
    assert all(r in summary['results'] for r in prior['results'])
    assert all(r in summary['paired_deltas'] for r in prior['paired_deltas'])
    macro = read(args.root / 'macro_summary.json')
    assert len(macro['results']) == 8 and len(macro['paired_deltas']) == 5
    old_macro = read('results/stage04/retained_matrix_audit_R01_R02_R03_R04_partial20/macro_summary.json')
    assert macro['scope'] == old_macro['scope'] and macro['bootstrap'] == old_macro['bootstrap']
    assert macro['incomplete_conditions'] == [r for r in old_macro['incomplete_conditions'] if r['backbone'] == 'vjepa21-l']
    assert len(macro['incomplete_conditions']) == 8
    groups = group_rows(macro, 'lora', True)
    assert set(groups) == {('dinov3-l', 'offline', None)}
    assert all(r['backbone'] == 'dinov3-l' and r['mode'] == 'offline' for r in macro['paired_deltas'])
    for ext in ('.png', '.svg', '_sources.json'):
        assert not Path(str(args.out) + ext).exists()
    # Original verifiers replay actual mixed evidence and all plotted statistics.
    # True here means four devices within this figure, never global matrix completeness.
    verify_mixed_sources(args, validation)
    verify_plotted_statistics(args, macro, groups, True)
    rows = groups['dinov3-l', 'offline', None]
    selected = [('Frozen P3', 'F_P3'), ('LoRA P0', 'L_P0'), ('LoRA P3', 'L_P3')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
    for axis, metric in zip(axes[:2], ('auroc', 'ap')):
        for x, (label, key), color in zip(range(3), selected, ('#245B86', '#B97812', '#728348')):
            point, lo, hi = [rows[key][metric + suffix] * 100 for suffix in ('_mean', '_ci_low', '_ci_high')]
            axis.bar(x, point, width=.55, color=color)
            axis.vlines(x, lo, hi, color='#333333')
            axis.hlines([lo, hi], x - .08, x + .08, color='#333333')
            axis.text(x, min(100, hi + 2), f'{point:.2f}', ha='center', fontsize=9)
        axis.set(xticks=range(3), xticklabels=[x[0] for x in selected], ylim=(0, 105), ylabel=metric.upper() + ' (%)')
    labels = []
    for index, (label, key) in enumerate((('LoRA P3 - Frozen P3', 'LoRA_minus_frozen_P3'), ('LoRA P3 - LoRA P0', 'LoRA_P3_minus_P0'))):
        matched = [r for r in macro['paired_deltas'] if r['comparison'] == key]
        assert len(matched) == 1
        for j, metric, color in zip((0, 1), ('auroc', 'ap'), ('#245B86', '#B97812')):
            point, lo, hi = [matched[0][metric + suffix] * 100 for suffix in ('_delta', '_delta_ci_low', '_delta_ci_high')]
            axes[2].plot(point, index * 2 + j, 'o', color=color)
            axes[2].hlines(index * 2 + j, lo, hi, color=color)
            labels.append(label + '\n' + metric.upper())
    axes[2].axvline(0, color='#666666', linestyle='--')
    axes[2].set(yticks=range(4), yticklabels=labels, ylim=(3.5, -.5), xlabel='Paired difference (pp)')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('DINOv3-L / offline / equal-weight Macro4\nAll four devices, exact seeds 0/1/2; global matrix 21/48', fontsize=13)
    note = ('Equal mean of four device means, each a mean of three seed metrics; no pooled frames/scores.\n'
            '1000 original-video draws independently inside each device; paired across variants/treatments; percentile 95% CI.\n'
            'V-JEPA Macro4 and online Macro4 remain incomplete. No measured runtime or decoder-only causal claim.\n'
            'Source: ' + str(args.root / 'macro_summary.json'))
    figures = save(fig, args.out, note)
    receipt = {'status': 'rendered_actual_complete_DINO_offline_Macro4_with_global_matrix_incomplete',
               'global_matrix_complete': False, 'global_completed_conditions': 21,
               'plotted_complete_model_mode': ['dinov3-l', 'offline'], 'devices': 4, 'seeds': 3,
               'validation_sha256': sha(args.root / 'validation.json'),
               'macro_summary_sha256': sha(args.root / 'macro_summary.json'),
               'plot_code_sha256': sha(__file__), 'original_verifier_sha256': {n: sha(n) for n in ('scripts/plot_retained_lora.py', 'scripts/plot_lora_matrix.py')},
               'figures': figures, 'root_PNG_review_required': True}
    Path(str(args.out) + '_sources.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)

if __name__ == '__main__':
    main()
