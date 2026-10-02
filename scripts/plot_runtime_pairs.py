"""Export measured runtime pairs with unchanged failed numerical gates made explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_runtime import audit, digest
from verify_runtime_parity import compare


def verified_pairs(root, manifest, data_root):
    """Require fresh trace audits and the exact saved all-target parity, including failures."""
    reports, proofs, parities, sources = {}, {}, {}, {}
    for precision, candidate in [('bf16', 'buffer'), ('fp32', 'reuse')]:
        for implementation in ['full', candidate]:
            folder = root / precision / implementation
            proof_path = folder / 'trace_audit.json'
            proof = json.loads(proof_path.read_text())
            if proof != audit(folder, manifest, data_root):
                raise ValueError('Actual runtime trace audit changed')
            reports[(precision, implementation)] = json.loads((folder / 'runtime.json').read_text())
            proofs[(precision, implementation)] = proof
            sources[str(proof_path)] = digest(proof_path)
            sources.update(proof['sources_sha256'])
        path = root / precision / 'parity.json'
        parity = json.loads(path.read_text())
        if parity != compare(root / precision / 'full', root / precision / candidate):
            raise ValueError('Saved measured parity differs; preserve the actual failed gate')
        parities[precision] = parity
        sources[str(path)] = digest(path)
    first = reports[('bf16', 'full')]
    common = ['backbone', 'mode', 'device', 'seed', 'adaptation', 'variant', 'arrival_fps',
              'normal_fit_sha256', 'phase_head_sha256', 'memory_sha256', 'weights_sha256',
              'manifest_sha256', 'runtime_code_sha256', 'score_state_sha256',
              'selected_adapter_sha256', 'teacher_weight', 'all_test_videos']
    for report in reports.values():
        if any(key not in first or key not in report or first[key] != report[key] for key in common):
            raise ValueError('Cross-precision comparison uses different selected conditions')
    code_path = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    sources[str(code_path)] = digest(code_path)
    return reports, proofs, parities, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='One DINO online condition containing bf16 and fp32 measured pairs')
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path, required=True, help='PNG/SVG/JSON figure stem')
    args = parser.parse_args()
    reports, proofs, parities, sources = verified_pairs(args.root, args.manifest, args.data_root)
    test_videos = proofs[('bf16', 'full')]['test_videos']
    blue, gold, red = '#245B86', '#B97812', '#A43232'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    plotted = []
    for column, (precision, candidate) in enumerate([('bf16', 'buffer'), ('fp32', 'reuse')]):
        parity = parities[precision]
        passed = parity['status'] == 'passed'
        labels = ['Full clip', 'GPU pixel buffer' if candidate == 'buffer' else 'Past feature reuse']
        values = [reports[(precision, implementation)] for implementation in ['full', candidate]]
        colors = [blue, gold if passed else red]
        for row, key, ylabel, title in [
                (0, 'sustained_input_fps', 'Completed input FPS', 'Paced FIFO throughput'),
                (1, 'target_latency_p95_ms', 'p95 target latency (ms, log scale)', 'Arrival-to-output latency')]:
            ax = axes[row, column]
            numbers = [report[key] for report in values]
            bars = ax.bar([0, 1], numbers, color=colors, width=.56)
            if not passed:
                bars[1].set_hatch('//')
            ax.set_xticks([0, 1], labels)
            ax.set(title=f'{precision.upper()}: {title}', ylabel=ylabel)
            if row == 0:
                ax.axhline(values[0]['arrival_fps'], color='#444444', linestyle='--',
                           linewidth=1.1, label='30 FPS arrival')
                ax.set_ylim(0, values[0]['arrival_fps'] * 1.25)
                ax.legend(frameon=False, loc='upper left', fontsize=9)
            else:
                ax.set_yscale('log')
                ax.set_ylim(min(numbers) / 3, max(numbers) * 3)
            for bar, number in zip(bars, numbers):
                ax.annotate(f'{number:,.2f}', (bar.get_x() + bar.get_width() / 2, number),
                            xytext=(0, 6), textcoords='offset points', ha='center', fontsize=11)
            ax.grid(axis='y', color='#DDDDDD', linewidth=.7, alpha=.7)
            ax.set_axisbelow(True)
        failed = [row['sequence'] for row in parity['sequences'] if not row['passed']]
        mismatch = sum(row['alarm_mismatches'] for row in parity['sequences'])
        fig.text(.08 + .48 * column, .09,
                 f'Score / alarm gate: {parity["status"].upper()} | alarm mismatches: {mismatch}\n'
                 + (f'Failed score videos: {", ".join(failed)}. Reuse equivalence is not established.'
                    if failed else f'All {test_videos} videos pass the fixed score and alarm gates.'),
                 color=red if failed else '#333333', fontsize=9)
        for implementation, report in zip(['full', candidate], values):
            plotted.append({'precision': precision, 'implementation': implementation,
                            'sustained_input_fps': report['sustained_input_fps'],
                            'target_latency_p95_ms': report['target_latency_p95_ms'],
                            'peak_allocated_vram_mib': report['peak_allocated_vram_mib'],
                            'frame_metrics_shared_mask': report['frame_metrics_shared_mask'],
                            'paired_gate_status': parity['status']})
    condition = proofs[('bf16', 'full')]['condition']
    fig.suptitle(f'{condition["backbone"]} / {condition["mode"]} / {condition["device"]} / '
                 f'{condition["adaptation"]} / seed {condition["seed"]}', fontsize=14, y=.99)
    fig.text(.02, .02,
             f'Actual {test_videos}-video file replay per run; warm file cache, 30 FPS arrivals, FIFO, no drops. '
             'The p95 panels use logarithmic axes.\n'
             'Separate precision pairs; one seed, no repeat-run CI, maximum-throughput or camera claim. '
             'Hatched red bars retain the failed score gate.', fontsize=9)
    fig.tight_layout(rect=(0, .145, 1, .965))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = [args.out.with_suffix('.png'), args.out.with_suffix('.svg')]
    for path in files:
        fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('Audited runtime sources changed during figure export')
    receipt = {'status': 'exported_actual_audited_runtime_pairs', 'condition': condition,
               'test_videos_per_run': test_videos, 'plotted_measurements': plotted,
               'parity': {precision: {'status': parity['status'],
                                     'failed_videos': [row['sequence'] for row in parity['sequences'] if not row['passed']],
                                     'alarm_mismatches': sum(row['alarm_mismatches'] for row in parity['sequences'])}
                          for precision, parity in parities.items()},
               'source_sha256': sources, 'files_sha256': {str(path): digest(path) for path in files},
               'scope': 'Two separate precision pairs for one measured condition; failed fixed gates remain failed; '
                        'measured target p95, no repeat-run CI, encoder/search replay, maximum throughput or camera claim.'}
    args.out.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'parity': receipt['parity']}))


if __name__ == '__main__':
    main()
