"""Plot an actual completed derived-cache retirement and verified retained files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_retained_lora import audit_retained
from ipad_jepa.cache_retention import digest
from ipad_jepa.cache_retirement import atomic_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.receipt.read_text())
    folder = args.receipt.parent
    release_path = folder / 'retirement.json'
    release = json.loads(release_path.read_text())
    execution_path = folder / 'retirement_execution_check.json'
    execution = json.loads(execution_path.read_text())
    post_path = folder / 'post_retirement_audit.json'
    _, post = audit_retained(Path.cwd(), args.receipt, release['publication_revision'],
                            args.manifest, args.data_root, release_path)
    if (execution['status'] != 'complete_retirement_and_retained_replay'
            or execution['retirement_sha256'] != digest(release_path)
            or execution['post_retirement_audit_sha256'] != digest(post_path)
            or json.loads(post_path.read_text()) != post):
        raise ValueError('Actual complete removal and matching independent post replay required')
    current_root = Path(inventory['cache_root'])
    current_arrays = 0
    current_bytes = 0
    preserved = {'Metadata JSON': 0, 'Target NPY': 0}
    for record in inventory['records']:
        base = current_root / record['relative_directory']
        for name, field, label in [('meta.json', 'metadata_sha256', 'Metadata JSON'),
                                    ('targets.npy', 'targets_sha256', 'Target NPY')]:
            if digest(base / name) != record[field]:
                raise ValueError('Retained metadata or target changed')
            preserved[label] += 1
        for name in record['arrays']:
            path = base / name
            if path.exists() or path.is_symlink():
                current_arrays += 1
                current_bytes += path.stat().st_size
    if current_arrays or current_bytes:
        raise ValueError('Completed retirement cannot be plotted with remaining payloads')
    before_count = inventory['array_files_checked']
    before_bytes = inventory['derived_bytes_checked']
    if (len(release['deleted_files']) != before_count
            or release['deleted_payload_bytes'] != before_bytes):
        raise ValueError('Historical and actual unlink totals differ')
    paths = [args.receipt, release_path, execution_path, post_path, args.manifest,
             folder / 'retirement_plan.json', folder / 'dependency_regeneration_check.json',
             Path('scripts/plot_cache_retirement.py'), Path('scripts/audit_retained_lora.py')]
    sources = {str(path): digest(path) for path in paths}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
    fig.suptitle('DINOv3-L LoRA / offline / R01 / seed 0\nCompleted derived-cache retirement', fontsize=13)
    gib = before_bytes / 1024 ** 3
    bars = axes[0].bar(['Before (archived)', 'After (current)'], [gib, 0], color=['#8398ae', '#257c71'], width=.58)
    axes[0].set(ylabel='Logical patch/global payload (GiB)', ylim=(0, gib * 1.28))
    axes[0].bar_label(bars, labels=[f'{gib:.2f} GiB\n{before_count} arrays', '0 GiB\n0 arrays'], padding=6)
    for index, (label, count) in enumerate(preserved.items()):
        axes[1].bar(index - .18, count, width=.34, color='#8398ae', label='Before (archived)' if index == 0 else None)
        axes[1].bar(index + .18, count, width=.34, color='#257c71', label='After (hash verified)' if index == 0 else None)
        axes[1].text(index - .18, count + 1, str(count), ha='center')
        axes[1].text(index + .18, count + 1, str(count), ha='center')
    axes[1].set(xticks=[0, 1], xticklabels=list(preserved), ylabel='Preserved files', ylim=(0, max(preserved.values()) * 1.32))
    axes[1].legend(loc='upper right', frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
    fig.supxlabel('Scope: one completed condition; payload bytes include NPY headers.\nRetained adapter/head/bank and evaluation traces replayed; no encoder regeneration or runtime claim.', fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(args.out.with_suffix('.' + extension), dpi=180)
    plt.close(fig)
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('Plot evidence changed during rendering')
    proof = {'status': 'plotted_actual_complete_retirement_and_current_retained_replay',
             'condition': inventory['condition'], 'historical_payload_files': before_count,
             'historical_payload_bytes': before_bytes, 'current_payload_files': current_arrays,
             'current_payload_bytes': current_bytes, 'current_hash_verified_retained_file_counts': preserved,
             'source_sha256': sources,
             'figure_sha256': {str(args.out.with_suffix('.' + ext)): digest(args.out.with_suffix('.' + ext))
                               for ext in ('png', 'svg')},
             'scope': post['limits']}
    atomic_record(args.out.with_suffix('.json'), proof)
    print(json.dumps({k: v for k, v in proof.items() if k not in ['source_sha256', 'figure_sha256']}, indent=2))


if __name__ == '__main__':
    main()
