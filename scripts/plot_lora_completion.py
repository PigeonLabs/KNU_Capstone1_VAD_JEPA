"""Show completed primary LoRA conditions from source-bound mixed evidence audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from lora_reporting import MODELS, MODES, DEVICES, SEEDS
from plot_neighbour_ablation import digest, save, NAMES
from plot_retained_lora import verify_mixed_sources


def completion_grid(validation):
    """Reject invented completion, missing seeds, or inconsistent pending inventories."""
    if validation.get('status') != 'passed_mixed_current_retained_evidence' or validation.get('kind') != 'lora':
        raise ValueError('Passed primary LoRA mixed evidence audit required')
    required = {(m, mode, d, s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS}

    def keys(rows):
        values = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 4 or type(row[3]) is not int:
                raise ValueError('Exact model/mode/device/integer seed condition required')
            key = tuple(row)
            if key not in required or key in values:
                raise ValueError('Unexpected or duplicate primary condition')
            values.append(key)
        return set(values)

    completed = keys([row['condition'] for row in validation['condition_checks']])
    pending = keys(validation['pending_conditions_or_pairs'])
    groups = [(m, mode, d) for m in MODELS for mode in MODES for d in DEVICES
              if all((m, mode, d, s) in completed for s in SEEDS)]
    if (not completed or completed | pending != required or completed & pending
            or validation['completed_conditions_or_pairs'] != len(completed)
            or validation['complete_three_seed_groups'] != len(groups)
            or type(validation['matrix_complete']) is not bool
            or validation['matrix_complete'] != (len(completed) == 48)
            or validation['normal_thresholds_replayed'] != len(completed) * 8):
        raise ValueError('Completion/group/full/threshold/pending counts contradict actual conditions')
    for row in validation['condition_checks']:
        if len(row['treatments']) != 1 or row['treatments'][0]['teacher_weight'] != 1.0:
            raise ValueError('Exactly one primary teacher-one LoRA treatment required')
    rows = [(m, mode) for m in MODELS for mode in MODES]
    columns = [(d, s) for d in DEVICES for s in SEEDS]
    grid = np.array([[int((m, mode, d, s) in completed) for d, s in columns]
                     for m, mode in rows], dtype=np.uint8)
    return grid, rows, columns, groups


def render(validation, out, source):
    grid, rows, columns, groups = completion_grid(validation)
    fig, axis = plt.subplots(figsize=(13, 4.7), layout='constrained')
    axis.imshow(grid, cmap=ListedColormap(['#edf0f2', '#245b86']), vmin=0, vmax=1, aspect='auto')
    for y, x in np.ndindex(grid.shape):
        axis.text(x, y, 'Done' if grid[y, x] else 'Pending', ha='center', va='center',
                  color='white' if grid[y, x] else '#59656d', fontsize=9)
    axis.set(xticks=range(len(columns)), xticklabels=[f'{d}\nseed {s}' for d, s in columns],
             yticks=range(len(rows)), yticklabels=[f'{NAMES[m]} / {mode}' for m, mode in rows])
    for x in (2.5, 5.5, 8.5):
        axis.axvline(x, color='white', linewidth=4)
    axis.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    axis.grid(which='minor', color='white', linewidth=1.5)
    axis.tick_params(which='both', length=0)
    axis.spines[:].set_visible(False)
    axis.legend(handles=[Patch(facecolor='#245b86', label='Full condition independently audited'),
                         Patch(facecolor='#edf0f2', label='Full condition not yet verified')],
                loc='upper center', bbox_to_anchor=(.5, 1.23), ncols=2, frameon=False)
    fig.suptitle(f'Primary LoRA / {int(grid.sum())} of 48 completed conditions\n'
                 f'{len(groups)} of 16 exact three-seed groups complete', fontsize=15)
    note = ('Done: full20 normal training, selected features, rebuilt memory/calibration, all real tests and independent audit.\n'
            'Pending includes running, queued and unstarted work; training alone is not a completed condition. No schedule or runtime claim.\n'
            'Only exact seeds0/1/2 form a group; partial seeds never imply a group mean or Macro4. Source: ' + str(source))
    return save(fig, out, note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.kind = 'lora'
    source = args.root / 'validation.json'
    validation = json.loads(source.read_text())
    completion_grid(validation)
    verify_mixed_sources(args, validation)
    source_hash = digest(source)
    figures = render(validation, args.out, source)
    if digest(source) != source_hash:
        raise ValueError('Audited inventory changed while rendering')
    receipt = {'status': 'rendered_from_fresh_source_bound_primary_condition_audit',
               'completed_conditions': validation['completed_conditions_or_pairs'],
               'complete_three_seed_groups': validation['complete_three_seed_groups'],
               'matrix_complete': validation['matrix_complete'],
               'validation_sha256': source_hash, 'plot_code_sha256': digest(__file__), 'figures': figures,
               'scope': 'All48 planned primary conditions. Pending means full condition unverified, including ongoing work. No partial-seed means, Macro4, measured runtime or schedule. Actual PNG review required.'}
    output = args.out.with_name(args.out.name + '_sources.json')
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
