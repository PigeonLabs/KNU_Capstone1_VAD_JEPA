"""Render the verified closure inventory without implying full-plan completion."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/setup/experiment_closure.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/final/experiment_closure'))
    args = parser.parse_args()
    info = json.loads(args.source.read_text())
    assert info['status'] == 'closed_at_user_requested_current_stage'
    keys = ['frozen_backbones', 'primary_lora', 'native_controlled', 'teacher_weight_zero', 'runtime_replay']
    labels = ['Frozen backbones (conditions)', 'Primary LoRA (conditions)',
              'IPAD controlled (conditions)', 'Teacher weight 0 (conditions)',
              'FIFO runtime (executions)']
    rows = [info['inventory'][key] for key in keys]
    for row in rows:
        assert 0 <= row['completed'] <= row['planned']
    completed = [100 * row['completed'] / row['planned'] for row in rows]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    positions = list(range(len(rows)))
    ax.barh(positions, completed, color='#2875ac', label='Verified completed')
    ax.barh(positions, [100 - value for value in completed], left=completed,
            color='#e2e6eb', hatch='//', edgecolor='#adb5bd', label='Not completed at closure')
    for i, row in enumerate(rows):
        ax.text(102, i, f"{row['completed']}/{row['planned']}", va='center', fontsize=10)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 117)
    ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Completion within each experiment family')
    ax.set_title('Experiment closed at the current stage', loc='left', fontweight='bold', pad=17)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
    fig.text(.03, .02, 'Primary LoRA: 25/48; online R01 seed 1 saved at epoch 11/20 is excluded.\n'
             'Counts have different units; no overall completion ratio or full real-time comparison is claimed.',
             fontsize=9, color='#46505b')
    fig.tight_layout(rect=(0, .15, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        fig.savefig(args.out.with_suffix(suffix), dpi=170, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
