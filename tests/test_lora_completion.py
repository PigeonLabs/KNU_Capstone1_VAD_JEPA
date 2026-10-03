"""Completion visualizations must not turn partial or contradictory audits into full results."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lora_reporting import MODELS, MODES, DEVICES, SEEDS
from plot_lora_completion import completion_grid


def inventory(completed):
    required = {(m, mode, d, s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS}
    groups = sum(all((m, mode, d, s) in completed for s in SEEDS)
                 for m in MODELS for mode in MODES for d in DEVICES)
    return {'status': 'passed_mixed_current_retained_evidence', 'kind': 'lora',
            'condition_checks': [{'condition': list(c), 'treatments': [{'teacher_weight': 1.}]} for c in sorted(completed)],
            'pending_conditions_or_pairs': [list(c) for c in sorted(required - completed)],
            'completed_conditions_or_pairs': len(completed), 'complete_three_seed_groups': groups,
            'matrix_complete': len(completed) == 48, 'normal_thresholds_replayed': len(completed) * 8}


def partial():
    return inventory({('dinov3-l', 'offline', 'R03', 0), ('dinov3-l', 'offline', 'R03', 1)})


def test_two_completed_seeds_stay_individual_with_third_pending_and_no_group():
    grid, rows, columns, groups = completion_grid(partial())
    y = rows.index(('dinov3-l', 'offline'))
    assert grid.sum() == 2 and grid.shape == (4, 12) and groups == []
    np.testing.assert_array_equal(grid[y, [columns.index(('R03', s)) for s in SEEDS]], [1, 1, 0])


@pytest.mark.parametrize('key,value', [('completed_conditions_or_pairs', 48),
                                      ('complete_three_seed_groups', 1),
                                      ('matrix_complete', True),
                                      ('normal_thresholds_replayed', 384)])
def test_claimed_full_count_or_group_cannot_override_incomplete_seed_evidence(key, value):
    proof = partial(); proof[key] = value
    with pytest.raises(ValueError, match='contradict'): completion_grid(proof)


def test_duplicate_witness_cannot_count_as_a_third_seed():
    proof = partial(); proof['condition_checks'].append(deepcopy(proof['condition_checks'][0]))
    proof['completed_conditions_or_pairs'] = 3
    with pytest.raises(ValueError, match='duplicate'): completion_grid(proof)


def test_missing_pending_condition_cannot_hide_remaining_scope():
    proof = partial(); proof['pending_conditions_or_pairs'].pop()
    with pytest.raises(ValueError, match='contradict'): completion_grid(proof)


def test_pending_and_completed_may_not_overlap():
    proof = partial(); proof['pending_conditions_or_pairs'][0] = proof['condition_checks'][0]['condition']
    with pytest.raises(ValueError, match='contradict'): completion_grid(proof)


def test_json_boolean_is_not_a_seed_number():
    proof = partial(); proof['condition_checks'][1]['condition'][3] = True
    with pytest.raises(ValueError, match='integer seed'): completion_grid(proof)


def test_teacher_zero_cannot_substitute_primary_teacher_one_completion():
    proof = partial(); proof['condition_checks'][0]['treatments'][0]['teacher_weight'] = 0.
    with pytest.raises(ValueError, match='teacher-one'): completion_grid(proof)


def test_only_exact_all48_form_complete16_groups():
    required = {(m, mode, d, s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS}
    grid, _, _, groups = completion_grid(inventory(required))
    assert np.all(grid == 1) and len(groups) == 16
