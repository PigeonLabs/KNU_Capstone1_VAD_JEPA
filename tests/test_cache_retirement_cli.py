"""Deletion eligibility requires the intended exact CI and an actually held runtime."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import retire_lora_cache as cli


@pytest.mark.parametrize('field,value', [('headSha', 'b' * 40), ('status', 'in_progress'),
                                       ('conclusion', 'failure'), ('workflowName', 'Unrelated green workflow')])
def test_successful_ci_refuses_wrong_head_pending_failed_or_unrelated_workflow(monkeypatch, field, value):
    result = {'headSha': 'a' * 40, 'status': 'completed', 'conclusion': 'success',
              'workflowName': 'CPU invariants and upload guard', 'url': 'https://example.invalid/fixture'}
    result[field] = value
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda args: json.dumps(result).encode())
    with pytest.raises(ValueError, match='Actual CI'):
        cli.successful_ci(1, 'a' * 40)


def test_successful_ci_accepts_only_matching_completed_validation_workflow(monkeypatch):
    result = {'headSha': 'a' * 40, 'status': 'completed', 'conclusion': 'success',
              'workflowName': 'CPU invariants and upload guard', 'url': 'https://example.invalid/fixture'}
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda args: json.dumps(result).encode())
    assert cli.successful_ci(1, 'a' * 40) == result


@pytest.mark.parametrize('state,live', [({'status': 'running', 'owner': {}, 'child': None}, True),
                                      ({'status': 'held_between_measurements', 'owner': {}, 'child': None}, False),
                                      ({'status': 'held_between_measurements', 'owner': {}, 'child': {'pid': 2}}, True)])
def test_stale_or_measuring_controller_cannot_enable_full_hash_io(tmp_path, monkeypatch, state, live):
    monkeypatch.chdir(tmp_path)
    root = Path('artifacts/tmp')
    root.mkdir(parents=True)
    (root / 'runtime_matrix.json').write_text(json.dumps(state))
    (root / 'runtime_matrix.hold').write_text('fixture')
    monkeypatch.setattr(cli, 'is_live', lambda identity: live)
    with pytest.raises(ValueError, match='must be held'):
        cli.check_measurement_gate()


def test_gpu_work_refused_even_with_live_held_controller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path('artifacts/tmp')
    root.mkdir(parents=True)
    (root / 'runtime_matrix.json').write_text(json.dumps({'status': 'held_between_measurements', 'owner': {}, 'child': None}))
    (root / 'runtime_matrix.hold').write_text('fixture')
    monkeypatch.setattr(cli, 'is_live', lambda identity: True)
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda args: b'123\n')
    with pytest.raises(ValueError, match='No GPU work'):
        cli.check_measurement_gate()
