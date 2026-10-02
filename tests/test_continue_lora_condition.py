"""A condition continuation must retain the full training protocol and ownership."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continue_lora_condition import check_previous_owner, training_action

CONDITION = ('vjepa21-l', 'offline', 'R01', 2)


def metadata(tmp_path, **changes):
    value = dict(zip(('backbone', 'mode', 'device', 'seed'), CONDITION))
    value.update(epochs=20, teacher_weight=1., accumulation=8, max_steps=None,
                 status='running', completed_epochs=1)
    value.update(changes)
    (tmp_path / 'lora_training.json').write_text(json.dumps(value))


def test_fresh_empty_training_and_unidentified_partial_outputs(tmp_path):
    assert training_action(tmp_path, CONDITION, False) == 'fresh'
    (tmp_path / 'unknown.pt').write_text('fixture')
    with pytest.raises(ValueError, match='Unidentified'):
        training_action(tmp_path, CONDITION, True)


def test_running_training_needs_explicit_resume_and_checkpoint(tmp_path):
    metadata(tmp_path)
    for resume in (False, True):
        with pytest.raises(ValueError, match='Incomplete training'):
            training_action(tmp_path, CONDITION, resume)
    (tmp_path / 'resume.pt').write_text('fixture')
    assert training_action(tmp_path, CONDITION, True) == 'resume'
    with pytest.raises(ValueError, match='Incomplete training'):
        training_action(tmp_path, CONDITION, False)


@pytest.mark.parametrize('changes', [{'epochs': 1}, {'teacher_weight': 0.}, {'max_steps': 10}, {'seed': 1}])
def test_shortened_pilot_other_treatment_or_other_seed_cannot_resume(tmp_path, changes):
    metadata(tmp_path, **changes)
    (tmp_path / 'resume.pt').write_text('fixture')
    with pytest.raises(ValueError, match='identity or protocol'):
        training_action(tmp_path, CONDITION, True)


def test_only_full20_completion_can_skip_training(tmp_path):
    metadata(tmp_path, status='complete_training', completed_epochs=19)
    with pytest.raises(ValueError, match='Incomplete training'):
        training_action(tmp_path, CONDITION, True)
    metadata(tmp_path, status='complete_training', completed_epochs=20)
    assert training_action(tmp_path, CONDITION, False) == 'selected_complete'


def test_live_owner_wrong_condition_and_implicit_restart_are_refused(tmp_path, monkeypatch):
    path = tmp_path / 'ledger.json'
    path.write_text(json.dumps({'condition': list(CONDITION), 'owner': {'pid': 10}}))
    monkeypatch.setattr('continue_lora_condition.is_live', lambda owner: True)
    with pytest.raises(ValueError, match='still live'):
        check_previous_owner(path, CONDITION, True)
    monkeypatch.setattr('continue_lora_condition.is_live', lambda owner: False)
    with pytest.raises(ValueError, match='explicit resume'):
        check_previous_owner(path, CONDITION, False)
    check_previous_owner(path, CONDITION, True)
    with pytest.raises(ValueError, match='another condition'):
        check_previous_owner(path, CONDITION[:-1] + (1,), True)
