"""Actual filesystem crash windows and safeguards for archived derived-cache retirement."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ipad_jepa.cache_retention import inspect_array
from ipad_jepa.cache_retirement import atomic_record, process_identity, retire_payloads


def fixture(tmp_path):
    root = tmp_path / 'artifacts/features_lora/dinov3-l/offline/R01/seed0'
    directory = root / 'R01/training/01'
    directory.mkdir(parents=True)
    plan = []
    for name, shape in [('patch.npy', (2, 576, 1024)), ('global.npy', (2, 1024))]:
        path = directory / name
        np.save(path, np.ones(shape, dtype=np.float16))
        plan.append((path, inspect_array(path, shape)))
    (directory / 'meta.json').write_text('{"preserved":true}\n')
    np.save(directory / 'targets.npy', np.array([8, 12], dtype=np.int64))
    evidence = {'inventory_sha256': 'd' * 64, 'publication_revision': 'a' * 40,
                'condition': ['dinov3-l', 'offline', 'R01', 0], 'teacher_weight': 1.,
                'publication_CI_conclusion': 'success', 'publication_CI_head_sha': 'a' * 40}
    journal = tmp_path / 'results/cache_retention/T1/dinov3-l/offline/R01/seed0/retirement.json'
    return root, plan, evidence, journal


def crash_before_unlink(path, state):
    raise RuntimeError('simulated interruption after durable intent')


def test_exact_deletion_retains_metadata_targets_and_durable_order(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    result = retire_payloads(tmp_path, root, plan, journal, evidence)
    assert result['status'] == 'complete_verified_derived_payload_retirement'
    assert result == json.loads(journal.read_text())
    assert [r['relative'] for r in result['deleted_files']] == ['R01/training/01/patch.npy', 'R01/training/01/global.npy']
    assert all(not path.exists() for path, _ in plan)
    assert (root / 'R01/training/01/meta.json').read_text() == '{"preserved":true}\n'
    np.testing.assert_array_equal(np.load(root / 'R01/training/01/targets.npy'), [8, 12])
    assert result['deleted_payload_bytes'] == sum(row['bytes'] for _, row in plan)


def test_entire_plan_checked_before_any_unlink_and_no_unexplained_missing_file(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    plan[1][0].unlink()
    with pytest.raises(ValueError, match='no durable unlink intent'):
        retire_payloads(tmp_path, root, plan, journal, evidence)
    assert plan[0][0].exists() and not journal.exists()


def test_full_payload_hash_rejects_mutation_even_with_restored_size_inode_mtime(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    path, record = plan[1]
    stat = path.stat()
    with path.open('r+b') as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b'\0')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='bytes changed'):
        retire_payloads(tmp_path, root, plan, journal, evidence)
    assert plan[0][0].exists() and not journal.exists()


def test_resume_from_intent_before_unlink(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    with pytest.raises(RuntimeError):
        retire_payloads(tmp_path, root, plan, journal, evidence, before_unlink=crash_before_unlink)
    state = json.loads(journal.read_text())
    assert state['pending_file']['relative'].endswith('patch.npy') and not state['deleted_files']
    assert all(path.exists() for path, _ in plan)
    with pytest.raises(ValueError, match='explicit resume'):
        retire_payloads(tmp_path, root, plan, journal, evidence)
    result = retire_payloads(tmp_path, root, plan, journal, evidence, resume=True)
    assert result['status'].startswith('complete') and result['recovered_absent_pending_files'] == []


def test_resume_from_unlink_before_confirmation(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    def crash_after_unlink(path, state):
        path.unlink()
        raise RuntimeError('simulated interruption before confirmation')
    with pytest.raises(RuntimeError):
        retire_payloads(tmp_path, root, plan, journal, evidence, before_unlink=crash_after_unlink)
    result = retire_payloads(tmp_path, root, plan, journal, evidence, resume=True)
    assert result['status'].startswith('complete')
    assert result['recovered_absent_pending_files'] == ['R01/training/01/patch.npy']


def test_resume_refuses_other_missing_file_without_its_own_intent(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    with pytest.raises(RuntimeError):
        retire_payloads(tmp_path, root, plan, journal, evidence, before_unlink=crash_before_unlink)
    plan[1][0].unlink()
    with pytest.raises(ValueError, match='no durable unlink intent'):
        retire_payloads(tmp_path, root, plan, journal, evidence, resume=True)
    assert plan[0][0].exists()


def test_resume_refuses_other_still_live_owner(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    with pytest.raises(RuntimeError):
        retire_payloads(tmp_path, root, plan, journal, evidence, before_unlink=crash_before_unlink)
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
    try:
        state = json.loads(journal.read_text())
        state['owner'] = process_identity(worker.pid)
        assert state['owner'] is not None
        atomic_record(journal, state)
        with pytest.raises(ValueError, match='still live'):
            retire_payloads(tmp_path, root, plan, journal, evidence, resume=True)
    finally:
        worker.terminate()
        worker.wait()


def test_complete_idempotent_resume_and_recreated_payload_refused(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    result = retire_payloads(tmp_path, root, plan, journal, evidence)
    assert retire_payloads(tmp_path, root, plan, journal, evidence, resume=True) == result
    plan[0][0].write_bytes(b'recreated')
    with pytest.raises(ValueError, match='recreated'):
        retire_payloads(tmp_path, root, plan, journal, evidence, resume=True)


@pytest.mark.parametrize('field,value', [('publication_revision', 'HEAD'),
                                       ('publication_CI_conclusion', 'failure'),
                                       ('publication_CI_head_sha', 'b' * 40),
                                       ('teacher_weight', 0.)])
def test_bad_archive_ci_or_teacher_cannot_delete(tmp_path, field, value):
    root, plan, evidence, journal = fixture(tmp_path)
    evidence[field] = value
    with pytest.raises(ValueError):
        retire_payloads(tmp_path, root, plan, journal, evidence)
    assert all(path.exists() for path, _ in plan)


def test_changed_plan_evidence_or_deletion_prefix_cannot_resume(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    with pytest.raises(RuntimeError):
        retire_payloads(tmp_path, root, plan, journal, evidence, before_unlink=crash_before_unlink)
    with pytest.raises(ValueError, match='evidence or exact plan changed'):
        retire_payloads(tmp_path, root, plan, journal, {**evidence, 'inventory_sha256': 'e' * 64}, resume=True)
    state = json.loads(journal.read_text())
    state['deleted_files'] = [{'relative': 'invented'}]
    atomic_record(journal, state)
    with pytest.raises(ValueError, match='deletion prefix'):
        retire_payloads(tmp_path, root, plan, journal, evidence, resume=True)


@pytest.mark.parametrize('name', ['targets.npy', 'meta.json', 'selected_adapter.pt'])
def test_nonpayload_members_cannot_be_removed(tmp_path, name):
    root, plan, evidence, journal = fixture(tmp_path)
    path = root / 'R01/training/01' / name
    if not path.exists():
        path.write_bytes(b'preserved')
    with pytest.raises(ValueError, match='Invalid derived cache member'):
        retire_payloads(tmp_path, root, [(path, plan[0][1])], journal, evidence)
    assert path.exists()


def test_symlink_swap_after_durable_intent_cannot_delete_external_payload(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    external = tmp_path / 'external.npy'
    external.write_bytes(b'outside')
    def swap(path, state):
        path.unlink()
        path.symlink_to(external)
    with pytest.raises(ValueError, match='Symlink'):
        retire_payloads(tmp_path, root, plan, journal, evidence, before_unlink=swap)
    assert external.read_bytes() == b'outside' and plan[1][0].exists()


def test_duplicate_or_noncanonical_journal_cannot_delete(tmp_path):
    root, plan, evidence, journal = fixture(tmp_path)
    with pytest.raises(ValueError, match='Duplicate'):
        retire_payloads(tmp_path, root, plan + plan[:1], journal, evidence)
    with pytest.raises(ValueError, match='Canonical'):
        retire_payloads(tmp_path, root, plan, tmp_path / 'elsewhere.json', evidence)
    assert all(path.exists() for path, _ in plan)
