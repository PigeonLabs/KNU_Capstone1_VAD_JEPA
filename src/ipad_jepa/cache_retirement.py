"""Durable, resumable removal of an exact previously archived derived-payload plan."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from ipad_jepa.cache_retention import ARRAY_NAMES, condition_cache, member, reject_symlinks


def atomic_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', prefix='.retirement-', suffix='.tmp',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks': fields[19]}
    except (OSError, IndexError):
        return None


def file_identity(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def verify_payload(path, record):
    """Read the exact file through O_NOFOLLOW, checking full bytes and its archived identity."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if file_identity(before) != record['file_identity']:
            raise ValueError('Derived payload identity changed after archive')
        header = stream.read(record['npy_header_bytes'])
        if hashlib.sha256(header).hexdigest() != record['npy_header_sha256']:
            raise ValueError('Archived NPY header changed')
        value = hashlib.sha256(header)
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
        if value.hexdigest() != record['sha256']:
            raise ValueError('Derived payload bytes changed after archive')
        after = os.fstat(stream.fileno())
        if file_identity(before) != file_identity(after):
            raise ValueError('Derived payload changed during full hash check')
    if file_identity(Path(path).stat(follow_symlinks=False)) != record['file_identity']:
        raise ValueError('Derived payload path changed after descriptor check')


def planned_files(repo, root, plan, evidence):
    if not re.fullmatch('[0-9a-f]{40}', evidence.get('publication_revision', '')):
        raise ValueError('Exact archived publication revision required')
    if (evidence.get('publication_CI_conclusion') != 'success'
            or evidence.get('publication_CI_head_sha') != evidence['publication_revision']):
        raise ValueError('Exact successful archive CI required')
    root = condition_cache(repo, root, tuple(evidence['condition']), evidence['teacher_weight'])
    if not isinstance(plan, list) or not plan:
        raise ValueError('Nonempty exact derived payload plan required')
    records = []
    for path, record in plan:
        relative = str(Path(path).absolute().relative_to(root))
        if member(root, relative, ARRAY_NAMES) != Path(path).absolute():
            raise ValueError('Derived payload path differs from canonical plan')
        if (not re.fullmatch('[0-9a-f]{64}', record.get('sha256', ''))
                or not re.fullmatch('[0-9a-f]{64}', record.get('npy_header_sha256', ''))
                or record.get('finite_payload_checked') is not True
                or not isinstance(record.get('npy_header_bytes'), int)
                or record['npy_header_bytes'] < 10
                or not isinstance(record.get('bytes'), int)
                or record['bytes'] < record['npy_header_bytes']
                or len(record.get('file_identity', [])) != 4
                or not all(isinstance(item, int) and item >= 0 for item in record['file_identity'])
                or record['file_identity'][2] != record['bytes']):
            raise ValueError('Complete original payload evidence required')
        records.append({'relative': relative, 'sha256': record['sha256'], 'bytes': record['bytes']})
    if len({row['relative'] for row in records}) != len(records):
        raise ValueError('Duplicate derived payload plan member')
    return root, records


def retire_payloads(repo, root, plan, release_path, evidence, *, resume=False, before_unlink=None):
    """Persist unlink intent first; unexpected missing/recreated files never become success.

    Caller must hold the per-condition kernel lock and independently establish the
    published inventory, successful CI and current scientific replay. The callback is
    a fault-injection hook for crash-window tests; the production CLI never sets it.
    """
    root, planned = planned_files(repo, root, plan, evidence)
    release_path = Path(release_path)
    expected_path = (Path(repo).absolute() / 'results/cache_retention'
                     / ('T0' if evidence['teacher_weight'] == 0 else 'T1')
                     / Path(*map(str, evidence['condition'][:3])) / f"seed{evidence['condition'][3]}"
                     / 'retirement.json')
    if release_path.absolute() != expected_path:
        raise ValueError('Canonical per-condition retirement journal required')
    reject_symlinks(release_path, repo)
    owner = process_identity(os.getpid())
    if owner is None:
        raise ValueError('Actual process identity required')
    if release_path.exists():
        if not resume:
            raise ValueError('Existing retirement journal requires explicit resume')
        state = json.loads(release_path.read_text())
        if (any(state.get(key) != value for key, value in evidence.items())
                or state.get('planned_files') != planned):
            raise ValueError('Existing journal evidence or exact plan changed')
        previous = state.get('owner')
        if previous != owner and previous and process_identity(previous['pid']) == previous:
            raise ValueError('Previous retirement owner is still live')
        if state.get('status') not in {'in_progress_derived_payload_retirement',
                                       'complete_verified_derived_payload_retirement'}:
            raise ValueError('Unexpected retirement journal state')
        completed = state['deleted_files']
        expected_prefix = [{**row, 'observed_absent_after_unlink': True}
                           for row in planned[:len(completed)]]
        if completed != expected_prefix:
            raise ValueError('Completed deletion prefix differs from exact plan')
        pending = state.get('pending_file')
        if pending is not None and (len(completed) == len(planned) or pending != planned[len(completed)]):
            raise ValueError('Durable pending unlink intent differs from next file')
    else:
        state = {**evidence, 'status': 'in_progress_derived_payload_retirement', 'owner': owner,
                 'planned_files': planned, 'deleted_files': [], 'pending_file': None,
                 'recovered_absent_pending_files': [],
                 'scope': 'Exact archived derived patch/global payload removal only; metadata/targets, '
                          'raw input, teacher caches, selected models/head/banks and score traces retained. '
                          'Each original payload rehashed before any unlink and immediately before its unlink. '
                          'Durable unlink intent precedes removal; post-removal scientific replay is separate.'}
    completed_count = len(state['deleted_files'])
    pending = state.get('pending_file')
    # Validate the entire remaining plan before deleting the first remaining file.
    for index, ((path, record), planned_row) in enumerate(zip(plan, planned)):
        member(root, planned_row['relative'], ARRAY_NAMES)
        present = path.exists() or path.is_symlink()
        if index < completed_count:
            if present:
                raise ValueError('Previously retired payload was recreated')
        elif not present:
            if not (index == completed_count and pending == planned_row):
                raise ValueError('Missing payload has no durable unlink intent')
        else:
            verify_payload(path, record)
    if state['status'] == 'complete_verified_derived_payload_retirement':
        if completed_count != len(planned) or pending is not None:
            raise ValueError('Incomplete journal claims complete retirement')
        return state
    state['owner'] = owner
    atomic_record(release_path, state)
    for index in range(completed_count, len(plan)):
        path, record = plan[index]
        row = planned[index]
        member(root, row['relative'], ARRAY_NAMES)
        if not path.exists():
            if state['pending_file'] != row:
                raise ValueError('Missing payload lacks the persisted unlink intent')
            state['recovered_absent_pending_files'].append(row['relative'])
        else:
            verify_payload(path, record)
            state['pending_file'] = row
            atomic_record(release_path, state)
            if before_unlink is not None:
                before_unlink(path, state)
            # Reject a replaced/symlinked inode even after durable intent was saved.
            member(root, row['relative'], ARRAY_NAMES)
            if file_identity(path.stat(follow_symlinks=False)) != record['file_identity']:
                raise ValueError('Payload changed immediately before unlink')
            path.unlink()
        if path.exists() or path.is_symlink():
            raise ValueError('Payload remains after unlink')
        state['deleted_files'].append({**row, 'observed_absent_after_unlink': True})
        state['pending_file'] = None
        atomic_record(release_path, state)
    if (len(state['deleted_files']) != len(planned) or state['pending_file'] is not None
            or any(path.exists() or path.is_symlink() for path, _ in plan)):
        raise ValueError('Actual completed unlink inventory differs from the entire plan')
    state['status'] = 'complete_verified_derived_payload_retirement'
    state['deleted_payload_bytes'] = sum(row['bytes'] for row in planned)
    atomic_record(release_path, state)
    return state
