"""Prepare or execute journalled retirement only after exact inventory and implementation CI."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from audit_retained_lora import archived_receipt, audit_retained, replay_tensors, verify_release
from ipad_jepa.cache_retention import check_saved_inventory, digest
from ipad_jepa.cache_retirement import atomic_record, planned_files, retire_payloads, verify_payload
from plot_lora_evaluation import calibration_check
from run_runtime_matrix import is_live
from summarize_clip_ablation import audit_condition as audit_p3
from summarize_retained_lora_matrix import treatment_paths


REPOSITORY = 'PigeonLabs/KNU_Capstone1_VAD_JEPA'


def successful_ci(run_id, revision):
    result = json.loads(subprocess.check_output([
        'gh', 'run', 'view', str(run_id), '--repo', REPOSITORY,
        '--json', 'headSha,status,conclusion,workflowName,url']))
    if (result['headSha'] != revision or result['status'] != 'completed'
            or result['conclusion'] != 'success'
            or result['workflowName'] != 'CPU invariants and upload guard'):
        raise ValueError('Actual CI is not successful for the exact required revision')
    return result


def check_measurement_gate():
    ledger = Path('artifacts/tmp/runtime_matrix.json')
    if ledger.exists():
        state = json.loads(ledger.read_text())
        if (not is_live(state.get('owner')) or state['status'] != 'held_between_measurements'
                or state.get('child') is not None or not Path('artifacts/tmp/runtime_matrix.hold').is_file()):
            raise ValueError('Live runtime controller must be held between measurements')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
    if gpu:
        raise ValueError('No GPU work may overlap full cache verification or removal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--publication-revision', required=True)
    parser.add_argument('--publication-ci-id', type=int, required=True)
    parser.add_argument('--implementation-revision')
    parser.add_argument('--implementation-ci-id', type=int)
    parser.add_argument('--execute', action='store_true', help='Default only prepares; this performs the exact derived unlink plan')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    args = parser.parse_args()
    if args.resume and not args.execute:
        raise ValueError('Explicit execute required to resume an existing deletion journal')
    repo = Path.cwd()
    receipt, witness = archived_receipt(repo, args.receipt, args.publication_revision)
    archive_ci = successful_ci(args.publication_ci_id, args.publication_revision)
    condition, weight = tuple(receipt['condition']), receipt['teacher_weight']
    paths = treatment_paths(condition, weight)
    if args.receipt.resolve() != paths['inventory'].resolve():
        raise ValueError('Canonical archived treatment inventory required')
    for name, path in [('public_root', paths['public']), ('training_root', paths['training']), ('local_root', paths['local'])]:
        if receipt[name] != str(path):
            raise ValueError('Archived treatment namespace differs')
    pins = dict(receipt['prepared_source_sha256'])
    pins.update({str(args.receipt): digest(args.receipt), str(args.receipt.parent / receipt['original_witness']): receipt['original_witness_sha256'],
                 'scripts/retire_lora_cache.py': digest(__file__),
                 'src/ipad_jepa/cache_retirement.py': digest('src/ipad_jepa/cache_retirement.py'),
                 'scripts/audit_retained_lora.py': digest('scripts/audit_retained_lora.py')})
    if receipt['manifest_sha256'] != digest(args.manifest) or any(digest(p) != sha for p, sha in pins.items()):
        raise ValueError('Archived inventory/preparation sources changed')
    for name, expected in witness['public_sources_sha256'].items():
        path = paths['public'] / name
        if digest(path) != expected:
            raise ValueError('Original audited public result changed')
        pins[str(path)] = expected
    implementation_ci = None
    if args.execute:
        if not args.implementation_revision or not args.implementation_ci_id:
            raise ValueError('Published successful implementation CI required before execute')
        implementation_ci = successful_ci(args.implementation_ci_id, args.implementation_revision)
        for name in ['scripts/retire_lora_cache.py', 'src/ipad_jepa/cache_retirement.py']:
            published = subprocess.check_output(['git', 'show', args.implementation_revision + ':' + name])
            if published != Path(name).read_bytes():
                raise ValueError('Executed retirement source differs from tested published implementation')
    check_measurement_gate()
    Path('artifacts/tmp').mkdir(parents=True, exist_ok=True)
    with (Path('artifacts/tmp') / ('cache-retirement-' + digest(args.receipt) + '.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads(args.manifest.read_text())['sequences']
        info = json.loads((paths['public'] / 'metrics.json').read_text())
        normal, _ = calibration_check(paths['public'], info, manifest, args.data_root)
        sources = {name: digest(Path('src/ipad_jepa') / (name + '.py')) for name in receipt['source_sha256']}
        root, plan = check_saved_inventory(repo, receipt['cache_root'], receipt, witness, manifest, normal, sources)
        relative = Path(condition[0]) / condition[1] / condition[2] / f'seed{condition[3]}'
        frozen, frozen_local = Path('results/stage02') / relative, Path('artifacts/runs') / relative
        replay_tensors(paths['training'], paths['local'], frozen, frozen_local, paths['public'], witness, normal, condition, weight)
        audit_p3(paths['public'], manifest, args.data_root, *condition, 16)
        audit_p3(frozen, manifest, args.data_root, *condition, 16)
        evidence = {'inventory_sha256': digest(args.receipt), 'publication_revision': args.publication_revision,
                    'condition': list(condition), 'teacher_weight': weight,
                    'publication_CI_conclusion': archive_ci['conclusion'], 'publication_CI_head_sha': archive_ci['headSha'],
                    'publication_CI_id': args.publication_ci_id, 'publication_CI_url': archive_ci['url']}
        _, planned = planned_files(repo, root, plan, evidence)
        plan_path = args.receipt.with_name('retirement_plan.json')
        prepared = {'status': 'prepared_verified_retirement_plan', **evidence, 'planned_files': planned,
                    'payload_bytes': sum(row['bytes'] for row in planned), 'source_sha256': pins,
                    'cache_removal_performed': False,
                    'scope': 'Exact successful archive CI, current metadata/targets/selected tensors/P3 and full payload hashes verified before any removal.'}
        if not args.execute:
            if args.receipt.with_name('retirement.json').exists():
                raise ValueError('A removal journal already exists; use explicit execute/resume and its retained audit')
            for path, record in plan:
                verify_payload(path, record)
            if any(digest(p) != sha for p, sha in pins.items()):
                raise ValueError('Preparation inputs changed')
            if plan_path.exists() and json.loads(plan_path.read_text()) != prepared:
                raise ValueError('Existing prepared plan changed; inspect before replacing')
            atomic_record(plan_path, prepared)
            print(json.dumps({'status': prepared['status'], 'array_files': len(plan), 'bytes': prepared['payload_bytes'], 'cache_removal_performed': False}))
            return
        if not plan_path.exists() or json.loads(plan_path.read_text()) != prepared:
            raise ValueError('Exact previously prepared and published retirement plan required')
        if subprocess.check_output(['git', 'show', args.implementation_revision + ':' + str(plan_path)]) != plan_path.read_bytes():
            raise ValueError('Prepared plan is not present in the tested implementation publication')
        check_measurement_gate()
        if any(digest(p) != sha for p, sha in pins.items()):
            raise ValueError('Pre-removal inputs changed')
        evidence.update(implementation_CI_head_sha=implementation_ci['headSha'],
                        implementation_CI_conclusion=implementation_ci['conclusion'],
                        implementation_CI_id=args.implementation_ci_id, implementation_CI_url=implementation_ci['url'],
                        retirement_source_sha256={p: pins[p] for p in ['scripts/retire_lora_cache.py', 'src/ipad_jepa/cache_retirement.py']})
        release_path = args.receipt.with_name('retirement.json')
        release = retire_payloads(repo, root, plan, release_path, evidence, resume=args.resume)
        verify_release(receipt, args.receipt, args.publication_revision, release_path, plan, root)
        _, proof = audit_retained(repo, args.receipt, args.publication_revision, args.manifest, args.data_root, release_path)
        proof_path = args.receipt.with_name('post_retirement_audit.json')
        atomic_record(proof_path, proof)
        if any(digest(p) != sha for p, sha in pins.items()):
            raise ValueError('Retained inputs changed during removal/replay')
        result = {'status': 'complete_retirement_and_retained_replay', 'condition': list(condition),
                  'array_files_removed': len(plan), 'logical_payload_bytes_removed': release['deleted_payload_bytes'],
                  'retirement_sha256': digest(release_path), 'post_retirement_audit_sha256': digest(proof_path),
                  'prepared_plan_sha256': digest(plan_path), 'source_sha256': pins,
                  'scope': proof['limits']}
        atomic_record(args.receipt.with_name('retirement_execution_check.json'), result)
        print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
