"""Replay retained LoRA evidence after explicitly recorded derived-cache retirement."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import torch
from ipad_jepa.cache_retention import digest, check_saved_inventory
from ipad_jepa.lora_audit import check_training, check_bank
from ipad_jepa.adapted_features import selected_protocol
from summarize_clip_ablation import audit_condition as audit_p3
from summarize_experiments import load_run, check_pair
from summarize_neighbour_ablation import read
from plot_lora_evaluation import calibration_check


def archived_receipt(repo, path, revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Exact full publication revision required')
    relative=Path(path).resolve().relative_to(Path(repo).resolve())
    if relative.parts[0]!='results' or relative.name!='payload_inventory.json':
        raise ValueError('Public derived inventory path required')
    blob=subprocess.check_output(['git','-C',str(repo),'show',revision+':'+str(relative)])
    if blob!=Path(path).read_bytes():raise ValueError('Published inventory differs from current receipt')
    receipt=json.loads(blob); name=Path(receipt['original_witness'])
    if name.is_absolute() or '..' in name.parts:raise ValueError('Original witness escapes archived inventory')
    witness_path=Path(path).parent/name
    witness_blob=subprocess.check_output(['git','-C',str(repo),'show',revision+':'+str(relative.parent/name)])
    if witness_blob!=witness_path.read_bytes() or digest(witness_path)!=receipt['original_witness_sha256']:
        raise ValueError('Original actual cache witness changed')
    return receipt,json.loads(witness_blob)


def replay_tensors(training, local, frozen, frozen_local, folder, witness, normal, condition, weight):
    """Current selected checkpoint/head/bank checks do not need encoder feature arrays."""
    meta,checkpoint=selected_protocol(training)
    check_training(meta,read(training/'lora_training.csv'),condition,weight)
    if digest(training/'lora_training.json')!=witness['training_sha256'] or digest(training/'lora_training.csv')!=witness['training_curve_sha256']:
        raise ValueError('Selected training source changed after original audit')
    old=witness['tensor_provenance']
    expected=[(training/'selected_adapter.pt',old['selected_adapter_sha256']),
              (local/'phase_head.pt',old['joint_head_sha256']),
              (local/'phase_training.json',old['phase_metadata_sha256']),
              (local/'memory.npz',old['rebuilt_memory_sha256']),
              (frozen_local/'memory.npz',old['frozen_memory_sha256'])]
    if any(digest(p)!=sha for p,sha in expected):raise ValueError('Retained selected tensor/bank source changed')
    phase=json.loads((local/'phase_training.json').read_text())
    head=torch.load(local/'phase_head.pt',map_location='cpu',weights_only=True)
    if head.keys()!=checkpoint['head'].keys() or any(not torch.equal(head[k],checkpoint['head'][k]) for k in head):
        raise ValueError('Retained head differs from actual selected joint head')
    if phase['selected_epoch']!=meta['selected_epoch'] or old['selected_epoch']!=meta['selected_epoch']:
        raise ValueError('Retained selected epoch differs')
    if phase['cache_fingerprints']!=normal['normal_cache_fingerprints']:
        raise ValueError('Retained adapted normal cache/head inventory differs')
    frozen_normal=json.loads((frozen/'normal_fit.json').read_text())
    if (set(meta['teacher_fingerprints'])!=set(phase['teacher_cache_fingerprints'])
            or set(meta['teacher_fingerprints'])!=set(frozen_normal['normal_cache_fingerprints'])):
        raise ValueError('Retained fixed teacher provenance differs')
    for name in ('lora_training.json','lora_training.csv'):
        if (folder/name).read_bytes()!=(training/name).read_bytes():raise ValueError('Retained public training export changed')
    with np.load(local/'memory.npz',allow_pickle=False) as bank, np.load(frozen_local/'memory.npz',allow_pickle=False) as base:
        check_bank(bank,normal)
        if np.array_equal(bank['prototypes'],base['prototypes']) or np.array_equal(bank['components'],base['components']):
            raise ValueError('Retained LoRA reused the frozen bank/PCA')
    return meta


def verify_release(receipt, receipt_path, revision, release_path, plan, cache_root):
    release=json.loads(Path(release_path).read_text())
    expected={'status':'complete_verified_derived_payload_retirement',
              'inventory_sha256':digest(receipt_path),'publication_revision':revision,
              'condition':receipt['condition'],'teacher_weight':receipt['teacher_weight']}
    if any(release.get(k)!=v for k,v in expected.items()):
        raise ValueError('Explicit completed retirement receipt required')
    if release.get('publication_CI_conclusion')!='success' or release.get('publication_CI_head_sha')!=revision:
        raise ValueError('Exact successful publication CI evidence required')
    expected_files=[{'relative':str(p.relative_to(cache_root)),
                     'sha256':v['sha256'],'bytes':v['bytes'],'observed_absent_after_unlink':True} for p,v in plan]
    if release.get('deleted_files')!=expected_files or any(p.exists() or p.is_symlink() for p,_ in plan):
        raise ValueError('Actual intentional payload removal inventory differs')
    return release


def audit_retained(repo, receipt_path, revision, manifest_path, data_root, release_path):
    receipt,witness=archived_receipt(repo,receipt_path,revision)
    if receipt['manifest_sha256']!=digest(manifest_path):raise ValueError('Archived manifest changed')
    if (witness['auditor_sha256']!=digest('scripts/summarize_lora_matrix.py')
            or witness['protocol_audit_sha256']!=digest('src/ipad_jepa/lora_audit.py')
            or witness['selected_tensor_verifier_sha256']!=digest('scripts/plot_lora_evaluation.py')):
        raise ValueError('Original actual condition verifier changed')
    manifest=json.loads(Path(manifest_path).read_text())['sequences']
    condition=tuple(receipt['condition']);weight=receipt['teacher_weight']
    relative=Path(condition[0])/condition[1]/condition[2]/f'seed{condition[3]}'
    zero=weight==0.
    expected_paths={'public_root':str((Path('results/stage05/ablations/teacher_weight/T0') if zero else Path('results/stage04'))/relative),
                    'training_root':str(Path('artifacts/lora_teacher0' if zero else 'artifacts/lora')/relative),
                    'local_root':str(Path('artifacts/runs_lora_teacher0' if zero else 'artifacts/runs_lora')/relative)}
    if any(receipt[k]!=v for k,v in expected_paths.items()):raise ValueError('Retained treatment namespaces differ')
    folder=Path(repo)/receipt['public_root'];training=Path(repo)/receipt['training_root'];local=Path(repo)/receipt['local_root']
    for name,sha in witness['public_sources_sha256'].items():
        if digest(folder/name)!=sha:raise ValueError('Retained original evaluated public source changed')
    info=json.loads((folder/'metrics.json').read_text())
    if info['status']!='complete_device_evaluation' or tuple(info[k] for k in ('backbone','mode','device','seed'))!=condition:
        raise ValueError('Retained completed evaluation condition differs')
    normal,sources=calibration_check(folder,info,manifest,Path(data_root))
    sources_now={name:digest(Path(repo)/'src/ipad_jepa'/(name+'.py')) for name in receipt['source_sha256']}
    root,plan=check_saved_inventory(repo,Path(repo)/receipt['cache_root'],receipt,witness,manifest,normal,sources_now)
    verify_release(receipt,receipt_path,revision,release_path,plan,root)
    frozen=Path(repo)/'results/stage02'/relative;frozen_local=Path(repo)/'artifacts/runs'/relative
    replay_tensors(training,local,frozen,frozen_local,folder,witness,normal,condition,weight)
    _,p3=audit_p3(folder,manifest,Path(data_root),*condition,16)
    frozen_info=json.loads((frozen/'metrics.json').read_text())
    _,frozen_sources=calibration_check(frozen,frozen_info,manifest,Path(data_root))
    audit_p3(frozen,manifest,Path(data_root),*condition,16)
    runs={v:load_run(folder,v,info) for v in ('P0','P1','P2','P3')}
    for variant,run in runs.items():
        check_pair(runs['P3'],run);check_pair(run,load_run(frozen,variant,frozen_info))
    proof={'status':'passed_retained_lora_condition_replay','condition':list(condition),'teacher_weight':weight,
           'cache_state':'derived_payloads_intentionally_retired_metadata_targets_retained',
           'historical_full_payload_inventory_sha256':digest(receipt_path),
           'historical_original_condition_audit_sha256':receipt['original_witness_sha256'],
           'publication_revision':revision,'release_sha256':digest(release_path),
           'normal_thresholds_replayed':8,'current_public_sources_sha256':sources,
           'current_frozen_sources_sha256':frozen_sources,'P3_replay':p3,
           'current_selected_adapter_sha256':digest(training/'selected_adapter.pt'),
           'current_head_sha256':digest(local/'phase_head.pt'),'current_bank_sha256':digest(local/'memory.npz'),
           'verifier_sha256':digest(__file__),'historical_array_files':len(plan),
           'historical_payload_bytes':receipt['derived_bytes_checked'],
           'limits':'Payload finite checks/hashes and actual original cache verification are historical, archived before intentional removal. Current metadata/targets/selected tensors/bank/calibration/GT/time/score/alarm/metrics independently replayed. No current payload presence, encoder replay, PCA refit, GPU distance/temperature-sample replay or runtime claim.'}
    return runs,proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--publication-revision',required=True)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    p.add_argument('--data-root',type=Path,default=Path('../IPAD_dataset/IPAD_dataset'))
    args=p.parse_args()
    _,proof=audit_retained(Path.cwd(),args.receipt,args.publication_revision,args.manifest,args.data_root,args.release)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ['current_public_sources_sha256','current_frozen_sources_sha256','P3_replay']},indent=2))


if __name__=='__main__':main()
