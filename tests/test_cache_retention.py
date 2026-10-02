"""Cache retirement inventories must preserve provenance and exclude active inputs."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from ipad_jepa.cache_retention import (condition_cache, member, inspect_array,
                                       prepare_inventory, check_saved_inventory, digest)

KEY=('dinov3-l','offline','R01',0)


def fixture(tmp_path):
    cache=tmp_path/'artifacts/features_lora/dinov3-l/offline/R01/seed0'
    sources={'adaptation':'a'*64,'adapted_features':'b'*64}
    row={'device':'R01','partition':'training','sequence':'01','split':'fit','frames':20,
         'frames_content_sha256':'frames','names_sha256':'names'}
    identity={'backbone':'dinov3-l','mode':'offline','fit_stride':4,'clip_frames':16,
              'adapter_sha256':'selected-identity'}
    spec={**identity,'device':'R01','partition':'training','sequence':'01','split':'fit','frames':20,
          'frame_content_sha256':'frames','frame_names_sha256':'names',
          'adaptation':'qv_lora_last4','adaptation_seed':0,'selected_epoch':3,
          'selected_adapter_sha256':'selected','adaptation_source_sha256':sources['adaptation'],
          'adapted_reader_sha256':sources['adapted_features']}
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    folder=cache/'R01/training/01';folder.mkdir(parents=True)
    (folder/'meta.json').write_text(json.dumps({'status':'complete','fingerprint':fingerprint,'spec':spec}))
    np.save(folder/'targets.npy',np.array([8,12],dtype=np.int64))
    np.save(folder/'patch.npy',np.ones((2,576,1024),dtype=np.float16))
    np.save(folder/'global.npy',np.ones((2,1024),dtype=np.float16))
    public=tmp_path/'results';public.mkdir();(public/'metrics.json').write_text('{}')
    old={'partition':'training','sequence':'01','clips':2,'fingerprint':fingerprint,
         'metadata_sha256':digest(folder/'meta.json')}
    proof={'status':'passed_completed_lora_condition','teacher_weight':1.,
           'backbone':'dinov3-l','mode':'offline','device':'R01','seed':0,
           'selected_adapter_sha256':'selected',
           'public_sources_sha256':{'metrics.json':digest(public/'metrics.json')},
           'tensor_provenance':{'selected_epoch':3,'cache_checks':[old]}}
    normal={'cache_identity':identity}
    receipt=prepare_inventory(tmp_path,cache,[row],proof,public,normal,KEY,1.,sources)
    return cache,[row],proof,normal,sources,receipt


def test_real_payload_inventory_has_full_hash_and_preserves_targets_and_metadata(tmp_path):
    cache,rows,proof,normal,sources,receipt=fixture(tmp_path)
    root,plan=check_saved_inventory(tmp_path,cache,receipt,proof,rows,normal,sources)
    assert root==cache and len(plan)==2
    assert {p.name for p,_ in plan}=={'patch.npy','global.npy'}
    assert all(record['finite_payload_checked'] for _,record in plan)
    assert receipt['derived_bytes_checked']==sum(p.stat().st_size for p,_ in plan)
    assert (cache/'R01/training/01/targets.npy').exists()
    assert (cache/'R01/training/01/meta.json').exists()


@pytest.mark.parametrize('namespace',['features','weights','features_clip8','lora','runs_lora','../IPAD_dataset'])
def test_frozen_teacher_raw_models_and_other_experiments_cannot_be_retirement_roots(tmp_path,namespace):
    with pytest.raises(ValueError):
        condition_cache(tmp_path,tmp_path/'artifacts'/namespace/'dinov3-l/offline/R01/seed0',KEY,1.)


def test_teacher_zero_and_one_namespaces_cannot_be_interchanged(tmp_path):
    root=tmp_path/'artifacts/features_lora_teacher0/dinov3-l/offline/R01/seed0'
    assert condition_cache(tmp_path,root,KEY,0.)==root
    with pytest.raises(ValueError):condition_cache(tmp_path,root,KEY,1.)


@pytest.mark.parametrize('path',['R01/training/01/targets.npy','R01/training/01/meta.json',
                                 'R01/training/../patch.npy','/tmp/patch.npy','R01/training/01/selected_adapter.pt'])
def test_payload_plan_cannot_include_targets_metadata_or_path_escape(tmp_path,path):
    with pytest.raises(ValueError):member(tmp_path,path,('patch.npy','global.npy'))


def test_symlinks_and_nonfinite_or_wrong_width_payloads_are_rejected(tmp_path):
    cache,rows,proof,normal,sources,receipt=fixture(tmp_path)
    path=cache/'R01/training/01/patch.npy'
    path.unlink();np.save(tmp_path/'external.npy',np.ones((2,576,1024),np.float16))
    path.symlink_to(tmp_path/'external.npy')
    with pytest.raises(ValueError,match='Symlink'):member(cache,'R01/training/01/patch.npy',('patch.npy',))
    path.unlink();np.save(path,np.full((2,576,1024),np.inf,np.float16))
    with pytest.raises(ValueError,match='Nonfinite'):inspect_array(path,(2,576,1024))
    np.save(path,np.ones((2,576,768),np.float16))
    with pytest.raises(ValueError,match='shape'):inspect_array(path,(2,576,1024))


def test_missing_payload_cannot_be_inventoried_and_changed_target_cannot_pass_retained_validation(tmp_path):
    cache,rows,proof,normal,sources,receipt=fixture(tmp_path)
    target=cache/'R01/training/01/targets.npy';np.save(target,np.array([8,11],dtype=np.int64))
    with pytest.raises(AssertionError):check_saved_inventory(tmp_path,cache,receipt,proof,rows,normal,sources)
    with pytest.raises(FileNotFoundError):inspect_array(cache/'absent.npy',(2,576,1024))


def test_modified_receipt_count_source_or_payload_shape_is_rejected(tmp_path):
    cache,rows,proof,normal,sources,receipt=fixture(tmp_path)
    for mutate in [lambda r:r.update(array_files_checked=1),
                   lambda r:r.update(derived_bytes_checked=0),
                   lambda r:r['records'][0]['arrays']['patch.npy'].update(shape=[2,576,768]),
                   lambda r:r['records'][0]['arrays']['global.npy'].update(finite_payload_checked=False)]:
        wrong=copy.deepcopy(receipt);mutate(wrong)
        with pytest.raises(ValueError):check_saved_inventory(tmp_path,cache,wrong,proof,rows,normal,sources)
    with pytest.raises(ValueError):check_saved_inventory(tmp_path,cache,receipt,proof,rows,normal,{**sources,'adaptation':'changed'})


def test_archived_payload_hash_header_and_byte_identity_must_be_complete(tmp_path):
    cache,rows,proof,normal,sources,receipt=fixture(tmp_path)
    for field,value in [('sha256','z'*64),('npy_header_sha256','short'),
                        ('npy_header_bytes',129),('file_identity',[1,2,3,4])]:
        changed=copy.deepcopy(receipt)
        changed['records'][0]['arrays']['patch.npy'][field]=value
        with pytest.raises(ValueError,match='incomplete'):
            check_saved_inventory(tmp_path,cache,changed,proof,rows,normal,sources)
