"""Historical cache evidence must be archived, unchanged and explicitly released."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_retained_lora import archived_receipt, verify_release
from ipad_jepa.cache_retention import digest


def archive(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    folder=repo/'results/cache_retention/T1/dinov3-l/offline/R01/seed0';folder.mkdir(parents=True)
    witness=folder/'original_condition_audit.json';witness.write_text('{"status":"fixture"}\n')
    receipt=folder/'payload_inventory.json'
    receipt.write_text(json.dumps({'original_witness':witness.name,'original_witness_sha256':digest(witness)}))
    for args in [('init','-q'),('config','user.name','Fixture'),('config','user.email','fixture@example.invalid'),
                 ('add','results'),('commit','-qm','Archive fixture')]:
        subprocess.run(['git','-C',str(repo),*args],check=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
    return repo,receipt,witness,commit


def test_exact_archived_receipt_and_original_witness_bytes_are_required(tmp_path):
    repo,path,witness,commit=archive(tmp_path)
    assert archived_receipt(repo,path,commit)[1]['status']=='fixture'
    witness.write_text('{"status":"changed"}\n')
    with pytest.raises(ValueError,match='witness changed'):archived_receipt(repo,path,commit)


def test_uncommitted_inventory_changes_and_ambiguous_revisions_are_rejected(tmp_path):
    repo,path,witness,commit=archive(tmp_path)
    with pytest.raises(ValueError,match='full publication'):archived_receipt(repo,path,'HEAD')
    with pytest.raises(ValueError,match='full publication'):archived_receipt(repo,path,'a'*7)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='inventory differs'):archived_receipt(repo,path,commit)


def release_fixture(tmp_path):
    root=tmp_path/'cache';folder=root/'R01/training/01';folder.mkdir(parents=True)
    receipt={'condition':['dinov3-l','offline','R01',0],'teacher_weight':1.}
    receipt_path=tmp_path/'payload_inventory.json';receipt_path.write_text(json.dumps(receipt))
    plan=[]
    for name in ('patch.npy','global.npy'):
        path=folder/name;path.write_bytes(b'fixture');plan.append((path,{'sha256':digest(path),'bytes':path.stat().st_size}))
    revision='a'*40
    release={'status':'complete_verified_derived_payload_retirement','inventory_sha256':digest(receipt_path),
             'publication_revision':revision,'condition':receipt['condition'],'teacher_weight':1.,
             'publication_CI_conclusion':'success','publication_CI_head_sha':revision,
             'deleted_files':[{'relative':str(p.relative_to(root)),'sha256':r['sha256'],'bytes':r['bytes'],
                               'observed_absent_after_unlink':True} for p,r in plan]}
    release_path=tmp_path/'release.json';release_path.write_text(json.dumps(release))
    return root,receipt,receipt_path,revision,release,release_path,plan


def test_intentional_completed_release_cannot_be_claimed_while_payloads_exist(tmp_path):
    root,receipt,path,revision,release,release_path,plan=release_fixture(tmp_path)
    with pytest.raises(ValueError,match='removal inventory'):verify_release(receipt,path,revision,release_path,plan,root)
    for p,_ in plan:p.unlink()
    assert verify_release(receipt,path,revision,release_path,plan,root)['status'].startswith('complete')


@pytest.mark.parametrize('field,value',[('status','in_progress'),('teacher_weight',0.),
                                      ('publication_CI_conclusion','failure'),('publication_CI_head_sha','b'*40)])
def test_partial_wrong_teacher_or_nonmatching_failed_publication_is_rejected(tmp_path,field,value):
    root,receipt,path,revision,release,release_path,plan=release_fixture(tmp_path)
    for p,_ in plan:p.unlink()
    release[field]=value;release_path.write_text(json.dumps(release))
    with pytest.raises(ValueError):verify_release(receipt,path,revision,release_path,plan,root)
