import json
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from ipad_jepa.ipad_baseline import train


def test_native_pilot_cannot_resume_as_full_baseline(tmp_path):
    out=tmp_path/'run'; out.mkdir()
    (out/'training.json').write_text(json.dumps({'max_steps':2}))
    with pytest.raises(ValueError,match='pilot'):
        train(tmp_path,[],tmp_path,out,0,resume=True)
    with pytest.raises(FileExistsError,match='use --resume'):
        train(tmp_path,[],tmp_path,out,0,resume=False)


def test_native_evaluator_rejects_pilot_before_gpu_or_raw_data(tmp_path):
    from ipad_jepa.ipad_evaluate import evaluate
    (tmp_path/'training.json').write_text(json.dumps({'status':'pilot_complete'}))
    with pytest.raises(ValueError,match='not a completed'):
        evaluate(tmp_path,[],tmp_path,tmp_path,tmp_path/'results')


def test_resume_restores_optimizer_shuffle_and_dropout_state(tmp_path,monkeypatch):
    import ipad_jepa.ipad_baseline as native
    import ipad_jepa.audit as audit

    class Clips(Dataset):
        def __init__(self,*args,**kwargs): pass
        def __len__(self): return 16
        def __getitem__(self,i): return torch.full((2,),float(i)/16),i,0,i

    class Tiny(nn.Module):
        def __init__(self,fail=False):
            super().__init__(); self.project=nn.Linear(2,2); self.phase=nn.Linear(2,200)
            self.dropout=nn.Dropout(.3); self.calls=0; self.fail=fail
        def forward(self,x):
            self.calls+=1
            if self.fail and self.calls==11: raise RuntimeError('simulated interruption')
            hidden=self.dropout(self.project(x))
            return {'output':hidden,'recon_index':self.phase(hidden),
                    'att':hidden.softmax(1)[:,:,None]}

    monkeypatch.setattr(native,'NativeClips',Clips)
    loader=native.DataLoader
    def local_loader(*args,**kwargs):
        kwargs['num_workers']=0; kwargs['pin_memory']=False
        return loader(*args,**kwargs)
    monkeypatch.setattr(native,'DataLoader',local_loader)
    monkeypatch.setattr(audit,'inspect_frames',lambda *_:{'frames_content_sha256':'fixed'})
    monkeypatch.setattr(native.subprocess,'check_output',lambda *_:b'pinned')
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(nn.Module,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *_:None)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda *_:None)
    rows=[{'device':'R01','partition':'training','split':'fit','sequence':'01',
           'relative_directory':'frames','frames_content_sha256':'fixed'}]
    monkeypatch.setattr(native,'load_model',lambda *args:Tiny())
    train(tmp_path,rows,tmp_path,tmp_path/'full',3,epochs=6)
    expected=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True)
    monkeypatch.setattr(native,'load_model',lambda *args:Tiny(fail=True))
    with pytest.raises(RuntimeError,match='simulated interruption'):
        train(tmp_path,rows,tmp_path,tmp_path/'resumed',3,epochs=6)
    saved=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
    assert saved['epoch']==5
    monkeypatch.setattr(native,'load_model',lambda *args:Tiny())
    train(tmp_path,rows,tmp_path,tmp_path/'resumed',3,epochs=6,resume=True)
    actual=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
    assert actual['epoch']==6 and actual['curves']==expected['curves']
    for name in expected['model']: torch.testing.assert_close(actual['model'][name],expected['model'][name],rtol=0,atol=0)
    for key in expected['optimizer']['state']:
        for name in expected['optimizer']['state'][key]:
            torch.testing.assert_close(actual['optimizer']['state'][key][name],expected['optimizer']['state'][key][name],rtol=0,atol=0)
