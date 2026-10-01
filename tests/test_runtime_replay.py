import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def module():
    source=Path(__file__).parents[1]/'scripts/benchmark_runtime.py'
    spec=importlib.util.spec_from_file_location('benchmark_runtime',source)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('mode',['online','offline'])
def test_paced_full_and_pixel_buffer_use_same_windows_and_report_real_queue_delay(tmp_path,monkeypatch,mode):
    mod=module(); clock=SimpleNamespace(now=0.)
    monkeypatch.setattr(mod,'time',SimpleNamespace(perf_counter=lambda:clock.now,
        sleep=lambda seconds:setattr(clock,'now',clock.now+seconds)))
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(mod,'inspect_frames',lambda folder:{'frames_content_sha256':'source'})
    class Reader:
        def __init__(self,*a,**k): self.paths=list(range(40))
        def frame(self,index):
            clock.now+=.001
            return torch.full((3,2,2),float(index))
    class Encoder:
        def __init__(self): self.inputs=[]
        def __call__(self,clip):
            self.inputs.append(clip[0,0,:,0,0].tolist())
            clock.now+=.05
            return torch.ones(1,1,1),clip[:,:,-1].mean().reshape(1,1)
    monkeypatch.setattr(mod,'ClipDataset',Reader)
    def score(local,pooled,*a):
        clock.now+=.005
        return float(pooled.item()/100),4.,0.,3.,2.
    monkeypatch.setattr(mod,'score_features',score)
    labels=np.zeros(40,dtype=np.int8); labels[25:31]=1
    np.save(tmp_path/'labels.npy',labels)
    row={'frames':40,'relative_directory':'frames','frames_content_sha256':'source',
         'label_file':'labels.npy','label_sha256':mod.file_hash(tmp_path/'labels.npy'),'sequence':'03'}
    cal={'median':[0,0],'mad_scale':[1,1],'threshold':1}
    results=[]
    for implementation in ['full','buffer']:
        clock.now=0.; encoder=Encoder()
        args=SimpleNamespace(data_root=tmp_path,mode=mode,implementation=implementation,
                             precision='fp32',arrival_fps=30)
        trace,report=mod.replay(args,row,encoder,None,None,cal,100)
        assert encoder.inputs==[list(range(last-15,last+1)) for last in range(15,40)]
        assert len(trace)==40 and report['queue_growth_ms']>0 and report['input_fps']<30
        eligible=[item for item in trace if item['inference_valid']]
        assert eligible[0]['target_frame']==19
        assert all(item['emitted_seconds']>=item['arrival_seconds'] for item in trace)
        if mode=='offline':
            assert eligible[-1]['target_frame']==32
            assert all(item['target_latency_ms']>=7000/30 for item in eligible)
        else:
            assert eligible[-1]['target_frame']==39
            assert all(item['alarm'] for item in eligible[-7:])
        results.append((trace,report))
    # Compare all eligible scores/alarms/timestamps; initial missing temporal scores are NaN.
    for first,second in zip(results[0][0],results[1][0]):
        if first['inference_valid']: assert first==second
    assert results[0][1]==results[1][1]


def test_runtime_refuses_concurrent_gpu_processes(monkeypatch):
    mod=module()
    monkeypatch.setattr(mod.subprocess,'check_output',lambda *a,**k:f'{os.getpid()}\n9999999\n')
    with pytest.raises(RuntimeError,match='GPU isolation'): mod.require_isolated_gpu()
    monkeypatch.setattr(mod.subprocess,'check_output',lambda *a,**k:f'{os.getpid()}\n')
    mod.require_isolated_gpu()


def test_runtime_rejects_partial_phase_training_before_loading_models(tmp_path):
    mod=module(); results=tmp_path/'results'; run=tmp_path/'run'
    results.mkdir(); run.mkdir()
    (results/'normal_fit.json').write_text(json.dumps({'status':'normal_fit_and_calibration_complete',
        'backbone':'dinov3-l','mode':'online','device':'R01','seed':0}))
    (run/'phase_training.json').write_text(json.dumps({'status':'complete','epochs':19,'seed':0}))
    args=SimpleNamespace(results=results,run=run,model='dinov3-l',mode='online',device='R01',seed=0)
    with pytest.raises(ValueError,match='completed primary'): mod.fixed_components(args)
