import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path=Path(__file__).parents[1]/'scripts/summarize_experiments.py'
    spec=importlib.util.spec_from_file_location('summarize_experiments',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_video_draws_are_shared_across_seeds(monkeypatch):
    mod=module(); seen=[]
    def statistic(run,selected):
        seen.append(tuple(selected)); return np.array([.5,.5])
    monkeypatch.setattr(mod,'statistic',statistic)
    values,rejected=mod.bootstrap([dict.fromkeys(['01','02','03'])]*3,count=20)
    assert values.shape==(20,2) and rejected==0
    assert all(seen[i]==seen[i+1]==seen[i+2] and len(seen[i])==3 for i in range(0,len(seen),3))
    assert any(len(set(draw))<3 for draw in seen)


def test_paired_comparison_rejects_frame_or_label_mismatch():
    mod=module()
    a={'01':(np.array([19,20]),np.array([0,1]),np.array([.1,.2]))}
    b={'01':(np.array([20,21]),np.array([0,1]),np.array([.1,.2]))}
    with pytest.raises(ValueError,match='frame/label'):
        mod.check_pair(a,b)


def test_score_reader_accepts_binary_float_labels_and_rejects_fractional_labels(tmp_path):
    mod=module(); folder=tmp_path/'P0'; folder.mkdir()
    source=folder/'01.csv'
    source.write_text('frame,label,valid,score\n19,0.0,1,0.1\n20,1.0,1,0.9\n')
    run=mod.load_run(tmp_path,'P0')
    assert run['01'][1].tolist()==[0,1]
    source.write_text('frame,label,valid,score\n19,0.2,1,0.1\n20,1.0,1,0.9\n')
    with pytest.raises(ValueError,match='binary'):
        mod.load_run(tmp_path,'P0')


def macro_inputs():
    stored={}
    for number in range(1,5):
        device=f'R{number:02d}'; repeats=20 if number==1 else 1
        labels=np.tile([0,1],repeats).astype(np.int8)
        for variant in ['P0','P3']:
            scores=labels.astype(float) if number==1 or variant=='P3' else 1-labels.astype(float)
            run={video:(np.arange(len(labels)),labels.copy(),scores.copy()) for video in ['01','02','03']}
            stored[('model','offline',device,variant)]=([run]*3,None)
    return stored


def test_macro4_uses_equal_device_weights_and_paired_component_delta():
    mod=module(); inputs=macro_inputs(); result=mod.macro_four_devices(inputs,count=20)
    rows={row['variant']:row for row in result['results']}
    assert rows['P0']['auroc_mean']==.25
    assert rows['P3']['auroc_mean']==1
    # Pooling would let the much longer, perfect R01 dominate the result.
    all_videos={f'{device}/{video}':value for (_,_,device,variant),(runs,_) in inputs.items()
                if variant=='P0' for video,value in runs[0].items()}
    assert mod.statistic(all_videos,list(all_videos))[0]>.8
    delta=result['paired_deltas'][0]
    assert delta['auroc_delta']==delta['auroc_delta_ci_low']==delta['auroc_delta_ci_high']==.75


def test_macro4_video_draws_are_independent_between_devices_and_shared_between_conditions(monkeypatch):
    mod=module(); seen={}; inputs=macro_inputs()
    for key,(runs,_) in inputs.items():
        for run in runs:
            seen[id(run)]=(key,[])
    def statistic(run,selected):
        seen[id(run)][1].append(tuple(selected))
        return np.array([.5,.5])
    monkeypatch.setattr(mod,'statistic',statistic)
    mod.macro_four_devices(inputs,count=20)
    draws={key:history[3:] for key,history in seen.values()}
    # Each bootstrap video selection repeats for all three seeds.
    for history in draws.values():
        assert len(history)==60
        assert all(history[i]==history[i+1]==history[i+2] for i in range(0,60,3))
    for number in range(1,5):
        device=f'R{number:02d}'
        assert draws[('model','offline',device,'P0')]==draws[('model','offline',device,'P3')]
    assert draws[('model','offline','R01','P0')]!=draws[('model','offline','R02','P0')]


def test_macro4_never_uses_a_partial_device_set_and_rejects_mismatched_targets():
    mod=module(); inputs=macro_inputs()
    del inputs[('model','offline','R04','P3')]
    result=mod.macro_four_devices(inputs,count=20)
    assert [row['variant'] for row in result['results']]==['P0']
    assert result['incomplete_conditions'][0]['missing_devices']==['R04']
    assert not result['paired_deltas']
    runs,_=inputs[('model','offline','R02','P3')]
    runs[0]['01']=(np.array([19,20]),np.array([0,1]),np.array([.1,.9]))
    with pytest.raises(ValueError,match='frame/label'):
        mod.macro_four_devices(inputs,count=20)
