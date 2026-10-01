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
