from types import SimpleNamespace
import numpy as np
import pytest
from ipad_jepa.cache_data import normal_candidates
from ipad_jepa.experiment import alarms,metrics


def test_candidate_sampling_uses_equal_video_quotas_and_only_fit():
    fit=[]
    for i,n in enumerate([4,20]):
        targets=np.linspace(1,19,n,dtype=int)
        patches=np.full((n,8,3),i,dtype=np.float16)
        fit.append(SimpleNamespace(row={'partition':'training','split':'fit','frames':20,'sequence':str(i)},
                                   targets=targets,patches=patches))
    x,phase,groups,info=normal_candidates(fit,bins=2,limit=16,seed=8)
    assert x.shape==(32,3)
    assert np.bincount(groups).tolist()==[16,16]
    assert [r['per_video'] for r in info]==[{'0':8,'1':8}]*2
    assert np.bincount(np.floor(phase*2).astype(int)).tolist()==[16,16]
    fit[0].row['split']='diagnostic'
    with pytest.raises(ValueError,match='normal fit'):
        normal_candidates(fit,bins=2,limit=16)


def test_alarm_streak_resets_at_invalid_frames():
    scores=np.array([2.,2.,2.,2.,2.,2.,0.,2.,2.,2.])
    valid=np.array([True,True,False,True,True,True,True,True,True,True])
    assert alarms(scores,1,valid).tolist()==[False,False,False,False,False,True,False,False,False,True]
    values=metrics(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
    assert values['frame_auroc']==1 and values['frame_ap']==1
    with pytest.raises(ValueError,match='Both labels'):
        metrics(np.array([0,0]),np.array([1.,2.]))


def test_failed_rerun_removes_old_completion_marker(tmp_path):
    from ipad_jepa.experiment import run
    out=tmp_path/'result'; out.mkdir()
    marker=out/'metrics.json'; marker.write_text('{"status":"complete_device_evaluation"}')
    with pytest.raises(ValueError,match='Missing or mixed'):
        run(tmp_path/'cache',[],tmp_path/'phase',out,tmp_path/'local',0,tmp_path/'data')
    assert not marker.exists()
