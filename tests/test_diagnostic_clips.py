"""Model inputs obey transformed-timeline clip boundaries and preserve source pixels."""
import numpy as np
import pytest
import torch
from ipad_jepa.diagnostics import scenarios
from ipad_jepa.diagnostic_evaluation import DiagnosticClips, check_source_pins
from ipad_jepa.features import file_hash

ROW=dict(device='R01',sequence='02',partition='training',split='diagnostic',frames=80)


def test_json_source_pins_accept_string_paths_and_reject_changed_source(tmp_path):
    source=tmp_path/'protocol.py'
    source.write_text('original protocol\n')
    pinned={str(source):file_hash(source)}
    check_source_pins(pinned)
    source.write_text('changed protocol\n')
    with pytest.raises(ValueError,match='Active diagnostic protocol sources changed'):
        check_source_pins(pinned)


def rgb():
    return np.broadcast_to(np.arange(80,dtype=np.uint8)[:,None,None,None],(80,384,384,3)).copy()


@pytest.mark.parametrize('mode,target,first,last',[('online',20,5,20),('offline',20,12,27)])
def test_real_tensor_clip_uses_only_declared_transformed_timeline(mode,target,first,last):
    plan=scenarios(ROW)[0]
    dataset=DiagnosticClips(rgb(),plan,mode)
    item=int(np.flatnonzero(dataset.targets==target)[0])
    clip,actual=dataset[item]
    assert clip.shape==(3,16,384,384) and actual==target
    expected=(torch.arange(first,last+1).float()/255-.485)/.229
    torch.testing.assert_close(clip[0,:,0,0],expected,atol=1e-6,rtol=0)
    if mode=='online':assert last==target
    else:assert last-target==7


def test_stall_and_appearance_apply_on_received_timeline_without_mutating_original():
    source=rgb();before=source.copy()
    plan=next(s for s in scenarios(ROW) if s.name=='stall_8__occlusion_1')
    dataset=DiagnosticClips(source,plan,'online')
    target=plan.temporal_start
    frame=dataset.frame(target)
    y0,y1,x0,x1=plan.box
    expected_black=-dataset.mean/dataset.std
    torch.testing.assert_close(frame[:,y0:y1,x0:x1],expected_black.expand(3,y1-y0,x1-x0))
    expected_outside=(torch.tensor([(target-1)/255]*3)-dataset.mean[:,0,0])/dataset.std[:,0,0]
    torch.testing.assert_close(frame[:,0,0],expected_outside,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(source,before)
