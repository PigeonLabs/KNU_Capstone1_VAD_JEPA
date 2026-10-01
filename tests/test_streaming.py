import torch
import pytest
from torch import nn
from ipad_jepa.streaming import DinoFrameRing


class FrameEncoder(nn.Module):
    def __init__(self): super().__init__(); self.calls=0
    def forward_features(self, x):
        assert x.is_contiguous()
        self.calls+=len(x)
        return {'x_norm_patchtokens':x.mean((2,3))[:,None].expand(-1,2,-1)}


def test_causal_reuse_matches_clip_mean_and_retains_only_past_frames():
    encoder=FrameEncoder().eval(); ring=DinoFrameRing(encoder,4)
    for t in range(9):
        ring.push(torch.full((3,2,2),float(t)),t)
        if t<3:
            with pytest.raises(ValueError,match='past frames'): ring.features()
        else:
            local,global_=ring.features()
            torch.testing.assert_close(local,torch.full((1,2,3),t-.5))
            torch.testing.assert_close(global_,torch.full((1,3),t-1.5))
        assert len(ring.tokens)<=4
    assert encoder.calls==9
    with pytest.raises(ValueError,match='sequential'): ring.push(torch.zeros(3,2,2),10)
    ring.reset(); assert not ring.tokens and ring.next_frame==0


def test_left_padding_repeats_first_frame_only_when_explicitly_requested():
    ring=DinoFrameRing(FrameEncoder().eval(),4)
    ring.push(torch.ones(3,2,2),0)
    local,global_=ring.features(padding=True)
    torch.testing.assert_close(local,torch.ones(1,2,3))
    torch.testing.assert_close(global_,torch.ones(1,3))
    with pytest.raises(ValueError,match='evaluation'): DinoFrameRing(FrameEncoder(),4).push(torch.ones(3,2,2),0)


def test_frame_input_matches_full_clip_contiguous_storage():
    frame=torch.ones(2,2,3).permute(2,0,1)
    assert not frame.is_contiguous()
    ring=DinoFrameRing(FrameEncoder().eval(),4)
    ring.push(frame,0)
    local,global_=ring.features(padding=True)
    torch.testing.assert_close(local,torch.ones(1,2,3))
    torch.testing.assert_close(global_,torch.ones(1,3))
