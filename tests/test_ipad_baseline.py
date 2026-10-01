from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from ipad_jepa.ipad_baseline import repaired_memory_forward,latent_score,losses


def memory():
    return SimpleNamespace(weight=torch.nn.Parameter(torch.randn(20,4)),mem_dim=20,shrink_thres=0,addressing='paper200')


def test_repaired_memory_is_per_video_and_has_phase_gradients():
    torch.manual_seed(3); m=memory()
    x=torch.randn(6,4)
    p=torch.randn(2,200,requires_grad=True)
    batch=repaired_memory_forward(m,x,p)
    separate=torch.cat([repaired_memory_forward(m,x[i*3:(i+1)*3],p[i:i+1])['output'] for i in range(2)])
    torch.testing.assert_close(batch['output'],separate)
    batch['output'].sum().backward()
    assert p.grad is not None and p.grad.abs().sum()>0
    assert m.weight.grad is not None
    torch.testing.assert_close(batch['att'].sum(1),torch.ones(6))


def test_literal_public126_empty_high_phase_window_is_preserved():
    m=memory(); m.addressing='public126'
    x=torch.randn(3,4); p=torch.zeros(1,200); p[0,199]=3
    actual=repaired_memory_forward(m,x,p)['att']
    torch.testing.assert_close(actual,(x@m.weight.T).softmax(1))


def test_b1_does_not_call_decoder_and_loss_matches_public_formula():
    class Fake(nn.Module):
        def transformer_encoder(self,x): return x
        def period(self,x): return torch.ones(len(x),200)
        def mem_rep(self,x,p): return {'output':x*.5,'att':torch.ones(len(x),2,4,2,2)*.5}
        def transformer_decoder(self,x): raise AssertionError('B1 called the decoder')
    x=torch.ones(2,3,4,2,2)
    torch.testing.assert_close(latent_score(Fake(),x),torch.full((2,),.25))
    pred={'output':x*.5,'att':torch.ones(2,2,4,2,2)*.5,'recon_index':torch.zeros(2,200)}
    loss,parts=losses(pred,x,torch.tensor([0,1]))
    expected=.25+.0002*np.log(2)+.02*np.log(200)
    assert abs(float(loss)-expected)<1e-6
