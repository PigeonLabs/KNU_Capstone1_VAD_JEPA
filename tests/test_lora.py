import torch
from ipad_jepa.backbones import QVLoRA


def test_lora_preserves_base_initially_and_updates_only_qv():
    torch.manual_seed(0)
    base=torch.nn.Linear(8,24)
    wrapper=QVLoRA(base,rank=2,dropout=0)
    x=torch.randn(2,3,8)
    original=base(x).detach()
    torch.testing.assert_close(wrapper(x),original)
    optimizer=torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad],lr=.01)
    wrapper(x).square().mean().backward()
    assert base.weight.grad is None and base.bias.grad is None
    optimizer.step()
    after=wrapper(x)
    torch.testing.assert_close(after[...,8:16],original[...,8:16])
    assert not torch.equal(after[...,:8],original[...,:8])
    assert not torch.equal(after[...,16:],original[...,16:])
