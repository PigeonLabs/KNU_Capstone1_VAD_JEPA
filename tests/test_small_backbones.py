"""B-size adapters retain the online local anchor and separate full-context phase input."""
import pytest
import torch
from torch import nn

from ipad_jepa.small_backbones import SmallBackbone


class ImageEncoder(nn.Module):
    def forward_features(self, frames):
        # Distinct time values expose an accidental average over the whole clip.
        ids = torch.arange(len(frames), dtype=torch.float32)
        return {"x_norm_patchtokens": ids[:, None, None].expand(-1, 576, 768)}


class VideoEncoder(nn.Module):
    def forward(self, video, training=False):
        ids = torch.arange(8, dtype=torch.float32).repeat_interleave(576)
        return ids[None, :, None].expand(len(video), -1, 768)


def adapter(name, encoder):
    model = SmallBackbone.__new__(SmallBackbone)
    nn.Module.__init__(model)
    model.name, model.mode, model.dimension = name, "online", 768
    model.encoder = encoder
    return model


@pytest.mark.parametrize("name,encoder,local_value,context_value", [
    ("dinov3-b", ImageEncoder(), 14.5, 7.5),
    ("vjepa21-b", VideoEncoder(), 7, 3.5),
])
def test_local_last_pair_and_context_pool_are_distinct(name, encoder, local_value, context_value):
    model = adapter(name, encoder)
    local, pooled = model(torch.zeros(1, 3, 16, 384, 384))
    assert local.shape == (1, 576, 768) and pooled.shape == (1, 768)
    assert torch.all(local == local_value) and torch.all(pooled == context_value)


def test_small_adapter_rejects_different_context_before_encoding():
    model = adapter("dinov3-b", ImageEncoder())
    with pytest.raises(ValueError, match="online"):
        model(torch.zeros(1, 3, 8, 384, 384))
