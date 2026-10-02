import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from benchmark_runtime import configure_inference_encoder, json_scalar
from ipad_jepa.adaptation import adapter_state, install_adapters, load_adapter_state
from ipad_jepa.streaming import DinoFrameRing


class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module() for _ in range(4)])
        for block in self.blocks:
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(3, 9)

    def forward_features(self, frames):
        value = frames.mean((2, 3))
        for block in self.blocks:
            value = block.attn.qkv(value)[:, :3]
        return {"x_norm_patchtokens": value[:, None]}


def backbone():
    model = nn.Module()
    model.encoder = FrameEncoder()
    return model


def test_selected_nonzero_lora_is_preserved_and_can_enter_fixed_frame_ring():
    torch.manual_seed(0)
    reference = backbone()
    model = backbone()
    model.load_state_dict(reference.state_dict())
    install_adapters(reference)
    selected = {name: torch.full_like(value, .125)
                for name, value in adapter_state(reference).items()}
    load_adapter_state(reference, selected)
    frame = torch.ones(3, 2, 2)
    expected = reference.encoder.forward_features(frame[None])["x_norm_patchtokens"]
    unadapted = model.encoder.forward_features(frame[None])["x_norm_patchtokens"]
    assert not torch.allclose(expected, unadapted)
    # eval alone leaves adapter parameters trainable and is insufficient for reuse.
    with pytest.raises(ValueError, match="frozen evaluation"):
        DinoFrameRing(reference.encoder).push(frame, 0)
    configure_inference_encoder(model, selected)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(module.training for module in model.modules())
    for name, parameter in model.named_parameters():
        if name in selected:
            torch.testing.assert_close(parameter, selected[name], rtol=0, atol=0)
    ring = DinoFrameRing(model.encoder)
    with torch.inference_mode():
        ring.push(frame, 0)
        local, pooled = ring.features(padding=True)
    torch.testing.assert_close(local, expected)
    torch.testing.assert_close(pooled, expected.mean(1))


def test_frozen_runtime_encoder_is_locked_without_changing_tensors():
    model = backbone()
    saved = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    configure_inference_encoder(model, None)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)


def test_actual_numpy_event_boundaries_serialize_without_changing_values():
    import json
    import numpy as np
    from ipad_jepa.runtime_state import event_metrics
    labels = np.array([0, 1, 1, 0, -1, 1, 0])
    frames = np.arange(len(labels))
    events = event_metrics(labels, frames, np.array([0, 0, 1, 0, 0, 1, 0], dtype=bool),
                          frames / 30. + .01, 30.)
    assert isinstance(events['events'][0]['onset_uncertain'], np.generic)
    saved = json.loads(json.dumps(events, default=json_scalar))
    assert saved == events
    assert saved['events'][0]['onset_uncertain'] is False
    assert saved['events'][1]['onset_uncertain'] is True
    assert saved['events'][1]['delay_seconds'] is None
    with pytest.raises(TypeError, match='Unsupported runtime JSON'):
        json_scalar(object())
