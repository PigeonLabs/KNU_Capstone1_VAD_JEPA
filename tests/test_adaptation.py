import pytest
import torch
from torch import nn
from ipad_jepa.adaptation import (CompatibleQVLoRA, install_adapters,
    adapter_training_mode, adapter_state, load_adapter_state,
    dense_teacher_loss, accumulation_divisor)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module() for _ in range(6)])
        for block in self.blocks:
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(8, 24)


def test_official_projection_interface_and_frozen_prefix_modes():
    model = nn.Module()
    model.encoder = ToyEncoder()
    assert install_adapters(model, rank=2) == 4 * 4 * 8 * 2
    adapter_training_mode(model, True)
    assert not model.training and not model.encoder.training
    assert not model.encoder.blocks[0].training
    for block in model.encoder.blocks[-4:]:
        qkv = block.attn.qkv
        assert qkv.in_features == 8 and qkv.out_features == 24
        assert qkv.training and qkv.dropout.training and not qkv.base.training
        x = torch.randn(3, 8)
        torch.testing.assert_close(qkv(x), qkv.base(x))
    assert all(not p.requires_grad for p in model.encoder.blocks[:2].parameters())
    saved = adapter_state(model)
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(1)
    load_adapter_state(model, saved)
    for name, value in adapter_state(model).items():
        torch.testing.assert_close(value, saved[name])
    with pytest.raises(ValueError):
        load_adapter_state(model, {})
    with pytest.raises(ValueError):
        install_adapters(model)


def test_teacher_loss_scale_and_no_teacher_gradient():
    student = torch.tensor([[[3., 0.], [0., 4.]]], requires_grad=True)
    teacher = torch.tensor([[[0., 7.], [0., 8.]]], requires_grad=True)
    loss = dense_teacher_loss(student, teacher)
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert student.grad is not None and teacher.grad is None


def test_partial_accumulation_matches_means_without_dropping_last_clip():
    parameter = nn.Parameter(torch.tensor(2.))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    values = torch.arange(1., 11.)
    observed = []
    for i, value in enumerate(values):
        (parameter * value / accumulation_divisor(i, len(values), 8)).backward()
        if (i + 1) % 8 == 0 or i + 1 == len(values):
            observed.append(parameter.grad.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    assert observed == pytest.approx([4.5, 9.5])


def test_anomaly_or_diagnostic_teacher_is_rejected_before_reading_raw_frames(tmp_path):
    from types import SimpleNamespace
    from ipad_jepa.train_lora import NormalTeacherClips
    for row in [{"partition": "testing", "split": "test"},
                {"partition": "training", "split": "diagnostic"}]:
        with pytest.raises(ValueError, match="only normal"):
            NormalTeacherClips([SimpleNamespace(row=row)], tmp_path, "fit")


def test_pilot_or_incomplete_adapter_cannot_supply_inference_checkpoint(tmp_path):
    import json
    from ipad_jepa.adapted_features import selected_protocol
    for status, epochs in [("pilot_only", 0), ("running", 19), ("complete_training", 1)]:
        (tmp_path / "lora_training.json").write_text(json.dumps({
            "status": status, "epochs": epochs, "completed_epochs": epochs, "max_steps": None}))
        with pytest.raises(ValueError, match="completed 20-epoch"):
            selected_protocol(tmp_path)
