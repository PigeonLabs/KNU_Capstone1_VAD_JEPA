"""Q/v adaptation preserving official fused-projection interfaces and encoder eval mode."""
from __future__ import annotations
import torch
from torch import nn
from ipad_jepa.backbones import QVLoRA


class CompatibleQVLoRA(QVLoRA):
    def __init__(self, base, rank=8, alpha=16, dropout=.05):
        super().__init__(base, rank, alpha, dropout)
        # DINOv3 Attention.compute_attention reads this interface after projection.
        self.in_features = base.in_features
        self.out_features = base.out_features


def install_adapters(backbone, blocks=4, rank=8, alpha=16, dropout=.05):
    if blocks < 1 or blocks > len(backbone.encoder.blocks):
        raise ValueError("Invalid adapter block count")
    backbone.encoder.requires_grad_(False)
    for block in backbone.encoder.blocks[-blocks:]:
        base = block.attn.qkv
        if isinstance(base, QVLoRA):
            raise ValueError("Adapter already installed")
        block.attn.qkv = CompatibleQVLoRA(base, rank, alpha, dropout).to(
            device=base.weight.device, dtype=base.weight.dtype)
    adapter_training_mode(backbone, False)
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)


def adapter_training_mode(backbone, training):
    # Avoid stochastic frozen-prefix/drop-path/RoPE augmentation in the official encoders.
    backbone.eval()
    for module in backbone.modules():
        if isinstance(module, CompatibleQVLoRA):
            module.train(training)
            module.base.eval()


def adapter_state(backbone):
    return {name: p.detach().cpu().clone() for name, p in backbone.named_parameters()
            if p.requires_grad}


def load_adapter_state(backbone, state):
    parameters = {name: p for name, p in backbone.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("Adapter parameter keys differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape or not value.isfinite().all():
                raise ValueError("Invalid adapter tensor")
            parameter.copy_(value)


def dense_teacher_loss(student, teacher):
    """Mean patchwise squared L2 between unit channel vectors (not divided by D)."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("Teacher/student dense shapes differ")
    student = nn.functional.normalize(student.float(), dim=-1)
    teacher = nn.functional.normalize(teacher.detach().float(), dim=-1)
    return (student - teacher).square().sum(-1).mean()


def accumulation_divisor(index, total, accumulation):
    if not 0 <= index < total or accumulation < 1:
        raise ValueError("Invalid accumulation position")
    start = index // accumulation * accumulation
    return min(accumulation, total - start)
