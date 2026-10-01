"""Official encoder-only adapters with strict local checkpoint loading."""
from __future__ import annotations
import importlib
from pathlib import Path
import sys
import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self, name: str, upstream: Path, weights: Path, mode: str = "offline"):
        super().__init__()
        if name not in {"vjepa21-l", "dinov3-l"} or mode not in {"offline", "online"}:
            raise ValueError("Unsupported backbone/mode")
        if not weights.is_file():
            raise FileNotFoundError(weights)
        self.name, self.mode = name, mode
        sys.path.insert(0, str(upstream.resolve()))
        if name == "vjepa21-l":
            module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
            self.encoder = module.vit_large(img_size=(384,384), patch_size=16, num_frames=64,
                tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
                use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
            checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
            state = checkpoint["ema_encoder"]
            cleaned = {}
            for key, value in state.items():
                while key.startswith(("module.", "backbone.")):
                    key = key.split(".",1)[1]
                cleaned[key] = value
            self.encoder.load_state_dict(cleaned, strict=True)
            del checkpoint, state, cleaned
        else:
            module = importlib.import_module("dinov3.hub.backbones")
            self.encoder = module.dinov3_vitl16(pretrained=False)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def forward(self, video: torch.Tensor):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] % 2:
            raise ValueError("Expected [B,3,even-T,H,W]")
        b, _, t, h, w = video.shape
        if h % 16 or w % 16:
            raise ValueError("Spatial shape must be divisible by patch size 16")
        patches = (h//16)*(w//16)
        if self.name == "vjepa21-l":
            tokens = self.encoder(video, training=False)
            expected = (b, (t//2)*patches, 1024)
            if tuple(tokens.shape) != expected:
                raise ValueError(f"Expected final dense {expected}, got {tuple(tokens.shape)}")
            dense = tokens.reshape(b,t//2,patches,1024)
            anchor = t//4 if self.mode == "offline" else t//2-1
            return dense[:,anchor], dense.mean(dim=(1,2))
        frames = video.permute(0,2,1,3,4).reshape(b*t,3,h,w)
        # Special tokens are excluded by the official encoder's named patch output.
        tokens = self.encoder.forward_features(frames)["x_norm_patchtokens"]
        dense = tokens.reshape(b,t,patches,1024)
        anchor = t//2 if self.mode == "offline" else t-2
        return dense[:,anchor:anchor+2].mean(1), dense.mean(dim=(1,2))


class PhaseHead(nn.Sequential):
    def __init__(self, dimension: int = 1024):
        super().__init__(nn.Linear(dimension,256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256,200))


class QVLoRA(nn.Module):
    """Add q/v low-rank updates to a fused qkv projection; k remains untouched."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = .05):
        super().__init__()
        if base.out_features != 3*base.in_features:
            raise ValueError("Expected fused qkv linear layer")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha/rank
        self.dropout = nn.Dropout(dropout)
        d = base.in_features
        self.q_a, self.q_b = nn.Linear(d,rank,bias=False), nn.Linear(rank,d,bias=False)
        self.v_a, self.v_b = nn.Linear(d,rank,bias=False), nn.Linear(rank,d,bias=False)
        nn.init.zeros_(self.q_b.weight); nn.init.zeros_(self.v_b.weight)

    def forward(self, x):
        q, k, v = self.base(x).chunk(3,dim=-1)
        update = self.dropout(x)
        q = q + self.scale*self.q_b(self.q_a(update))
        v = v + self.scale*self.v_b(self.v_a(update))
        return torch.cat((q,k,v),dim=-1)


def apply_lora(backbone: Backbone, blocks: int = 4):
    backbone.encoder.requires_grad_(False)
    for block in backbone.encoder.blocks[-blocks:]:
        base = block.attn.qkv
        block.attn.qkv = QVLoRA(base).to(device=base.weight.device,dtype=base.weight.dtype)
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)

