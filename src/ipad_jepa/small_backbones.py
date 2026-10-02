"""Official encoder-only 768-dimensional B adapters for the online size comparison."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch
from torch import nn


class SmallBackbone(nn.Module):
    def __init__(self, name: str, upstream: Path, weights: Path):
        super().__init__()
        if name not in {"dinov3-b", "vjepa21-b"}:
            raise ValueError("Only the accepted online B backbones are supported")
        if not weights.is_file():
            raise FileNotFoundError(weights)
        self.name, self.mode, self.dimension = name, "online", 768
        sys.path.insert(0, str(upstream.resolve()))
        if name == "dinov3-b":
            module = importlib.import_module("dinov3.hub.backbones")
            self.encoder = module.dinov3_vitb16(pretrained=False)
            state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
            self.encoder.load_state_dict(state, strict=True)
            del state
        else:
            module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
            self.encoder = module.vit_base(img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
                use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                img_temporal_dim_size=1, interpolate_rope=True)
            checkpoint = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
            state = checkpoint["ema_encoder"]
            cleaned = {}
            for key, value in state.items():
                while key.startswith(("module.", "backbone.")):
                    key = key.split(".", 1)[1]
                cleaned[key] = value
            self.encoder.load_state_dict(cleaned, strict=True)
            del checkpoint, state, cleaned
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def forward(self, video):
        if video.ndim != 5 or video.shape[1:] != (3, 16, 384, 384):
            raise ValueError("Expected online [B,3,16,384,384] input")
        batch, _, frames, _, _ = video.shape
        if self.name == "dinov3-b":
            inputs = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, 3, 384, 384)
            tokens = self.encoder.forward_features(inputs)["x_norm_patchtokens"]
            if tuple(tokens.shape) != (batch * frames, 576, 768):
                raise ValueError("DINOv3-B final patch output shape differs")
            dense = tokens.reshape(batch, frames, 576, 768)
            local = dense[:, -2:].mean(1)
        else:
            tokens = self.encoder(video, training=False)
            if tuple(tokens.shape) != (batch, 8 * 576, 768):
                raise ValueError("V-JEPA 2.1-B final dense output shape differs")
            dense = tokens.reshape(batch, 8, 576, 768)
            local = dense[:, -1]
        return local, dense.mean(dim=(1, 2))
