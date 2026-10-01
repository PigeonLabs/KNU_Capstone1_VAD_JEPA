"""Causal DINO frame reuse; each frame is encoded once, retaining only one clip."""
from collections import deque
import torch


class DinoFrameRing:
    def __init__(self, encoder, frames=16):
        if frames < 2 or frames % 2:
            raise ValueError('An even clip length >=2 is required')
        self.encoder, self.frames = encoder, frames
        self.tokens = deque(maxlen=frames)
        self.next_frame = 0

    def reset(self):
        self.tokens.clear()
        self.next_frame = 0

    def push(self, frame, index):
        if index != self.next_frame or frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError('Frames must arrive once in sequential order as [3,H,W]')
        if self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError('Frame reuse requires a frozen evaluation encoder')
        # ClipDataset.frame keeps HWC storage strides after CHW permutation.
        # Full-clip stacking makes the encoder input NCHW contiguous; match it.
        dense = self.encoder.forward_features(frame[None].contiguous())['x_norm_patchtokens'][0]
        if dense.ndim != 2 or not dense.isfinite().all():
            raise ValueError('Invalid dense frame features')
        self.tokens.append(dense)
        self.next_frame += 1

    def features(self, padding=False):
        if not self.tokens or (len(self.tokens)<self.frames and not padding):
            raise ValueError('Not enough actual past frames')
        values = list(self.tokens)
        values = [values[0]]*(self.frames-len(values))+values
        dense = torch.stack(values)
        return dense[-2:].mean(0)[None], dense.mean((0,1))[None]
