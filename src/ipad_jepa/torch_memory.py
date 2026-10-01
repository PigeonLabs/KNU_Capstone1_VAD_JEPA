"""FP32 batched prototype search, numerically matched to the NumPy reference."""
import math
import numpy as np
import torch
from torch import nn
from ipad_jepa.memory import PrototypeMemory


class TorchMemory(nn.Module):
    def __init__(self, memory: PrototypeMemory):
        super().__init__()
        self.bins=memory.bins
        self.dimensions=memory.dimensions
        self.temperature=memory.temperature
        self.register_buffer('mean',torch.from_numpy(memory.pca.mean_.astype(np.float32)))
        self.register_buffer('components',torch.from_numpy(memory.pca.components_.astype(np.float32)))
        self.register_buffer('prototypes',torch.from_numpy(memory.prototypes.astype(np.float32)))

    def transform(self, features):
        x=(features.float()-self.mean)@self.components.T
        return x/torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp_min(1e-12)

    def nearest(self, features, phase, k=5, global_search=False):
        if features.ndim!=3 or phase.shape!=(len(features),):
            raise ValueError('Expected B×patches×features and B phases')
        if not features.isfinite().all() or not phase.isfinite().all() or torch.any((phase<0)|(phase>=1)):
            raise ValueError('Invalid query features or phases')
        z=self.transform(features)
        if global_search:
            bank=self.prototypes.reshape(1,-1,self.dimensions).expand(len(z),-1,-1)
        else:
            b=(phase*self.bins).long()
            # sorted unique phase-bin IDs exactly match the NumPy reference.
            ids=torch.stack([(b-1)%self.bins,b,(b+1)%self.bins],dim=1).sort(dim=1).values
            if self.bins<3:
                ids=torch.stack([torch.unique(i,sorted=True) for i in ids])
            bank=self.prototypes[ids].reshape(len(z),-1,self.dimensions)
        if k<1 or k>bank.shape[1]:
            raise ValueError('Invalid neighbour count')
        distances=(z.square().sum(-1,keepdim=True)+bank.square().sum(-1)[:,None,:]-2*torch.bmm(z,bank.transpose(1,2))).clamp_min(0)
        distances,ids=distances.topk(k,dim=-1,largest=False,sorted=True)
        neighbours=bank[torch.arange(len(z),device=z.device)[:,None,None],ids]
        return z,neighbours,distances

    def patch_scores(self,features,phase,k=5,global_search=False):
        z,neighbours,distances=self.nearest(features,phase,k,global_search)
        if k==1:
            return distances[...,0]
        weights=(-distances/max(self.temperature,1e-6)).softmax(-1)
        projected=(neighbours*weights[...,None]).sum(-2)
        return (z-projected).square().sum(-1)

    def frame_scores(self,features,phase,k=5,global_search=False,top_fraction=.05):
        if not 0<top_fraction<=1:
            raise ValueError('Invalid top-patch fraction')
        scores=self.patch_scores(features,phase,k,global_search)
        count=max(1,math.ceil(scores.shape[1]*top_fraction))
        return scores.topk(count,dim=1).values.mean(1)
