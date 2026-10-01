"""Validate cache identity and read bounded normal-only candidate samples."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ipad_jepa.features import anchors
from ipad_jepa.memory import balanced_counts

IDENTITY_KEYS=('backbone','mode','weights_sha256','upstream_commit','adapter_sha256',
               'reader_sha256','image_size','clip_frames','fit_stride','preprocessing','feature_dtype')


class FeatureSequence:
    def __init__(self,cache:Path,row:dict):
        self.row=row
        self.folder=cache/row['device']/row['partition']/row['sequence']
        self.meta=json.loads((self.folder/'meta.json').read_text())
        spec=self.meta['spec']
        fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        expected={'device':row['device'],'sequence':row['sequence'],'partition':row['partition'],
                  'frames':row['frames'],'split':row.get('split','test'),
                  'frame_content_sha256':row['frames_content_sha256'],'frame_names_sha256':row['names_sha256']}
        if self.meta.get('status')!='complete' or self.meta.get('fingerprint')!=fingerprint:
            raise ValueError('Invalid cache fingerprint/status')
        if any(spec.get(k)!=v for k,v in expected.items()):
            raise ValueError('Cache differs from audited manifest')
        self.identity={k:spec[k] for k in IDENTITY_KEYS}
        self.targets=np.load(self.folder/'targets.npy',allow_pickle=False)
        target=anchors(row['frames'],spec['mode'],spec['split'],spec['clip_frames'],spec['fit_stride'])
        if not np.array_equal(target,self.targets):
            raise ValueError('Invalid cache targets')
        self.patches=np.load(self.folder/'patch.npy',mmap_mode='r',allow_pickle=False)
        self.global_features=np.load(self.folder/'global.npy',mmap_mode='r',allow_pickle=False)
        if self.patches.shape!=(len(target),576,1024) or self.global_features.shape!=(len(target),1024):
            raise ValueError('Invalid cache dimensions')
        if self.patches.dtype!=np.float16 or self.global_features.dtype!=np.float16:
            raise ValueError('Invalid cache dtype')


def sequences(cache,rows,split):
    selected=[r for r in rows if r.get('split','test')==split and r['partition']==('testing' if split=='test' else 'training')]
    result=[FeatureSequence(cache,r) for r in selected]
    if not result or any(s.identity!=result[0].identity for s in result):
        raise ValueError('Missing or mixed feature caches')
    return result


def normal_candidates(fit,bins=16,limit=10000,seed=0):
    if any(s.row['partition']!='training' or s.row.get('split')!='fit' for s in fit):
        raise ValueError('Prototype candidates must be normal fit data only')
    rng=np.random.default_rng(seed)
    values,phases,groups=[],[],[]
    sampling=[]
    for b in range(bins):
        clips=[np.flatnonzero(np.floor(s.targets/s.row['frames']*bins).astype(int)==b) for s in fit]
        capacities=np.array([len(i)*s.patches.shape[1] for i,s in zip(clips,fit)])
        counts=balanced_counts(capacities,limit,rng)
        for video,(s,indices,count) in enumerate(zip(fit,clips,counts)):
            if not count:
                continue
            selected=rng.choice(int(capacities[video]),int(count),replace=False)
            ci=indices[selected//s.patches.shape[1]]; pi=selected%s.patches.shape[1]
            values.append(s.patches[ci,pi].astype(np.float32))
            phases.append(s.targets[ci]/s.row['frames'])
            groups.append(np.full(int(count),video,dtype=np.int64))
        sampling.append({'phase_bin':b,'tokens':int(counts.sum()),'per_video':{s.row['sequence']:int(n) for s,n in zip(fit,counts)}})
    return np.concatenate(values),np.concatenate(phases),np.concatenate(groups),sampling
