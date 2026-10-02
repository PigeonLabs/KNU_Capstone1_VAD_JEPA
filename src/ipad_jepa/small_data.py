"""Validate cache identity and read bounded normal-only candidate samples."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ipad_jepa.features import anchors
from ipad_jepa.features import file_hash
from ipad_jepa.cache_data import normal_candidates

IDENTITY_KEYS=('backbone','mode','weights_sha256','upstream_commit','adapter_sha256',
               'reader_sha256','base_reader_sha256','temporal_code_sha256','image_size','clip_frames','fit_stride',
               'feature_dimension','preprocessing','feature_dtype')


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
        source = Path(__file__).parent
        if (spec['backbone'] not in {'dinov3-b', 'vjepa21-b'} or spec['mode'] != 'online'
                or spec['feature_dimension'] != 768 or spec['clip_frames'] != 16 or spec['fit_stride'] != 4
                or spec['image_size'] != 384 or spec['padding'] != (row.get('split') == 'fit')
                or spec['adapter_sha256'] != file_hash(source/'small_backbones.py')
                or spec['reader_sha256'] != file_hash(source/'small_features.py')
                or spec['base_reader_sha256'] != file_hash(source/'features.py')
                or spec['temporal_code_sha256'] != file_hash(source/'temporal.py')
                or spec['preprocessing'] != 'RGB full-frame PIL bilinear resize; ImageNet mean/std'
                or spec['feature_dtype'] != 'float16 from BF16 inference'):
            raise ValueError('B cache dimension, online protocol or source provenance differs')
        self.targets=np.load(self.folder/'targets.npy',allow_pickle=False)
        target=anchors(row['frames'],spec['mode'],spec['split'],spec['clip_frames'],spec['fit_stride'])
        if self.targets.dtype != np.int64 or spec["rows"] != len(target) or not np.array_equal(target,self.targets):
            raise ValueError('Invalid cache targets')
        self.patches=np.load(self.folder/'patch.npy',mmap_mode='r',allow_pickle=False)
        self.global_features=np.load(self.folder/'global.npy',mmap_mode='r',allow_pickle=False)
        if self.patches.shape!=(len(target),576,768) or self.global_features.shape!=(len(target),768):
            raise ValueError('Invalid cache dimensions')
        if self.patches.dtype!=np.float16 or self.global_features.dtype!=np.float16:
            raise ValueError('Invalid cache dtype')


def sequences(cache,rows,split):
    selected=[r for r in rows if r.get('split','test')==split and r['partition']==('testing' if split=='test' else 'training')]
    result=[FeatureSequence(cache,r) for r in selected]
    if not result or any(s.identity!=result[0].identity for s in result):
        raise ValueError('Missing or mixed feature caches')
    return result

