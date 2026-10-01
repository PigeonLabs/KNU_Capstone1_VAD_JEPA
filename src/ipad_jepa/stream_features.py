"""Frozen online DINO cache generation by causal per-frame reuse, after a parity gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from ipad_jepa.audit import inspect_frames
from ipad_jepa.backbones import Backbone
from ipad_jepa.features import anchors,ClipDataset,file_hash
from ipad_jepa.streaming import DinoFrameRing
import ipad_jepa.streaming as implementation
import ipad_jepa.features as reader
from ipad_jepa.cache_data import FeatureSequence


def extract(model,row,root,cache,fingerprint,fit_stride=4):
    folder=root/row['relative_directory']
    if inspect_frames(folder)['frames_content_sha256']!=row['frames_content_sha256']:
        raise ValueError('Raw frame contents differ from audit')
    split=row.get('split','test')
    targets=anchors(row['frames'],'online',split,fit_stride=fit_stride)
    spec={**fingerprint,'sequence':row['sequence'],'device':row['device'],'partition':row['partition'],
          'frame_names_sha256':row['names_sha256'],'frame_content_sha256':row['frames_content_sha256'],
          'split':split,'fit_stride':fit_stride,'rows':len(targets),'frames':row['frames'],'padding':split=='fit'}
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    output=cache/row['device']/row['partition']/row['sequence']
    if (output/'meta.json').exists():
        previous=FeatureSequence(cache,row)
        if previous.meta['fingerprint']!=key: raise ValueError('Stale streaming cache; use a new namespace')
        print(f"streaming cache verified: {row['device']}/{row['sequence']}",flush=True)
        return previous.meta
    if not len(targets): raise ValueError('No eligible online targets')
    output.mkdir(parents=True,exist_ok=True)
    data=ClipDataset(folder,targets,'online',padding=spec['padding'])
    if len(data.paths)!=row['frames']: raise ValueError('Raw frame inventory changed')
    local=np.lib.format.open_memmap(output/'patch.tmp.npy',mode='w+',dtype=np.float16,shape=(len(targets),576,1024))
    global_=np.lib.format.open_memmap(output/'global.tmp.npy',mode='w+',dtype=np.float16,shape=(len(targets),1024))
    ring=DinoFrameRing(model.encoder); cursor=0; start=time.perf_counter()
    with torch.inference_mode():
        for index in range(int(targets[-1])+1):
            frame=data.frame(index).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ring.push(frame,index)
                if index!=targets[cursor]: continue
                patch,pooled=ring.features(padding=spec['padding'])
            local[cursor]=patch[0].float().cpu().numpy().astype(np.float16)
            global_[cursor]=pooled[0].float().cpu().numpy().astype(np.float16)
            cursor+=1
    if cursor!=len(targets): raise ValueError('Incomplete streaming cache')
    local.flush(); global_.flush(); del local,global_
    (output/'patch.tmp.npy').replace(output/'patch.npy')
    (output/'global.tmp.npy').replace(output/'global.npy')
    np.save(output/'targets.npy',targets,allow_pickle=False)
    result={'status':'complete','fingerprint':key,'spec':spec,'frames_encoded':ring.next_frame,
            'extraction_seconds':time.perf_counter()-start,
            'note':'Sequential cache generation wall time, not a real-time FPS or arrival-to-alarm benchmark'}
    (output/'meta.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"streaming extracted: {row['device']}/{row['sequence']} targets={cursor} frames_encoded={ring.next_frame}",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('third_party/dinov3'))
    p.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    p.add_argument('--parity',type=Path,default=Path('results/setup/dinov3_streaming.json'))
    p.add_argument('--cache',type=Path,default=Path('artifacts/features_streaming/dinov3-l/online'))
    p.add_argument('--devices',nargs='+',default=['R01','R02','R03','R04'])
    p.add_argument('--splits',nargs='+',default=['fit','calibration','test'],choices=['fit','calibration','test','diagnostic'])
    p.add_argument('--fit-stride',type=int,default=4)
    args=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    if args.fit_stride<1: raise ValueError('Positive fit stride required')
    parity=json.loads(args.parity.read_text())
    expected={'weights_sha256':file_hash(args.weights),'streaming_sha256':file_hash(Path(implementation.__file__)),
              'adapter_sha256':file_hash(Path(implementation.__file__).with_name('backbones.py'))}
    if parity.get('status')!='real_clip_numerical_parity_passed' or any(parity.get(k)!=v for k,v in expected.items()):
        raise ValueError('Current DINO encoder/reuse implementation has not passed the real-clip parity gate')
    sources=[Path(__file__),Path(implementation.__file__),Path(reader.__file__)]
    reader_hash=hashlib.sha256(''.join(file_hash(p) for p in sources).encode()).hexdigest()
    fingerprint={'schema_version':1,'backbone':'dinov3-l','mode':'online',
                 'weights_sha256':expected['weights_sha256'],'adapter_sha256':expected['adapter_sha256'],
                 'reader_sha256':reader_hash,'image_size':384,'clip_frames':16,
                 'upstream_commit':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD']).decode().strip(),
                 'preprocessing':'RGB full-frame PIL bilinear resize; ImageNet mean/std',
                 'feature_dtype':'float16 from BF16 inference','torch':torch.__version__,
                 'implementation':'Causal single-frame DINO encoding; 16-frame ring; no future input',
                 'parity_sha256':file_hash(args.parity)}
    rows=[r for r in json.loads(args.manifest.read_text())['sequences'] if r['device'] in args.devices and r.get('split','test') in args.splits]
    if not rows: raise ValueError('No matching sequences')
    model=Backbone('dinov3-l',args.upstream,args.weights,'online').cuda().eval()
    for row in rows: extract(model,row,args.data_root,args.cache,fingerprint,args.fit_stride)


if __name__=='__main__': main()
