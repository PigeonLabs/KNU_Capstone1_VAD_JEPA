"""Compare real-frame causal DINO reuse against full-clip inference on the GPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ipad_jepa.backbones import Backbone
from ipad_jepa.features import ClipDataset,file_hash
from ipad_jepa.streaming import DinoFrameRing
import ipad_jepa.streaming as implementation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('third_party/dinov3'))
    p.add_argument('--out',type=Path,default=Path('results/setup/dinov3_streaming.json'))
    args=p.parse_args()
    model=Backbone('dinov3-l',args.upstream,args.weights,'online').cuda().eval()
    targets=np.array([15,16,19,31])
    data=ClipDataset(args.sequence,targets,'online')
    ring=DinoFrameRing(model.encoder)
    comparisons=[]; passed=True
    for precision in ['BF16','FP32']:
        ring.reset()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='BF16'):
            for index in range(32):
                ring.push(data.frame(index).cuda(),index)
                if index not in targets: continue
                clip,_=data[int(np.flatnonzero(targets==index)[0])]
                expected=model(clip[None].cuda())
                actual=ring.features()
                for kind,a,b in zip(['patch','global'],actual,expected):
                    a,b=a.float(),b.float()
                    difference=(a-b).abs()
                    relative=float((a-b).norm()/b.norm().clamp_min(1e-12))
                    limit=1e-3 if precision=='BF16' else 1e-5
                    comparisons.append({'target':index,'precision':precision,'feature':kind,'shape':list(a.shape),
                                        'mean_absolute_error':float(difference.mean()),'max_absolute_error':float(difference.max()),
                                        'relative_l2_error':relative,'exact_fraction':float((a==b).float().mean()),
                                        'relative_l2_limit':limit,'passed':relative<=limit})
                    passed=passed and relative<=limit
    result={'status':'real_clip_numerical_parity_passed' if passed else 'real_clip_numerical_parity_failed','model':'dinov3-l','mode':'online',
            'weights_sha256':file_hash(args.weights),'streaming_sha256':file_hash(Path(implementation.__file__)),
            'adapter_sha256':file_hash(Path(implementation.__file__).with_name('backbones.py')),
            'precision':'BF16 and FP32 inference, comparisons in FP32','relative_l2_limit':1e-3,
            'comparisons':comparisons,'frames_encoded_per_precision':32,'retained_frames_max':16,
            'note':'Diagnostic comparisons on four real normal clips; use status and per-comparison pass flags. Not an FPS, latency or anomaly detection benchmark.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not passed: raise SystemExit('Numerical gate failed; keep full-clip inference for the primary comparison')


if __name__=='__main__': main()
