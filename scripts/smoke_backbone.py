"""Validate official weights and a real IPAD clip; not a detection benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
import torch
from ipad_jepa.backbones import Backbone


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=["vjepa21-l","dinov3-l"],required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--sequence",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/setup"))
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; run with the required host GPU permission")
    paths=sorted(args.sequence.glob("*.jpg"),key=lambda x:int(x.stem))[:16]
    if len(paths)!=16:
        raise ValueError("Need 16 real frames")
    images=[]
    for path in paths:
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB").resize((384,384),Image.Resampling.BILINEAR)))
    video=torch.from_numpy(np.stack(images)).permute(3,0,1,2).float()/255
    mean=torch.tensor([.485,.456,.406])[:,None,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None,None]
    video=((video-mean)/std)[None].cuda()
    model=Backbone(args.model,args.upstream,args.weights).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        local, global_features=model(video)
    torch.cuda.synchronize()
    if tuple(local.shape)!=(1,576,1024) or tuple(global_features.shape)!=(1,1024):
        raise ValueError("Unexpected feature shape")
    if not local.isfinite().all() or not global_features.isfinite().all():
        raise ValueError("Nonfinite features")
    result={"model":args.model,"status":"strict_load_and_real_clip_passed",
        "upstream_commit":subprocess.check_output(["git","-C",str(args.upstream),"rev-parse","HEAD"]).decode().strip(),
        "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
        "bf16":torch.cuda.is_bf16_supported(),"local_shape":list(local.shape),"global_shape":list(global_features.shape),
        "first_forward_seconds":time.perf_counter()-start,"peak_vram_mib":torch.cuda.max_memory_allocated()/1024**2,
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "scope":"Single real normal clip, cold forward; NOT AUROC or throughput benchmark"}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f"{args.model}_smoke.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

