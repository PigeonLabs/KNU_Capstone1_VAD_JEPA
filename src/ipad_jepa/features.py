"""Frozen-backbone clip extraction with versioned, local-only feature caches."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from ipad_jepa.backbones import Backbone
from ipad_jepa.temporal import clip_indices
from ipad_jepa.audit import inspect_frames


def file_hash(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b""):
            h.update(chunk)
    return h.hexdigest()


class ClipDataset(Dataset):
    def __init__(self, folder: Path, targets: np.ndarray, mode: str, size: int=384,
                 frames: int=16, padding: bool=False, cache_frames: int=64):
        self.paths=sorted(folder.glob("*.jpg"),key=lambda x:int(x.stem))
        self.targets=targets
        self.mode,self.size,self.frames,self.padding=mode,size,frames,padding
        self.cache=OrderedDict()
        self.cache_frames=cache_frames
        self.mean=torch.tensor([.485,.456,.406])[:,None,None]
        self.std=torch.tensor([.229,.224,.225])[:,None,None]

    def __len__(self):
        return len(self.targets)

    def frame(self, index):
        if index not in self.cache:
            with Image.open(self.paths[index]) as image:
                pixels=np.array(image.convert("RGB").resize((self.size,self.size),Image.Resampling.BILINEAR))
            value=torch.from_numpy(pixels).permute(2,0,1).float()/255
            self.cache[index]=(value-self.mean)/self.std
            if len(self.cache)>self.cache_frames:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self,index):
        t=int(self.targets[index])
        ids=clip_indices(t,len(self.paths),self.mode,self.frames,self.padding)
        return torch.stack([self.frame(int(i)) for i in ids],dim=1),t


def anchors(length: int, mode: str, split: str, frames: int=16, fit_stride: int=4):
    if mode=="offline":
        start,stop=frames//2,length-(frames-frames//2-1)
    else:
        start,stop=(0 if split=="fit" else frames-1),length
    stride=fit_stride if split=="fit" else 1
    return np.arange(start,max(start,stop),stride,dtype=np.int64)


def extract_sequence(model, row, root: Path, cache: Path, fingerprint: dict,
                     batch_size: int, workers: int, fit_stride: int=4):
    split=row.get("split","test")
    current = inspect_frames(root/row["relative_directory"])
    if current["frames_content_sha256"] != row.get("frames_content_sha256"):
        raise ValueError("Frame contents changed or content hashes missing; rerun data audit")
    targets=anchors(row["frames"],model.mode,split,fit_stride=fit_stride)
    spec={**fingerprint,"sequence":row["sequence"],"device":row["device"],"partition":row["partition"],
          "frame_names_sha256":row["names_sha256"],"frame_content_sha256":row["frames_content_sha256"],
          "split":split,"fit_stride":fit_stride,
          "rows":len(targets),"frames":row["frames"],"padding":model.mode=="online" and split=="fit"}
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    folder=cache / row["device"] / row["partition"] / row["sequence"]
    metadata=folder/"meta.json"
    if metadata.exists():
        previous=json.loads(metadata.read_text())
        if previous.get("fingerprint")!=key:
            raise ValueError(f"Stale cache: {folder}; use a separate cache namespace")
        for name,shape in [("patch",(len(targets),576,1024)),("global",(len(targets),1024))]:
            data=np.load(folder/f"{name}.npy",mmap_mode="r",allow_pickle=False)
            if data.shape!=shape or data.dtype!=np.float16:
                raise ValueError(f"Invalid cache array: {folder}/{name}.npy")
        if not np.array_equal(np.load(folder/"targets.npy",allow_pickle=False),targets):
            raise ValueError(f"Invalid target IDs: {folder}")
        print(f"cache verified: {model.name}/{model.mode}/{row['device']}/{row['sequence']}",flush=True)
        return previous
    if len(targets)==0:
        raise ValueError("No valid clips")
    folder.mkdir(parents=True,exist_ok=True)
    dataset=ClipDataset(root/row["relative_directory"],targets,model.mode,padding=spec["padding"])
    if len(dataset.paths)!=row["frames"]:
        raise ValueError("Manifest frame count changed; rerun audit")
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    local=np.lib.format.open_memmap(folder/"patch.tmp.npy",mode="w+",dtype=np.float16,shape=(len(targets),576,1024))
    pooled=np.lib.format.open_memmap(folder/"global.tmp.npy",mode="w+",dtype=np.float16,shape=(len(targets),1024))
    cursor=0; start=time.perf_counter()
    with torch.inference_mode():
        for video,ids in loader:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                patches,global_features=model(video.cuda(non_blocking=True))
            if not patches.isfinite().all() or not global_features.isfinite().all():
                raise ValueError("Nonfinite encoder features")
            n=len(ids)
            local[cursor:cursor+n]=patches.float().cpu().numpy().astype(np.float16)
            pooled[cursor:cursor+n]=global_features.float().cpu().numpy().astype(np.float16)
            cursor+=n
    if cursor!=len(targets):
        raise ValueError("Incomplete feature extraction")
    local.flush(); pooled.flush(); del local,pooled
    (folder/"patch.tmp.npy").replace(folder/"patch.npy")
    (folder/"global.tmp.npy").replace(folder/"global.npy")
    np.save(folder/"targets.npy",targets,allow_pickle=False)
    result={"fingerprint":key,"spec":spec,"status":"complete","extraction_seconds":time.perf_counter()-start,
            "note":"Cache generation wall time, not a real-time throughput benchmark"}
    metadata.write_text(json.dumps(result,indent=2)+"\n")
    print(f"extracted: {model.name}/{model.mode}/{row['device']}/{row['sequence']} clips={cursor} seconds={result['extraction_seconds']:.2f}",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,default=Path("results/stage00/manifest.json"))
    p.add_argument("--model",choices=["vjepa21-l","dinov3-l"],required=True)
    p.add_argument("--mode",choices=["offline","online"],required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--devices",nargs="+",default=["R01","R02","R03","R04"])
    p.add_argument("--splits",nargs="+",default=["fit","calibration"],choices=["fit","calibration","diagnostic","test"])
    p.add_argument("--cache",type=Path,default=Path("artifacts/features"))
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--fit-stride",type=int,default=4)
    p.add_argument("--max-sequences",type=int,help="Pilot only; output does not represent full benchmark")
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; GPU host permission required")
    if args.fit_stride<1 or args.batch_size<1:
        raise ValueError("Positive batch size and fit stride required")
    rows=json.loads(args.manifest.read_text())["sequences"]
    rows=[r for r in rows if r["device"] in args.devices and r.get("split","test") in args.splits]
    if args.max_sequences:
        rows=rows[:args.max_sequences]
    if not rows:
        raise ValueError("No matching sequences")
    fingerprint={"schema_version":1,"backbone":args.model,"mode":args.mode,
        "adapter_sha256":file_hash(Path(__file__).with_name("backbones.py")),
        "reader_sha256":file_hash(Path(__file__)),
        "upstream_commit":subprocess.check_output(["git","-C",str(args.upstream),"rev-parse","HEAD"]).decode().strip(),
        "weights_sha256":file_hash(args.weights),"image_size":384,"clip_frames":16,
        "preprocessing":"RGB full-frame PIL bilinear resize; ImageNet mean/std",
        "feature_dtype":"float16 from BF16 inference","torch":torch.__version__}
    model=Backbone(args.model,args.upstream,args.weights,args.mode).cuda().eval()
    cache=args.cache/args.model/args.mode
    for row in rows:
        extract_sequence(model,row,args.data_root,cache,fingerprint,args.batch_size,args.workers,args.fit_stride)


if __name__=="__main__":
    main()
