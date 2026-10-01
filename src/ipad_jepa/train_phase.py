"""Train the compact phase head on frozen normal features only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from ipad_jepa.backbones import PhaseHead
from ipad_jepa.temporal import circular_phase


def load_normal(cache: Path, rows: list[dict], split: str):
    values,labels,positions=[],[],[]
    common_spec = None
    for row in rows:
        if row.get("split")!=split or row["partition"]!="training":
            continue
        folder=cache/row["device"]/"training"/row["sequence"]
        meta=json.loads((folder/"meta.json").read_text())
        if meta["spec"]["split"]!=split or meta["status"]!="complete":
            raise ValueError("Invalid normal feature cache split/status")
        spec=meta["spec"]
        fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        if meta.get("fingerprint")!=fingerprint:
            raise ValueError("Invalid feature cache fingerprint")
        expected={"device":row["device"],"sequence":row["sequence"],
                  "partition":"training","frames":row["frames"],
                  "frame_content_sha256":row["frames_content_sha256"],
                  "frame_names_sha256":row["names_sha256"]}
        if any(spec.get(k)!=v for k,v in expected.items()):
            raise ValueError("Feature cache does not match manifest")
        identity={k:spec[k] for k in ("backbone","mode","weights_sha256","upstream_commit",
                  "adapter_sha256","reader_sha256","image_size","clip_frames","fit_stride")}
        if common_spec is not None and identity!=common_spec:
            raise ValueError("Mixed feature cache provenance")
        common_spec=identity
        targets=np.load(folder/"targets.npy",allow_pickle=False)
        value=np.load(folder/"global.npy",allow_pickle=False)
        if value.shape!=(len(targets),1024) or len(targets)!=spec["rows"]:
            raise ValueError("Misaligned frozen features")
        if not np.all(np.isfinite(value)) or np.any(targets<0) or np.any(targets>=row["frames"]) or np.any(np.diff(targets)<=0):
            raise ValueError("Invalid frozen features or target order")
        phase=targets/row["frames"]
        values.append(value.astype(np.float32))
        positions.append(phase)
        labels.append(np.floor(200*phase).astype(np.int64))
    if not values:
        raise ValueError(f"No normal {split} caches")
    return np.concatenate(values),np.concatenate(labels),np.concatenate(positions)


def train(cache: Path, rows: list[dict], out: Path, seed: int, epochs: int=20, batch: int=256):
    if epochs < 1 or batch < 1:
        raise ValueError("Positive epoch and batch counts required")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x,y,_=load_normal(cache,rows,"fit")
    vx,vy,vphi=load_normal(cache,rows,"calibration")
    identities=[]
    for r in rows:
        if r["partition"]=="training" and r.get("split") in {"fit","calibration"}:
            spec=json.loads((cache/r["device"]/"training"/r["sequence"]/"meta.json").read_text())["spec"]
            identities.append(tuple(spec[k] for k in ("backbone","mode","weights_sha256","upstream_commit","adapter_sha256","reader_sha256","image_size","clip_frames","fit_stride")))
    if len(set(identities))!=1:
        raise ValueError("Fit and calibration feature provenance differ")
    tx=torch.from_numpy(x); ty=torch.from_numpy(y)
    loader=DataLoader(TensorDataset(tx,ty),batch_size=batch,shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    model=PhaseHead().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    out.mkdir(parents=True,exist_ok=True)
    best=float("inf"); curves=[]; selected_epoch=0; start=time.perf_counter()
    for epoch in range(epochs):
        model.train(); loss_sum=0
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(bx.cuda())
                loss=torch.nn.functional.cross_entropy(logits,by.cuda())
            loss.backward(); optimizer.step()
            loss_sum+=float(loss.detach())*len(bx)
        model.eval(); validation_loss=0; probabilities=[]
        with torch.inference_mode():
            for i in range(0,len(vx),batch):
                bx=torch.from_numpy(vx[i:i+batch]).cuda()
                by=torch.from_numpy(vy[i:i+batch]).cuda()
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    logits=model(bx)
                validation_loss+=float(torch.nn.functional.cross_entropy(logits.float(),by,reduction="sum"))
                probabilities.append(logits.float().softmax(1).cpu().numpy())
        validation_loss/=len(vx)
        phase=circular_phase(np.concatenate(probabilities))
        error=np.abs((phase-vphi+.5)%1-.5)
        entry={"epoch":epoch+1,"train_ce":loss_sum/len(x),"normal_calibration_ce":validation_loss,
               "normal_calibration_circular_mae":float(error.mean())}
        if not np.all(np.isfinite(list(entry.values()))):
            raise ValueError("Nonfinite phase-head training statistics")
        curves.append(entry)
        if validation_loss<best:
            best=validation_loss; selected_epoch=epoch+1
            torch.save(model.state_dict(),out/"phase_head.pt")
        print(json.dumps(entry),flush=True)
    with (out/"phase_training.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    metadata={"status":"complete","seed":seed,"epochs":epochs,"selected_epoch":selected_epoch,
              "backbone":cache.parent.name,"mode":cache.name,"device":rows[0]["device"],
              "cache_fingerprints":[json.loads((cache/r["device"]/"training"/r["sequence"]/"meta.json").read_text())["fingerprint"] for r in rows if r.get("split") in {"fit","calibration"}],
              "selection":"Minimum normal calibration CE; no anomaly labels",
              "fit_clips":len(x),"calibration_clips":len(vx),"seconds":time.perf_counter()-start,
              "training":"Frozen backbone caches, phase head only","phase_batch_size":batch,
              "trainable_parameters":sum(p.numel() for p in model.parameters()),
              "best_normal_calibration_ce":best}
    (out/"phase_training.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True,help="artifacts/features/<backbone>/<mode>")
    p.add_argument("--device",choices=["R01","R02","R03","R04"],required=True)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--manifest",type=Path,default=Path("results/stage00/manifest.json"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=20)
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    rows=[r for r in json.loads(args.manifest.read_text())["sequences"] if r["device"]==args.device]
    print(json.dumps(train(args.cache,rows,args.out,args.seed,args.epochs),indent=2))


if __name__=="__main__":
    main()
