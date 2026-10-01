"""Plot actual normal-only phase-head training curves, not anomaly performance."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    info=json.loads((args.results/"phase_training.json").read_text())
    with (args.results/"phase_training.csv").open() as f:
        rows=list(csv.DictReader(f))
    epoch=np.array([int(r["epoch"]) for r in rows])
    train=np.array([float(r["train_ce"]) for r in rows])
    val=np.array([float(r["normal_calibration_ce"]) for r in rows])
    mae=np.array([float(r["normal_calibration_circular_mae"])*100 for r in rows])
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    ax[0].plot(epoch,train,label="Normal fit",color="#245B86",marker="o",markersize=3)
    ax[0].plot(epoch,val,label="Normal calibration",color="#C17A1A",linestyle="--",marker="s",markersize=3)
    selected=info["selected_epoch"]-1
    ax[0].scatter(epoch[selected],val[selected],marker="*",s=180,color="#202020",zorder=5,label="Selected by calibration CE")
    ax[0].set(ylabel="Cross-entropy (200 relative-phase classes)",ylim=(0,max(train.max(),val.max())*1.12),title="Phase-head objective")
    ax[0].legend(frameon=False,fontsize=9)
    ax[1].plot(epoch,mae,color="#245B86",marker="o",markersize=3)
    ax[1].set(ylabel="Circular MAE (% of relative cycle)",ylim=(0,max(mae.max()*1.15,1)),title="Normal calibration relative-phase error")
    for a in ax:
        a.set(xlabel="Epoch",xticks=[1,5,10,15,20] if len(epoch)==20 else epoch)
        a.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"{info['backbone']} / {info['device']} / {info['mode']} / seed {info['seed']}",fontsize=14)
    fig.text(.02,-.04,"Source: phase_training.csv. Normal data only; no anomaly labels. This is not an AUROC/FPS benchmark.",fontsize=9)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in (".png",".svg"):
        fig.savefig(args.out.with_suffix(suffix),dpi=180,bbox_inches="tight",facecolor="white")


if __name__=="__main__":
    main()
