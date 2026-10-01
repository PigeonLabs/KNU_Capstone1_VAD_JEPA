"""Source-backed IPAD metadata figures for GitHub (PNG and SVG)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"fit": "#245B86", "calibration": "#C17A1A", "diagnostic": "#AC638B", "test": "#667C3D"}


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(audit: dict, manifest: dict, splits: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.titlepad": 16})
    devices = list(splits["devices"])
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    x, bottom = np.arange(len(devices)), np.zeros(len(devices))
    for key in ("fit", "calibration", "diagnostic"):
        values = np.array([len(splits["devices"][d][key]) for d in devices])
        ax.bar(x, values, bottom=bottom, width=.58, color=COLORS[key], label=key.capitalize(),
               edgecolor="white", linewidth=1)
        for i, v in enumerate(values):
            ax.text(i, bottom[i]+v/2, str(v), ha="center", va="center", color="white", weight="bold")
        bottom += values
    ax.set(xticks=x, xticklabels=devices, ylabel="Normal training sequences", ylim=(0, 42),
           title="IPAD real-device normal-only split (seed 42)")
    ax.legend(ncol=3, loc="upper center", frameon=False)
    fig.text(.02, -.03, "Source: results/stage00/splits.json. Whole sequences; no frame-level split.", fontsize=9)
    save(fig, out, "normal_split")

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for shift, field, color, label in [(-.18, "training_frames", COLORS["fit"], "Normal training"),
                                      (.18, "testing_frames", COLORS["test"], "Actual test")]:
        values = [r[field] for r in audit["devices"]]
        bars = ax.bar(x+shift, values, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, fontsize=10)
    ax.set(xticks=x, xticklabels=devices, ylabel="JPEG frames", ylim=(0, max(r["training_frames"] for r in audit["devices"])*1.3),
           title="IPAD R01-R04 frame inventory")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    fig.text(.02, -.03, "Source: results/stage00/audit.json. Counts before score-window and unknown-label masks.", fontsize=9)
    save(fig, out, "frame_inventory")

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    samples = [[r["frames"] for r in manifest["sequences"] if r["device"] == d and r["partition"] == "training"] for d in devices]
    bp = ax.boxplot(samples, tick_labels=devices, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#245B86"})
    for box in bp["boxes"]:
        box.set(facecolor="#D5E3EF", edgecolor=COLORS["fit"])
    for median in bp["medians"]:
        median.set(color=COLORS["fit"], linewidth=1.5)
    ax.set(ylabel="Frames per normal sequence", title="Normal-cycle duration varies by device")
    fig.text(.02, -.03, "Source: results/stage00/manifest.json. Box: Q1-Q3; line: median; diamond: mean; whiskers: 1.5 IQR.", fontsize=9)
    save(fig, out, "cycle_lengths")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/stage00"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/stage00"))
    args = parser.parse_args()
    plot(*(json.loads((args.results / f"{name}.json").read_text()) for name in ("audit", "manifest", "splits")), args.out)


if __name__ == "__main__":
    main()

