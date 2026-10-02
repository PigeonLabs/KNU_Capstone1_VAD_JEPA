"""Render source-verified normal candidate counts for the paired representation ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from ipad_jepa.representation import capacity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/setup/representation_capacity.json"))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/stage05/representation_capacity"))
    parser.add_argument("--proof", type=Path, default=Path("results/setup/representation_capacity_figure_sources.json"))
    args = parser.parse_args()
    sources = [args.source, args.manifest, Path("src/ipad_jepa/representation.py"), Path(__file__)]
    pinned = {str(p): digest(p) for p in sources}
    info = json.loads(args.source.read_text())
    sampling = Path(__file__).resolve().parents[1] / "src/ipad_jepa/representation.py"
    if (info["status"] != "manifest_geometry_verified" or info["manifest_sha256"] != digest(args.manifest)
            or info["sampling_code_sha256"] != digest(sampling)):
        raise ValueError("Capacity source no longer matches audited manifest/code")
    rows = json.loads(args.manifest.read_text())["sequences"]
    expected = []
    for mode in ["offline", "online"]:
        for device in ["R01", "R02", "R03", "R04"]:
            fit = [r for r in rows if r["device"] == device and r.get("split") == "fit"]
            for stride in [4, 1]:
                expected.append({"device": device, **capacity(fit, mode, stride)})
    if info["checks"] != expected:
        raise ValueError("Chart counts differ from independently recomputed manifest geometry")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.9), sharex=True, sharey=True)
    gold, blue, ink = "#BF8B26", "#326EAD", "#30343A"
    labels = ["R01 (23 fit videos)", "R02 (21)", "R03 (15)", "R04 (17)"]
    for ax, mode in zip(axes, ["offline", "online"]):
        for offset, stride, color, hatch in [(-.17, 4, gold, "//"), (.17, 1, blue, None)]:
            values = [next(r["minimum_mean_candidates"] for r in expected
                           if r["device"] == device and r["mode"] == mode and r["fit_stride"] == stride)
                      for device in ["R01", "R02", "R03", "R04"]]
            marks = ax.barh(np.arange(4) + offset, values, height=.28, color=color, hatch=hatch,
                            edgecolor=ink, linewidth=.8, zorder=3)
            for bar, value in zip(marks, values):
                ax.text(value + 10, bar.get_y() + bar.get_height() / 2, str(value), ha="left", va="center", color=ink,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .5}, zorder=5)
        ax.axvline(128, color=ink, linestyle="--", linewidth=1.2, zorder=4)
        ax.set_title(mode.capitalize(), loc="left", pad=12, fontweight="bold", color=ink)
        ax.set_yticks(range(4), labels)
        ax.set_xlim(0, 850); ax.set_xticks([0, 200, 400, 600, 800])
        ax.set_xlabel("Minimum candidates in one phase bin", labelpad=12)
        ax.xaxis.grid(True, color="#E4E5E7", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#B8BBC0")
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle("Normal-fit candidate capacity", x=.04, y=.98, ha="left", fontsize=19, fontweight="bold", color=ink)
    fig.text(.04, .919, "One spatial-mean query per observed target; 16 phase bins, 128 prototypes per bin", color=ink)
    handles = [Patch(facecolor=gold, edgecolor=ink, hatch="//", label="Original fit stride 4"),
               Patch(facecolor=blue, label="Paired dense-fit stride 1"),
               Line2D([], [], color=ink, linestyle="--", label="Required: 128")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.033, .877), frameon=False, ncol=3)
    fig.subplots_adjust(left=.18, right=.97, top=.735, bottom=.23, wspace=.11)
    fig.text(.04, .069, "Both representations use the same stride 1 fit targets. Counts are manifest geometry; no accuracy or runtime measured.", color="#555B63", fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(args.out.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    if any(digest(p) != sha for p, sha in pinned.items()):
        raise ValueError("Pinned chart source changed during rendering")
    proof = {"status": "manifest_counts_recomputed_and_rendered", "scope": "Capacity geometry only; no accuracy/CI/runtime result",
             "source_sha256": pinned,
             "figure_sha256": {str(args.out.with_suffix(ext)): digest(args.out.with_suffix(ext)) for ext in [".png", ".svg"]},
             "displayed_rows": expected}
    args.proof.parent.mkdir(parents=True, exist_ok=True)
    args.proof.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "displayed_rows"}, indent=2))


if __name__ == "__main__":
    main()
