"""Plot measured paired offline/online accuracy; no inferred runtime claims."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/stage02/device_summary.json"))
    parser.add_argument("--device", default="R01")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/stage02"))
    args = parser.parse_args()
    info = json.loads(args.summary.read_text())
    rows = [r for r in info["results"] if r["device"] == args.device]
    models = sorted({r["backbone"] for r in rows if r["mode"] == "online"} &
                    {r["backbone"] for r in rows if r["mode"] == "offline"})
    if not models:
        raise ValueError("No completed paired modes")
    names = {"dinov3-l": "DINOv3-L", "vjepa21-l": "V-JEPA 2.1-L"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    labels = [(model, variant) for model in models for variant in ["P0", "P3"]]
    for axis, metric in zip(axes, ["auroc", "ap"]):
        for i, mode in enumerate(["offline", "online"]):
            chosen = [next(r for r in rows if (r["backbone"], r["variant"], r["mode"]) ==
                           (model, variant, mode)) for model, variant in labels]
            x = np.arange(len(labels)) + (i - .5) * .32
            axis.bar(x, [r[metric + "_mean"] * 100 for r in chosen], width=.28,
                     color=["#245B86", "#C17A1A"][i], label=mode)
            low = [r[metric + "_ci_low"] * 100 for r in chosen]
            high = [r[metric + "_ci_high"] * 100 for r in chosen]
            axis.vlines(x, low, high, color="#222222", linewidth=1.2)
            axis.hlines(low, x - .04, x + .04, color="#222222")
            axis.hlines(high, x - .04, x + .04, color="#222222")
        axis.set(xticks=np.arange(len(labels)),
                 xticklabels=[f"{names[model]}\n{variant}" for model, variant in labels],
                 ylabel=metric.upper() + " (%)", ylim=(0, 100))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    fig.suptitle(f"{args.device} / paired target frames / 3-seed mean; 95% video-bootstrap CI")
    fig.supxlabel("Source: device_summary.json and paired score CSVs. Accuracy only; runtime not measured.", fontsize=9)
    args.out.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(args.out / f"{args.device}_mode_comparison.{extension}", dpi=160)
    svg = args.out / f"{args.device}_mode_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
