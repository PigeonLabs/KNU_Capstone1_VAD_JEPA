"""Visualize complete frozen Macro4 modes and their paired accuracy differences."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/stage02/macro_summary.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/stage02/macro4_mode_comparison"))
    args = parser.parse_args()
    info = json.loads(args.summary.read_text())
    labels = [(model, variant) for model in ["dinov3-l", "vjepa21-l"] for variant in ["P0", "P3"]]
    names = {"dinov3-l": "DINOv3-L", "vjepa21-l": "V-JEPA 2.1-L"}
    rows = {}
    for model, variant in labels:
        for mode in ["offline", "online"]:
            matches = [r for r in info["results"] if (r["backbone"], r["mode"], r["variant"]) == (model, mode, variant)]
            if len(matches) != 1 or matches[0]["seeds"] != 3:
                raise ValueError("Requires both backbones, modes, four devices and three seeds")
            row = matches[0]
            if set(row["device_counts"]) != {"R01", "R02", "R03", "R04"}:
                raise ValueError("Incomplete Macro4 device inventory")
            rows[model, variant, mode] = row
    differences = []
    for model, variant in labels:
        matches = [r for r in info["paired_deltas"] if (r["backbone"], r["mode"], r["comparison"]) == (model, "online_minus_offline", variant + "_mode")]
        if len(matches) != 1:
            raise ValueError("Missing paired mode difference")
        differences.append(matches[0])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, metric in enumerate(["auroc", "ap"]):
        upper, lower = axes[:, column]
        for index, mode in enumerate(["offline", "online"]):
            chosen = [rows[model, variant, mode] for model, variant in labels]
            positions = x + (index - .5) * .32
            upper.bar(positions, [r[metric + "_mean"] * 100 for r in chosen], width=.28,
                      color=["#245B86", "#C17A1A"][index], label=mode)
            low = np.array([r[metric + "_ci_low"] * 100 for r in chosen])
            high = np.array([r[metric + "_ci_high"] * 100 for r in chosen])
            upper.vlines(positions, low, high, color="#222222", linewidth=1.2)
            upper.hlines(low, positions - .04, positions + .04, color="#222222")
            upper.hlines(high, positions - .04, positions + .04, color="#222222")
        low = np.array([r[metric + "_delta_ci_low"] * 100 for r in differences])
        high = np.array([r[metric + "_delta_ci_high"] * 100 for r in differences])
        lower.axhline(0, color="#777777", linestyle=":", linewidth=1)
        lower.scatter(x, [r[metric + "_delta"] * 100 for r in differences], color="#245B86", zorder=3)
        lower.vlines(x, low, high, color="#245B86", linewidth=1.5)
        lower.hlines(low, x - .06, x + .06, color="#245B86")
        lower.hlines(high, x - .06, x + .06, color="#245B86")
        upper.set(ylabel=metric.upper() + " (%)", ylim=(0, 100), title="Equal-weight four-device accuracy")
        lower.set(ylabel=metric.upper() + " difference (pp)", title="Online minus offline / paired 95% CI")
        for axis in [upper, lower]:
            axis.set(xticks=x, xticklabels=[f"{names[model]}\n{variant}" for model, variant in labels])
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Frozen encoders / Macro4 / three-seed means; 95% whole-video bootstrap CI", fontsize=13)
    fig.text(.015, -.035, "Source: macro_summary.json. Same evaluated frames; videos resampled within each device, conditions paired.\nSeparate mode-specific normal fitting and heads; accuracy comparison, not a real-time benchmark.", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in [".png", ".svg"]:
        fig.savefig(args.out.with_suffix(extension), dpi=180, bbox_inches="tight", facecolor="white")
    svg = args.out.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
