"""Plot fixed-encoder history sensitivity and paired differences on one shared mask."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(rows, **keys):
    matches = [r for r in rows if all(r[k] == value for k, value in keys.items())]
    if len(matches) != 1:
        raise ValueError("Missing or duplicate history result")
    return matches[0]


def save(fig, path):
    fig.text(.015, -.05, "Shared normal calibration and evaluated targets: t=45..N-8. Encoder, phase head and memory fixed.\nSource: history/macro_summary.json; three-seed means, 95% whole-video bootstrap CI. H5 differs from the primary t=19..N-8 protocol.", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight", facecolor="white")
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/stage05/ablations/history"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/stage05"))
    args = parser.parse_args()
    proof = json.loads((args.root / "validation.json").read_text())
    if (proof["status"] != "passed" or proof["thresholds_checked"] != 144 or
        proof["source_conditions_replayed"] != 48 or
        proof["macro_summary_sha256"] != digest(args.root / "macro_summary.json")):
        raise ValueError("Verified complete history matrix required")
    summary = json.loads((args.root / "macro_summary.json").read_text())
    models = ["dinov3-l", "vjepa21-l"]
    names = {"dinov3-l": "DINOv3-L", "vjepa21-l": "V-JEPA 2.1-L"}
    histories = [5, 15, 31]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout="constrained")
    for row_index, model in enumerate(models):
        for column, metric in enumerate(["auroc", "ap"]):
            axis = axes[row_index, column]
            for index, mode in enumerate(["offline", "online"]):
                rows = [choose(summary["results"], backbone=model, mode=mode, history=h) for h in histories]
                if any(r["seeds"] != 3 or r["devices"] != 4 for r in rows):
                    raise ValueError("Expected full three-seed Macro4")
                positions = np.array(histories) + (index - .5) * .6
                means = np.array([r[metric + "_mean"] * 100 for r in rows])
                low = np.array([r[metric + "_ci_low"] * 100 for r in rows])
                high = np.array([r[metric + "_ci_high"] * 100 for r in rows])
                color = ["#245B86", "#C17A1A"][index]
                axis.plot(positions, means, color=color, marker=["s", "o"][index],
                          linestyle=["--", "-"][index], label=mode)
                axis.vlines(positions, low, high, color=color, linewidth=1.2)
                axis.hlines(low, positions - .5, positions + .5, color=color)
                axis.hlines(high, positions - .5, positions + .5, color=color)
            axis.set(xlabel="Phase history length (frames)", xticks=histories, xlim=(2, 34),
                     ylabel=metric.upper() + " (%)", ylim=(0, 100), title=names[model])
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Frozen encoders / P3 history sensitivity / equal-weight four-device accuracy", fontsize=13)
    save(fig, args.out / "history_macro4")

    pairs = [(m, mode, h) for m in models for mode in ["offline", "online"] for h in [15, 31]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True, layout="constrained")
    labels = [f"{names[m]} / {mode} / H{h}-H5" for m, mode, h in pairs]
    for axis, metric in zip(axes, ["auroc", "ap"]):
        axis.axvline(0, color="#777777", linestyle=":", linewidth=1)
        for index, (model, mode, history) in enumerate(pairs):
            row = choose(summary["paired_deltas"], backbone=model, mode=mode, comparison=f"H{history}_minus_H5")
            value, low, high = [row[metric + suffix] * 100 for suffix in ["_delta", "_delta_ci_low", "_delta_ci_high"]]
            color = "#245B86" if model == "dinov3-l" else "#C17A1A"
            axis.scatter(value, index, color=color, marker="s" if mode == "offline" else "o", zorder=3)
            axis.hlines(index, low, high, color=color, linewidth=1.5)
            axis.vlines([low, high], index - .08, index + .08, color=color)
        axis.set(xlabel=metric.upper() + " difference (pp)", yticks=np.arange(len(pairs)),
                 yticklabels=labels, ylim=(len(pairs) - .5, -.5))
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Longer history minus H5 / paired whole-video 95% CI / same scored frames", fontsize=13)
    save(fig, args.out / "history_paired_differences")


if __name__ == "__main__":
    main()
