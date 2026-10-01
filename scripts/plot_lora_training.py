"""Plot completed normal-only joint LoRA curves and the prespecified CE selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.results / "lora_training.json").read_text())
    receipt = json.loads((args.results / "training_export.json").read_text())
    if (meta["status"] != "complete_training" or meta["completed_epochs"] != 20 or
        meta["epochs"] != 20 or meta["max_steps"] is not None or
        receipt["training_metadata_sha256"] != digest(args.results / "lora_training.json") or
        receipt["training_curve_sha256"] != digest(args.results / "lora_training.csv")):
        raise ValueError("A verified completed full training export is required")
    with (args.results / "lora_training.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    epochs = np.array([int(r["epoch"]) for r in rows])
    values = {k: np.array([float(r[k]) for r in rows]) for k in
              ["train_ce", "train_dense_l2", "normal_calibration_ce", "normal_calibration_circular_mae"]}
    if not np.array_equal(epochs, np.arange(1, 21)) or not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Invalid completed training curve")
    selected = int(np.argmin(values["normal_calibration_ce"]))
    if selected + 1 != meta["selected_epoch"] or values["normal_calibration_ce"][selected] != meta["best_normal_calibration_ce"]:
        raise ValueError("Selected epoch differs from minimum normal calibration CE")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    axes[0].plot(epochs, values["train_ce"], label="Normal fit", color="#245B86", marker="o", markersize=3)
    axes[0].plot(epochs, values["normal_calibration_ce"], label="Normal calibration", color="#C17A1A", linestyle="--", marker="s", markersize=3)
    axes[0].scatter(epochs[selected], values["normal_calibration_ce"][selected], marker="*", s=150, color="#222222", zorder=4, label="Selected by normal CE")
    axes[0].set(ylabel="Phase cross-entropy (200 classes)", ylim=(0, max(values["train_ce"].max(), values["normal_calibration_ce"].max()) * 1.15), title="Phase objective")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].plot(epochs, values["train_dense_l2"], color="#245B86", marker="o", markersize=3)
    axes[1].set(ylabel="Normalized dense L2 to fixed teacher", ylim=(0, values["train_dense_l2"].max() * 1.15), title="Normal fit teacher objective")
    mae = values["normal_calibration_circular_mae"] * 100
    axes[2].plot(epochs, mae, color="#245B86", marker="o", markersize=3)
    axes[2].scatter(epochs[selected], mae[selected], marker="*", s=150, color="#222222", zorder=4)
    axes[2].set(ylabel="Circular MAE (% of relative cycle)", ylim=(0, mae.max() * 1.15), title="Normal calibration / not the selector")
    for axis in axes:
        axis.set(xlabel="Completed epoch", xticks=[1, 5, 10, 15, 20])
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{meta['backbone']} / {meta['mode']} / {meta['device']} / seed {meta['seed']} / completed 20 epochs", fontsize=13)
    fig.text(.015, -.06, "Source: completed lora_training.csv/json. Joint loss = phase CE + 1.0 normalized dense teacher L2; fixed teacher, normal data only.\nStars use the same epoch selected by minimum normal CE. Curves are not anomaly AUROC/AP or real-time FPS.", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        fig.savefig(args.out.with_suffix(suffix), dpi=180, bbox_inches="tight", facecolor="white")
    svg = args.out.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
