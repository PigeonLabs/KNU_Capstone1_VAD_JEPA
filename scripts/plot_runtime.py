"""Export actual audited FIFO latency, paced throughput and component-cost figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_runtime import audit, digest, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Figure stem; exports PNG, SVG and source receipt")
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("../IPAD_dataset/IPAD_dataset"))
    args = parser.parse_args()
    proof = json.loads(args.audit.read_text())
    if proof != audit(args.runtime, args.manifest, args.data_root):
        raise ValueError("Actual runtime audit or its sources changed")
    report = json.loads((args.runtime / "runtime.json").read_text())
    sequences = [row["sequence"] for row in report["sequences"]]
    traces = {sequence: read(args.runtime / f"{sequence}.csv") for sequence in sequences}
    p50, p95 = [], []
    for sequence in sequences:
        values = [float(row["target_latency_ms"]) / 1000 for row in traces[sequence] if row["inference_valid"] == "1"]
        lo, hi = np.quantile(values, [.5, .95])
        p50.append(float(lo)); p95.append(float(hi))
    costs = report["stage_mean_ms"]
    blue, gold = "#245B86", "#B97812"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.7))
    x = np.arange(len(sequences))
    ax = axes[0, 0]
    ax.bar(x, [row["input_fps"] for row in report["sequences"]], color=blue)
    ax.axhline(30, color="#444444", linewidth=1.2, linestyle="--", label="30 FPS arrival")
    ax.set(title="Paced FIFO throughput by test video", xlabel="Test video", ylabel="Completed input FPS", ylim=(0, 33))
    ax.set_xticks(x, sequences); ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(x, p50, "o-", color=blue, label="p50")
    ax.plot(x, p95, "s--", color=gold, label="p95")
    ax.set(title="Arrival-to-target-output latency", xlabel="Test video", ylabel="Latency (seconds)")
    ax.set_ylim(bottom=0); ax.set_xticks(x, sequences); ax.legend(frameon=False)
    names = ["Decode / resize", "Encoder + transfer", "Feature cast", "Phase head", "PCA / search", "Score / alarm", "Total service"]
    keys = ["decode_resize_ms", "encoder_ms", "feature_cast_ms", "phase_head_ms", "projection_search_ms", "score_alarm_ms", "service_ms"]
    ax = axes[1, 0]
    values = [costs[key] for key in keys]
    bars = ax.barh(np.arange(len(keys)), values, color=[blue] * 6 + [gold])
    ax.set_yticks(np.arange(len(keys)), names); ax.invert_yaxis()
    ax.set(title="Mean service cost per eligible target", xlabel="Milliseconds", xlim=(0, max(values) * 1.2))
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    # Fixed video 03 is shared with earlier accuracy examples; no worst-video selection.
    chosen = "03" if "03" in traces else sequences[0]
    rows = traces[chosen]
    ax = axes[1, 1]
    ax.plot([int(r["arrival_frame"]) for r in rows], [float(r["queue_wait_ms"]) / 1000 for r in rows],
            color=blue, label="Queue wait")
    eligible = [r for r in rows if r["inference_valid"] == "1"]
    ax.plot([int(r["arrival_frame"]) for r in eligible], [float(r["target_latency_ms"]) / 1000 for r in eligible],
            color=gold, linestyle="--", label="Target latency")
    ax.set(title=f"FIFO trajectory: fixed video {chosen}", xlabel="Arrival frame", ylabel="Seconds")
    ax.set_ylim(bottom=0); ax.set_xlim(0, len(rows) - 1); ax.legend(frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", color="#DDDDDD", linewidth=.7, alpha=.7)
        ax.set_axisbelow(True)
    condition = " / ".join(str(report[k]) for k in ["backbone", "mode", "device", "adaptation", "precision", "implementation"])
    fig.suptitle(f"{condition} / seed {report['seed']}", fontsize=13, y=.99)
    fig.text(.02, .015,
             f"Actual {len(sequences)}-video replay; 30 FPS arrivals, no frame drops, warm file cache. "
             "p50/p95 are measured target percentiles, not confidence intervals.\n"
             "One seed and one implementation; camera capture, maximum throughput, repeat-run uncertainty and paired implementation parity are not established.",
             fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .965))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = [args.out.with_suffix(".png"), args.out.with_suffix(".svg")]
    for path in files:
        fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    receipt = {"status": "exported_actual_audited_runtime_figures", "runtime_sha256": digest(args.runtime / "runtime.json"),
               "audit_sha256": digest(args.audit), "plot_code_sha256": digest(__file__), "condition": proof["condition"],
               "test_videos": len(sequences), "eligible_targets": proof["eligible_targets"],
               "per_video_latency_seconds": {sequence: {"p50": lo, "p95": hi} for sequence, lo, hi in zip(sequences, p50, p95)},
               "stage_mean_ms": costs, "fixed_trajectory_sequence": chosen,
               "files_sha256": {str(path): digest(path) for path in files},
               "scope": "Measured single-condition distributions and stage means; no repeat-run CI or speedup/parity claim."}
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "files": list(receipt["files_sha256"])}))


if __name__ == "__main__":
    main()
