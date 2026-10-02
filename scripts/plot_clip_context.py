"""Render the implemented 8/16-frame windows, bound to actual clip8 GPU checks."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ipad_jepa.temporal import clip_indices


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/stage05/clip_context"))
    parser.add_argument("--receipt", type=Path, default=Path("results/setup/clip_context_figure_sources.json"))
    args = parser.parse_args()
    sources = {}
    for model in ["dinov3-l", "vjepa21-l"]:
        path = Path(f"results/setup/{model}_clip8_smoke.json")
        smoke = json.loads(path.read_text())
        if smoke["status"] != "strict_load_and_real_clip8_both_modes_passed" or smoke["backbone"] != model:
            raise ValueError("Actual complete GPU clip8 smoke required")
        for field, file in [("reader_sha256", "clip8_features.py"), ("adapter_sha256", "backbones.py"),
                            ("base_reader_sha256", "features.py"), ("temporal_code_sha256", "temporal.py")]:
            source = Path("src/ipad_jepa") / file
            if smoke["spec"][field] != digest(source):
                raise ValueError("Verified clip source changed")
            sources[str(source)] = digest(source)
        if len(smoke["checks"]) != 2 or {r["mode"] for r in smoke["checks"]} != {"offline", "online"}:
            raise ValueError("Both actual clip modes required")
        for check in smoke["checks"]:
            expected = clip_indices(check["target"], 100, check["mode"], 8, False)
            if (check["input_frame_ids"] != expected.tolist() or not check["finite"]
                    or check["input_shape"] != [1, 3, 8, 384, 384]
                    or check["local_shape"] != [1, 576, 1024] or check["global_shape"] != [1, 1024]
                    or check["uses_future"] != (check["mode"] == "offline")):
                raise ValueError("GPU smoke clip bounds or outputs differ")
        sources[str(path)] = digest(path)
    rows = []
    fig, axis = plt.subplots(figsize=(12, 5.2), layout="constrained")
    for index, (frames, mode) in enumerate([(16, "offline"), (16, "online"), (8, "offline"), (8, "online")]):
        ids = clip_indices(40, 100, mode, frames, False) - 40
        local = [0, 1] if mode == "offline" else [-1, 0]
        for offset in ids:
            axis.add_patch(Rectangle((offset - .42, index - .25), .84, .5,
                                     facecolor="#C17A1A" if offset in local else "#8BBBD8", edgecolor="white"))
        axis.text(8.2, index, f"lookahead: {max(0, int(ids.max()))} frames", va="center", fontsize=10)
        rows.append({"clip_frames": frames, "mode": mode, "relative_input_ids": ids.tolist(),
                     "local_anchor_ids": local, "lookahead_frames": max(0, int(ids.max()))})
    axis.axvline(0, color="#333333", linestyle="--", linewidth=1.2)
    axis.set(yticks=range(4), yticklabels=[f"{r['clip_frames']} frames / {r['mode']}" for r in rows],
             xticks=[-15, -8, -7, -4, -1, 0, 1, 3, 7], xlim=(-16, 14.4), ylim=(3.7, -.8),
             xlabel="Frame index relative to target t (0 = target)")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.suptitle("Clip-context ablation / implemented input windows", fontsize=14)
    legend = [Rectangle((0, 0), 1, 1, color="#8BBBD8", label="Other context frames"),
              Rectangle((0, 0), 1, 1, color="#C17A1A", label="Local patch anchor (2 frames / 1 tubelet)")]
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False, ncol=2)
    fig.text(.02, -.14, "Method diagram, not accuracy or latency results. Actual 8-frame GPU input/output checks passed for both L encoders.\n"
             "Global phase features pool every block, including the local anchor.\n"
             "Head, PCA, prototypes and normal calibration are refitted; accuracy compares common t=19..N-8 targets.", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for extension in [".png", ".svg"]:
        path = args.out.with_suffix(extension)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        if extension == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        outputs[str(path)] = digest(path)
    plt.close(fig)
    receipt = {"status": "rendered_from_verified_clip8_checks_and_implemented_windows",
               "scope": "Method illustration; lookahead counts are input geometry, not measured end-to-end latency",
               "source_sha256": {**sources, str(Path(__file__)): digest(__file__)}, "windows": rows,
               "figure_sha256": outputs, "matplotlib": matplotlib.__version__, "numpy": np.__version__}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"windows": rows, "figures": outputs}, indent=2))


if __name__ == "__main__":
    main()
