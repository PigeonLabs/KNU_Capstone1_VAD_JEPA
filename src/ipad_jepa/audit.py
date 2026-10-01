"""Read-only dataset audit; publish metadata, never original images or arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

DEVICES = ("R01", "R02", "R03", "R04")
CALIBRATION_COUNTS = {"R01": 5, "R02": 5, "R03": 4, "R04": 4}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_cycles(ids: list[str], seed: int = 42, calibration_count: int | None = None) -> dict[str, list[str]]:
    """Independent, identical-seed shuffling per device; never split frames."""
    shuffled = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    nfit = math.floor(0.7 * len(ids))
    ncal = calibration_count if calibration_count is not None else math.ceil(0.15 * len(ids))
    if not nfit or len(ids) - nfit - ncal < 1:
        raise ValueError("Insufficient cycles for disjoint fit/calibration/diagnostic splits")
    return {"fit": sorted(shuffled[:nfit]), "calibration": sorted(shuffled[nfit:nfit+ncal]),
            "diagnostic": sorted(shuffled[nfit+ncal:])}


def inspect_frames(folder: Path) -> dict:
    frames = list(folder.glob("*.jpg"))
    if not frames:
        raise ValueError(f"No JPG frames: {folder}")
    if any(not p.stem.isdecimal() for p in frames):
        raise ValueError(f"Non-numeric frame name: {folder}")
    frames.sort(key=lambda p: int(p.stem))
    ids = [int(p.stem) for p in frames]
    if ids != list(range(len(ids))):
        raise ValueError(f"Frames must be unique and contiguous from zero: {folder}")
    name_hash = hashlib.sha256("\n".join(p.name for p in frames).encode()).hexdigest()
    sizes = []
    for i in sorted({0, len(frames)//2, len(frames)-1}):
        with Image.open(frames[i]) as image:
            sizes.append(list(image.size))
            image.verify()
    return {"frames": len(frames), "names_sha256": name_hash,
            "first_frame": frames[0].name, "last_frame": frames[-1].name,
            "sampled_image_sizes": sizes, "image_check": "first_middle_last_only"}


def audit(root: Path, seed: int = 42) -> tuple[dict, dict, dict]:
    root = root.resolve()
    manifest, splits, mismatches = [], {}, []
    summaries = []
    for device in DEVICES:
        fit_folder = root / device / "training" / "frames"
        train_ids = sorted(p.name for p in fit_folder.iterdir() if p.is_dir())
        splits[device] = split_cycles(train_ids, seed, CALIBRATION_COUNTS[device])
        totals = {"device": device, "training_sequences": len(train_ids), "training_frames": 0,
                  "testing_sequences": 0, "testing_frames": 0}
        for partition in ("training", "testing"):
            folders = sorted((root / device / partition / "frames").iterdir())
            folders = [p for p in folders if p.is_dir()]
            label_ids = set()
            if partition == "testing":
                labels = list((root / device / "test_label").glob("*.npy"))
                label_ids = {int(p.stem) for p in labels}
                if len(label_ids) != len(labels) or label_ids != {int(p.name) for p in folders}:
                    raise ValueError(f"Missing, duplicate or orphan test labels in {device}")
                totals["testing_sequences"] = len(folders)
            for folder in folders:
                row = {"device": device, "sequence": folder.name, "partition": partition,
                       "relative_directory": folder.relative_to(root).as_posix(), **inspect_frames(folder)}
                totals[f"{partition}_frames"] += row["frames"]
                if partition == "training":
                    row["split"] = next(k for k, values in splits[device].items() if folder.name in values)
                else:
                    label_path = next(p for p in labels if int(p.stem) == int(folder.name))
                    values = np.load(label_path, allow_pickle=False)
                    if values.ndim != 1 or not np.all(np.isin(values, [0, 1])):
                        raise ValueError(f"Expected 1D binary labels: {label_path}")
                    row.update(label_file=label_path.relative_to(root).as_posix(),
                               label_sha256=digest(label_path), label_count=int(values.size),
                               anomaly_labels=int(values.sum()),
                               label_transitions=(np.flatnonzero(np.diff(values) != 0) + 1).tolist(),
                               alignment_status="matched" if values.size == row["frames"] else "review_required")
                    if row["alignment_status"] != "matched":
                        mismatches.append({"device": device, "sequence": folder.name,
                                           "frames": row["frames"], "labels": int(values.size),
                                           "transitions": row["label_transitions"]})
                manifest.append(row)
        summaries.append(totals)
    report = {"schema_version": 1, "source": "Local IPAD_dataset/IPAD_dataset; R01-R04 only",
              "split_seed": seed, "devices": summaries, "mismatches": mismatches,
              "training_sequences": sum(r["training_sequences"] for r in summaries),
              "testing_sequences": sum(r["testing_sequences"] for r in summaries),
              "training_frames": sum(r["training_frames"] for r in summaries),
              "testing_frames": sum(r["testing_frames"] for r in summaries),
              "label_alignment_complete": not mismatches,
              "image_decode_scope": "3 sampled frames per sequence; not a full corruption scan",
              "cycle_assumption": "One training directory is one cycle, following IPAD relative-position supervision",
              "normality_assumption": "Training frames treated as normal by the dataset protocol"}
    return report, {"schema_version": 1, "sequences": manifest}, {"schema_version": 1, "seed": seed, "devices": splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/stage00"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report, manifest, splits = audit(args.data_root, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, obj in [("audit", report), ("manifest", manifest), ("splits", splits)]:
        (args.out / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "devices"}, indent=2))


if __name__ == "__main__":
    main()
