"""Extract online B features with explicit 768-dimensional provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ipad_jepa.audit import inspect_frames
from ipad_jepa.small_backbones import SmallBackbone
from ipad_jepa.features import ClipDataset, anchors, file_hash
from ipad_jepa.temporal import clip_indices

FRAMES = 16


def sequence_spec(row, mode, fingerprint, fit_stride=4):
    if mode != "online" or fit_stride != 4:
        raise ValueError("Invalid mode or fit stride")
    if (fingerprint.get("clip_frames") != FRAMES or fingerprint.get("mode") != mode
            or fingerprint.get("feature_dimension") != 768
            or fingerprint.get("backbone") not in {"dinov3-b", "vjepa21-b"}):
        raise ValueError("Sixteen-frame B fingerprint and actual mode must agree")
    split = row.get("split", "test")
    targets = anchors(row["frames"], mode, split, frames=FRAMES, fit_stride=fit_stride)
    if not len(targets):
        raise ValueError("No valid sixteen-frame B clips")
    spec = {**fingerprint, "sequence": row["sequence"], "device": row["device"],
            "partition": row["partition"], "split": split, "fit_stride": fit_stride,
            "frame_names_sha256": row["names_sha256"],
            "frame_content_sha256": row["frames_content_sha256"],
            "rows": len(targets), "frames": row["frames"],
            "padding": mode == "online" and split == "fit"}
    return targets, spec


def check_outputs(patches, global_features, batch):
    if tuple(patches.shape) != (batch, 576, 768) or tuple(global_features.shape) != (batch, 768):
        raise ValueError("Unexpected sixteen-frame B encoder output dimensions")
    if not bool(patches.isfinite().all()) or not bool(global_features.isfinite().all()):
        raise ValueError("Nonfinite sixteen-frame B encoder features")


def extract_sequence(model, row, root, cache, fingerprint, batch_size=4, workers=2, fit_stride=4):
    current = inspect_frames(root / row["relative_directory"])
    if (current["frames_content_sha256"] != row["frames_content_sha256"]
            or current["names_sha256"] != row["names_sha256"] or current["frames"] != row["frames"]):
        raise ValueError("Audited source frames changed")
    targets, spec = sequence_spec(row, model.mode, fingerprint, fit_stride)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    folder = cache / row["device"] / row["partition"] / row["sequence"]
    metadata = folder / "meta.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous.get("status") != "complete" or previous.get("fingerprint") != key or previous.get("spec") != spec:
            raise ValueError(f"Incomplete or stale sixteen-frame B cache: {folder}")
        for name, shape in [("patch", (len(targets), 576, 768)), ("global", (len(targets), 768))]:
            value = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if value.shape != shape or value.dtype != np.float16:
                raise ValueError(f"Invalid cached {name}: {folder}")
        if not np.array_equal(np.load(folder / "targets.npy", allow_pickle=False), targets):
            raise ValueError("Cached target IDs differ")
        print(f"small cache verified: {model.name}/{model.mode}/{row['device']}/{row['sequence']}", flush=True)
        return previous
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Partial cache exists; verify its live owner before recovery: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    dataset = ClipDataset(root / row["relative_directory"], targets, model.mode,
                          frames=FRAMES, padding=spec["padding"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    local = np.lib.format.open_memmap(folder / "patch.tmp.npy", mode="w+", dtype=np.float16,
                                     shape=(len(targets), 576, 768))
    pooled = np.lib.format.open_memmap(folder / "global.tmp.npy", mode="w+", dtype=np.float16,
                                      shape=(len(targets), 768))
    cursor = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for video, ids in loader:
            if video.shape[2] != FRAMES or not np.array_equal(ids.numpy(), targets[cursor:cursor + len(ids)]):
                raise ValueError("Clip length or extraction order changed")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patches, global_features = model(video.cuda(non_blocking=True))
            check_outputs(patches, global_features, len(ids))
            end = cursor + len(ids)
            local[cursor:end] = patches.float().cpu().numpy().astype(np.float16)
            pooled[cursor:end] = global_features.float().cpu().numpy().astype(np.float16)
            cursor = end
    if cursor != len(targets):
        raise ValueError("Incomplete extraction")
    local.flush()
    pooled.flush()
    del local, pooled
    (folder / "patch.tmp.npy").replace(folder / "patch.npy")
    (folder / "global.tmp.npy").replace(folder / "global.npy")
    np.save(folder / "targets.npy", targets, allow_pickle=False)
    result = {"fingerprint": key, "spec": spec, "status": "complete",
              "extraction_seconds": time.perf_counter() - started,
              "note": "Actual sixteen-frame B re-encoding; cache wall time is not runtime throughput"}
    temporary = folder / "meta.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(metadata)
    print(f"small extracted: {model.name}/{model.mode}/{row['device']}/{row['sequence']} clips={cursor}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--model", choices=["dinov3-b", "vjepa21-b"], required=True)
    parser.add_argument("--mode", choices=["online"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["R01", "R02", "R03", "R04"])
    parser.add_argument("--splits", nargs="+", choices=["fit", "calibration", "diagnostic", "test"],
                        default=["fit", "calibration", "test"])
    parser.add_argument("--cache", type=Path, default=Path("artifacts/features_small"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fit-stride", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if args.batch_size < 1 or args.workers < 0 or args.fit_stride < 1:
        raise ValueError("Invalid extraction parameters")
    rows = [r for r in json.loads(args.manifest.read_text())["sequences"]
            if r["device"] in args.devices and r.get("split", "test") in args.splits]
    if not rows:
        raise ValueError("No selected sequences")
    source = Path(__file__).parent
    fingerprint = {"schema_version": 1, "backbone": args.model, "mode": args.mode,
                   "adapter_sha256": file_hash(source / "small_backbones.py"),
                   "reader_sha256": file_hash(Path(__file__)),
                   "base_reader_sha256": file_hash(source / "features.py"),
                   "temporal_code_sha256": file_hash(source / "temporal.py"),
                   "upstream_commit": subprocess.check_output(
                       ["git", "-C", str(args.upstream), "rev-parse", "HEAD"]).decode().strip(),
                   "weights_sha256": file_hash(args.weights), "image_size": 384, "clip_frames": FRAMES, "feature_dimension": 768,
                   "preprocessing": "RGB full-frame PIL bilinear resize; ImageNet mean/std",
                   "feature_dtype": "float16 from BF16 inference", "torch": torch.__version__}
    torch.backends.cuda.matmul.allow_tf32 = False
    model = SmallBackbone(args.model, args.upstream, args.weights).cuda().eval()
    cache = args.cache / args.model / "online"
    for row in rows:
        extract_sequence(model, row, args.data_root, cache, fingerprint,
                         args.batch_size, args.workers, args.fit_stride)


if __name__ == "__main__":
    main()
