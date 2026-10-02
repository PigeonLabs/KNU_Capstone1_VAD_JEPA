"""Encode every dense normal-fit clip; retain paired pooled features and sampled patches."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ipad_jepa.audit import inspect_frames
from ipad_jepa.backbones import Backbone
from ipad_jepa.cache_data import sequences
from ipad_jepa.features import ClipDataset, file_hash
from ipad_jepa.representation import SEEDS, candidate_plan, capacity, fit_targets, local_spatial_mean


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(rows, mode, model_name, weights, upstream, root, original_cache, out, batch=4, workers=2):
    if out.exists():
        raise ValueError("Fresh compact dense-fit namespace required; no implicit recovery")
    fit_rows = [r for r in rows if r["partition"] == "training" and r.get("split") == "fit"]
    audit = capacity(fit_rows, mode, 1)
    if not audit["enough_for_128_per_bin"]:
        raise ValueError("Not enough distinct observed pooled candidates")
    fit = sequences(original_cache, rows, "fit")
    cal = sequences(original_cache, rows, "calibration")
    if fit[0].identity != cal[0].identity:
        raise ValueError("Original frozen fit/calibration identities differ")
    source = Path(__file__).parent
    identity = {"backbone": model_name, "mode": mode, "weights_sha256": file_hash(weights),
                "upstream_commit": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"]).decode().strip(),
                "adapter_sha256": file_hash(source / "backbones.py"), "image_size": 384, "clip_frames": 16,
                "preprocessing": "RGB full-frame PIL bilinear resize; ImageNet mean/std",
                "feature_dtype": "float16 from BF16 inference", "base_reader_sha256": file_hash(source / "features.py"),
                "reader_sha256": file_hash(Path(__file__)), "sampling_code_sha256": file_hash(source / "representation.py"),
                "torch": torch.__version__, "fit_stride": 1}
    # The derived fit reader is distinct. The original calibration reader is recorded
    # explicitly, rather than relabeling previously encoded arrays as new extraction.
    for key in ["backbone", "mode", "weights_sha256", "upstream_commit", "adapter_sha256", "image_size",
                "clip_frames", "preprocessing", "feature_dtype"]:
        if fit[0].identity[key] != identity[key]:
            raise ValueError(f"Dense/original encoder semantics differ: {key}")
    if fit[0].identity["reader_sha256"] != identity["base_reader_sha256"]:
        raise ValueError("Original frozen reader differs from the pinned dense-fit base reader")
    out.mkdir(parents=True)
    started = time.perf_counter()
    plans, arrays, files = {}, {}, []
    for seed in SEEDS:
        coordinates, phases, sampling = candidate_plan(fit_rows, mode, seed)
        plans[seed] = (coordinates, phases, sampling)
        for name, value in [("coordinates", coordinates), ("phases", phases)]:
            path = out / f"seed{seed}_{name}.npy"
            np.save(path, value, allow_pickle=False)
            files.append(path)
        path = out / f"seed{seed}_patch.npy"
        arrays[seed] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(coordinates), 1024))
        files.append(path)
    model = Backbone(model_name, upstream, weights, mode).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    video_proofs = []
    for video, row in enumerate(fit_rows):
        current = inspect_frames(root / row["relative_directory"])
        if any(current[k] != row[k] for k in ["frames", "names_sha256", "frames_content_sha256"]):
            raise ValueError("Normal fit raw frames no longer match the audited manifest")
        targets = fit_targets(row["frames"], mode)
        folder = out / "fit" / row["sequence"]
        folder.mkdir(parents=True)
        np.save(folder / "targets.npy", targets, allow_pickle=False)
        pooled = np.lib.format.open_memmap(folder / "global.npy", mode="w+", dtype=np.float16, shape=(len(targets), 1024))
        mean = np.lib.format.open_memmap(folder / "mean.npy", mode="w+", dtype=np.float32, shape=(len(targets), 1, 1024))
        dataset = ClipDataset(root / row["relative_directory"], targets, mode, frames=16, padding=mode == "online")
        if len(dataset.paths) != row["frames"]:
            raise ValueError("Normal fit frame count changed")
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
        selection = {}
        for seed, (coordinates, _, _) in plans.items():
            output = np.flatnonzero(coordinates[:, 0] == video)
            selection[seed] = (output, coordinates[output, 1], coordinates[output, 2])
        written = {seed: 0 for seed in SEEDS}
        cursor = 0
        with torch.inference_mode():
            for inputs, ids in loader:
                if not np.array_equal(ids.numpy(), targets[cursor:cursor + len(ids)]):
                    raise ValueError("Dense extraction target ordering changed")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    local, global_features = model(inputs.cuda(non_blocking=True))
                if tuple(local.shape) != (len(ids), 576, 1024) or tuple(global_features.shape) != (len(ids), 1024):
                    raise ValueError("Invalid dense encoder dimensions")
                if not local.isfinite().all() or not global_features.isfinite().all():
                    raise ValueError("Nonfinite dense encoder features")
                stored = local.float().cpu().numpy().astype(np.float16)
                stored_global = global_features.float().cpu().numpy().astype(np.float16)
                if not np.all(np.isfinite(stored)) or not np.all(np.isfinite(stored_global)):
                    raise ValueError("FP16 feature storage overflow")
                pooled[cursor:cursor + len(ids)] = stored_global
                mean[cursor:cursor + len(ids)] = local_spatial_mean(stored)
                for seed, (output, ci, pi) in selection.items():
                    keep = (ci >= cursor) & (ci < cursor + len(ids))
                    arrays[seed][output[keep]] = stored[ci[keep] - cursor, pi[keep]]
                    written[seed] += int(keep.sum())
                cursor += len(ids)
        if cursor != len(targets) or any(written[s] != len(selection[s][0]) for s in SEEDS):
            raise ValueError("Incomplete dense clips or candidate coordinates")
        pooled.flush(); mean.flush()
        del pooled, mean, loader, dataset
        for name in ["targets.npy", "global.npy", "mean.npy"]:
            files.append(folder / name)
        video_proofs.append({"sequence": row["sequence"], "frames": row["frames"], "clips": cursor,
                             "frame_names_sha256": row["names_sha256"], "frame_content_sha256": row["frames_content_sha256"],
                             "padding": mode == "online", "candidate_rows_written": {str(s): written[s] for s in SEEDS}})
        print(f"representation dense fit: {model_name}/{mode}/{row['device']}/{row['sequence']} clips={cursor}", flush=True)
    for array in arrays.values():
        array.flush()
    arrays.clear()
    del model, fit, cal
    gc.collect(); torch.cuda.empty_cache()
    result = {"status": "complete_dense_normal_fit", "device": fit_rows[0]["device"], "identity": identity,
              "original_cache_identity": sequences(original_cache, rows, "calibration")[0].identity,
              "capacity": audit, "fit_inventory": video_proofs, "seeds": list(SEEDS),
              "candidate_sampling": {str(s): plans[s][2] for s in SEEDS},
              "outputs_sha256": {str(p.relative_to(out)): file_hash(p) for p in files},
              "extraction_seconds": time.perf_counter() - started,
              "mean_representation": "FP32 spatial mean of FP16 target-local patches; one query per target",
              "phase_input": "Whole-context pooled feature; shared by both representations",
              "scope": "Every dense normal-fit clip encoded; bounded retained patch candidates; no accuracy or runtime benchmark"}
    write_json(out / "dense_fit.json", result)
    print(f"representation dense group complete: {model_name}/{mode}/{fit_rows[0]['device']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--model", choices=["dinov3-l", "vjepa21-l"], required=True)
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--device", choices=["R01", "R02", "R03", "R04"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--original-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.batch_size < 1 or args.workers < 0:
        raise ValueError("CUDA and valid extraction parameters required")
    rows = [r for r in json.loads(args.manifest.read_text())["sequences"] if r["device"] == args.device]
    prepare(rows, args.mode, args.model, args.weights, args.upstream, args.data_root,
            args.original_cache, args.out, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
