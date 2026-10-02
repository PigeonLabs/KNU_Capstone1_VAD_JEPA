"""Validate compact dense-fit provenance and derive bounded mean queries from frozen caches."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ipad_jepa.cache_data import sequences
from ipad_jepa.features import file_hash
from ipad_jepa.representation import SEEDS, candidate_plan, capacity, fit_targets, local_spatial_mean
from ipad_jepa.representation_features import write_json


def validate_dense(root, rows, original_cache, seed=None):
    info = json.loads((root / "dense_fit.json").read_text())
    fit = [r for r in rows if r["partition"] == "training" and r.get("split") == "fit"]
    identity = info["identity"]
    cal = sequences(original_cache, rows, "calibration")
    source = Path(__file__).parent
    if (info["status"] != "complete_dense_normal_fit" or info["seeds"] != list(SEEDS)
            or info["device"] != fit[0]["device"]
            or identity["fit_stride"] != 1 or identity["clip_frames"] != 16
            or info["capacity"] != capacity(fit, identity["mode"], 1)
            or info["original_cache_identity"] != cal[0].identity):
        raise ValueError("Dense fit protocol or original calibration identity differs")
    expected_sources = {"reader_sha256": "representation_features.py", "sampling_code_sha256": "representation.py",
                        "base_reader_sha256": "features.py", "adapter_sha256": "backbones.py"}
    if any(identity[key] != file_hash(source / name) for key, name in expected_sources.items()):
        raise ValueError("Pinned dense-fit producer changed")
    for key in ["backbone", "mode", "weights_sha256", "upstream_commit", "adapter_sha256", "image_size",
                "clip_frames", "preprocessing", "feature_dtype"]:
        if identity[key] != cal[0].identity[key]:
            raise ValueError(f"Dense/calibration encoder semantics differ: {key}")
    if cal[0].identity["reader_sha256"] != identity["base_reader_sha256"]:
        raise ValueError("Original cached reader is not the pinned base reader")
    expected_inventory = [{"sequence": r["sequence"], "frames": r["frames"],
                           "clips": len(fit_targets(r["frames"], identity["mode"])),
                           "frame_names_sha256": r["names_sha256"], "frame_content_sha256": r["frames_content_sha256"],
                           "padding": identity["mode"] == "online"} for r in fit]
    if [{k: p.get(k) for k in expected_inventory[0]} for p in info["fit_inventory"]] != expected_inventory:
        raise ValueError("Dense fit inventory does not match normal manifest")
    required = {f"fit/{r['sequence']}/{name}.npy" for r in fit for name in ["targets", "global", "mean"]}
    required |= {f"seed{s}_{name}.npy" for s in SEEDS for name in ["coordinates", "phases", "patch"]}
    if set(info["outputs_sha256"]) != required:
        raise ValueError("Dense fit output inventory is incomplete or unexpected")
    for name, sha in info["outputs_sha256"].items():
        if name.startswith("fit/") or (seed is not None and name.startswith(f"seed{seed}_")):
            if file_hash(root / name) != sha:
                raise ValueError(f"Dense fit output changed: {name}")
    for row in fit:
        folder = root / "fit" / row["sequence"]
        targets = np.load(folder / "targets.npy", allow_pickle=False)
        if not np.array_equal(targets, fit_targets(row["frames"], identity["mode"])):
            raise ValueError("Dense fit targets are incomplete")
        for name, shape, dtype in [("global", (len(targets), 1024), np.float16), ("mean", (len(targets), 1, 1024), np.float32)]:
            value = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if value.shape != shape or value.dtype != dtype or not np.all(np.isfinite(value)):
                raise ValueError("Invalid compact dense-fit features")
    return info, fit, cal


def dense_candidates(root, fit_rows, mode, seed, representation):
    queries = 576 if representation == "patch" else 1
    coordinates, phases, sampling = candidate_plan(fit_rows, mode, seed, queries=queries)
    if representation == "patch":
        if (not np.array_equal(coordinates, np.load(root / f"seed{seed}_coordinates.npy", allow_pickle=False))
                or not np.array_equal(phases, np.load(root / f"seed{seed}_phases.npy", allow_pickle=False))):
            raise ValueError("Stored normal patch sampling differs from the seeded plan")
        value = np.load(root / f"seed{seed}_patch.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != (len(coordinates), 1024) or value.dtype != np.float16:
            raise ValueError("Invalid normal patch candidate dimensions")
        features = value.astype(np.float32)
    elif representation == "global_mean":
        features = np.empty((len(coordinates), 1024), dtype=np.float32)
        for video, row in enumerate(fit_rows):
            ids = np.flatnonzero(coordinates[:, 0] == video)
            value = np.load(root / "fit" / row["sequence"] / "mean.npy", mmap_mode="r", allow_pickle=False)
            features[ids] = value[coordinates[ids, 1], 0]
    else:
        raise ValueError("Unknown representation")
    if not np.all(np.isfinite(features)):
        raise ValueError("Nonfinite observed fit candidates")
    return features, phases, coordinates[:, 0], sampling


def mean_sequence(sequence, view_root):
    """Retain the original full-context phase input and replace only local queries."""
    folder = view_root / sequence.row["partition"] / sequence.row["sequence"]
    metadata = folder / "mean_source.json"
    identity = {"source_fingerprint": sequence.meta["fingerprint"],
                "source_metadata_sha256": file_hash(sequence.folder / "meta.json"),
                "source_targets_sha256": file_hash(sequence.folder / "targets.npy"),
                "pooling_code_sha256": file_hash(Path(__file__).with_name("representation.py")),
                "derivation_code_sha256": file_hash(Path(__file__)),
                "representation": "FP32 spatial mean of FP16 target-local patches", "rows": len(sequence.targets)}
    if metadata.exists():
        info = json.loads(metadata.read_text())
        if info.get("status") != "complete_mean_view" or info.get("identity") != identity:
            raise ValueError("Stale or partial local mean view")
        if file_hash(folder / "mean.npy") != info["mean_sha256"]:
            raise ValueError("Derived mean queries changed")
    else:
        if folder.exists():
            raise ValueError("Partial mean view; do not overwrite a possible live writer")
        folder.mkdir(parents=True)
        value = np.lib.format.open_memmap(folder / "mean.npy", mode="w+", dtype=np.float32,
                                         shape=(len(sequence.targets), 1, 1024))
        for start in range(0, len(sequence.targets), 32):
            value[start:start + 32] = local_spatial_mean(sequence.patches[start:start + 32])
        value.flush(); del value
        write_json(metadata, {"status": "complete_mean_view", "identity": identity,
                              "mean_sha256": file_hash(folder / "mean.npy")})
    result = copy.copy(sequence)
    result.patches = np.load(folder / "mean.npy", mmap_mode="r", allow_pickle=False)
    if (result.patches.shape != (len(result.targets), 1, 1024) or result.patches.dtype != np.float32
            or not np.all(np.isfinite(result.patches))):
        raise ValueError("Invalid mean query shape, dtype or values")
    result.mean_source = json.loads(metadata.read_text())
    return result
