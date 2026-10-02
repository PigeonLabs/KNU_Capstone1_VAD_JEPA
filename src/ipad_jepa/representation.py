"""Torch-free geometry and sampling for the paired dense-fit representation ablation."""
from __future__ import annotations

import numpy as np
from ipad_jepa.memory import balanced_counts

BINS = 16
PER_BIN = 128
PATCHES = 576
DIMENSION = 1024
SEEDS = (0, 1, 2)


def fit_targets(length, mode, stride=1):
    if length < 16 or mode not in {"offline", "online"} or stride < 1:
        raise ValueError("Complete sixteen-frame normal fit inputs required")
    start, stop = (8, length - 7) if mode == "offline" else (0, length)
    return np.arange(start, stop, stride, dtype=np.int64)


def capacity(rows, mode, stride):
    if not rows or any(r["partition"] != "training" or r.get("split") != "fit" for r in rows):
        raise ValueError("Only audited normal fit videos may supply candidates")
    if len({(r["device"], r["sequence"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate fit videos")
    if len({r["device"] for r in rows}) != 1:
        raise ValueError("Do not combine device-specific memories")
    targets = [fit_targets(r["frames"], mode, stride) for r in rows]
    counts = np.zeros(BINS, dtype=np.int64)
    for row, ids in zip(rows, targets):
        counts += np.bincount(np.floor(ids / row["frames"] * BINS).astype(int), minlength=BINS)
    return {"mode": mode, "fit_stride": stride, "fit_videos": len(rows),
            "fit_clips": sum(map(len, targets)), "mean_candidates_per_bin": counts.tolist(),
            "minimum_mean_candidates": int(counts.min()),
            "enough_for_128_per_bin": bool(np.all(counts >= PER_BIN))}


def candidate_plan(rows, mode, seed, limit=10000, queries=PATCHES):
    """Sample actual (video, dense-fit row, spatial patch) coordinates without replacement.

    This is the existing normal_candidates sampling order, before any encoder output
    is opened. Only the requested candidates are retained; every dense-fit clip is
    still encoded and supplies its pooled phase input and local spatial mean.
    """
    audit = capacity(rows, mode, 1)
    if not audit["enough_for_128_per_bin"] or limit < PER_BIN or queries not in {1, PATCHES}:
        raise ValueError("Insufficient observed normal candidates; never duplicate prototypes")
    targets = [fit_targets(r["frames"], mode) for r in rows]
    rng = np.random.default_rng(seed)
    coordinates, phases, sampling = [], [], []
    for b in range(BINS):
        clips = [np.flatnonzero(np.floor(ids / r["frames"] * BINS).astype(int) == b)
                 for r, ids in zip(rows, targets)]
        capacities = np.array([len(ids) * queries for ids in clips], dtype=np.int64)
        counts = balanced_counts(capacities, limit, rng)
        for video, (row, ids, count, available) in enumerate(zip(rows, clips, counts, capacities)):
            if not count:
                continue
            chosen = rng.choice(int(available), int(count), replace=False)
            ci, pi = ids[chosen // queries], chosen % queries
            coordinates.append(np.column_stack((np.full(len(ci), video), ci, pi)))
            phases.append(targets[video][ci] / row["frames"])
        sampling.append({"phase_bin": b, "tokens": int(counts.sum()),
                         "per_video": {r["sequence"]: int(n) for r, n in zip(rows, counts)}})
    return np.concatenate(coordinates).astype(np.int64), np.concatenate(phases), sampling


def local_spatial_mean(patches):
    """Average the target-local FP16 cache-equivalent patches in FP32, retaining one query.

    The whole-context pooled feature remains the separate phase-head input.
    No averaging across clip time, normalization before averaging, or special tokens.
    """
    patches = np.asarray(patches)
    if patches.ndim != 3 or patches.shape[1:] != (PATCHES, DIMENSION):
        raise ValueError("Expected target-local B x 576 x 1024 patches")
    if not np.all(np.isfinite(patches)):
        raise ValueError("Nonfinite local patches")
    result = patches.mean(axis=1, dtype=np.float32, keepdims=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite pooled query")
    return result
