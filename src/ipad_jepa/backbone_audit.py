"""CPU-only checks for the separately refitted 768-dimensional B pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def online_targets(frames, split):
    if split not in {"fit", "calibration", "test"} or frames < 16:
        raise ValueError("Expected a complete online sequence and accepted split")
    return np.arange(0, frames, 4, dtype=np.int64) if split == "fit" else np.arange(15, frames, dtype=np.int64)


def check_small_cache(meta, row, smoke, ids, patch, context):
    spec = meta["spec"]
    split = row.get("split", "test")
    expected = {"backbone": smoke["model"], "mode": "online", "feature_dimension": 768,
                "clip_frames": 16, "fit_stride": 4, "image_size": 384,
                "device": row["device"], "partition": row["partition"], "sequence": row["sequence"],
                "split": split, "frames": row["frames"], "padding": split == "fit",
                "frame_names_sha256": row["names_sha256"], "frame_content_sha256": row["frames_content_sha256"],
                "weights_sha256": smoke["weight_sha256"], "upstream_commit": smoke["upstream_commit"],
                "adapter_sha256": smoke["adapter_sha256"], "base_reader_sha256": smoke["reader_sha256"],
                "preprocessing": "RGB full-frame PIL bilinear resize; ImageNet mean/std",
                "feature_dtype": "float16 from BF16 inference"}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (meta.get("status") != "complete" or meta.get("fingerprint") != fingerprint
            or any(spec.get(k) != v for k, v in expected.items())
            or row["partition"] != ("testing" if split == "test" else "training")):
        raise ValueError("B cache identity, normal split or online protocol differs")
    expected_ids = online_targets(row["frames"], split)
    if ids.dtype != np.int64 or spec.get("rows") != len(expected_ids):
        raise ValueError("B target count/dtype differs")
    np.testing.assert_array_equal(ids, expected_ids)
    if (patch.shape != (len(ids), 576, 768) or context.shape != (len(ids), 768)
            or patch.dtype != np.float16 or context.dtype != np.float16):
        raise ValueError("B feature geometry/dtype differs")
    if not np.isfinite(context).all():
        raise ValueError("Nonfinite B phase input")


def check_phase_selection(phase, curve, expected):
    if (phase.get("status") != "complete" or phase.get("epochs") != 20
            or phase.get("feature_dimension") != 768 or phase.get("phase_batch_size") != 256
            or any(phase.get(k) != v for k, v in expected.items())
            or len(curve) != 20 or [int(r["epoch"]) for r in curve] != list(range(1, 21))):
        raise ValueError("Twenty complete normal B phase epochs required")
    values = np.array([[float(r[k]) for k in ["train_ce", "normal_calibration_ce", "normal_calibration_circular_mae"]] for r in curve])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite B phase training statistics")
    selected = int(values[:, 1].argmin()) + 1
    if phase.get("selected_epoch") != selected or phase.get("best_normal_calibration_ce") != float(values[selected - 1, 1]):
        raise ValueError("B head does not minimize normal calibration CE")


def check_bank(bank, normal):
    expected = {"mean": (768,), "components": (256, 768), "prototypes": (16, 128, 256),
                "temperature": (), "cycle_length": ()}
    if set(bank.files) != set(expected):
        raise ValueError("B bank array inventory differs")
    for name, shape in expected.items():
        if bank[name].shape != shape or not np.isfinite(bank[name]).all():
            raise ValueError("B bank geometry or finite values differ")
        if name in {"mean", "components", "prototypes"} and bank[name].dtype != np.float32:
            raise ValueError("B bank PCA/prototype dtype differs")
    np.testing.assert_allclose([float(bank["temperature"]), float(bank["cycle_length"])],
                               [normal["temperature"], normal["cycle_length_fit_median"]], rtol=0, atol=0)
    np.testing.assert_allclose(np.linalg.norm(bank["prototypes"], axis=-1), 1, rtol=0, atol=2e-5)
    np.testing.assert_allclose(bank["components"] @ bank["components"].T, np.eye(256), rtol=0, atol=2e-4)


def check_pins(pinned, required, digest):
    actual = {}
    for name, value in pinned.items():
        path = Path(name)
        if path.is_absolute() and path.name == "run_small_matrix.py":
            name = "scripts/run_small_matrix.py"
        if name not in required or name in actual or digest(name) != value:
            raise ValueError("B pinned source changed, duplicated or unexpected")
        actual[name] = value
    if set(actual) != set(required):
        raise ValueError("B required pinned sources missing")
    return actual
