"""Causal history OFAT using immutable frozen feature residuals and phase predictions."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ipad_jepa.scoring import NormalCalibration
from ipad_jepa.temporal import common_mask, temporal_score

HISTORIES = (5, 15, 31)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


@dataclass
class SavedInference:
    """Inference inputs contain no test labels or annotation-validity flags."""
    frames: int
    targets: np.ndarray
    phase: np.ndarray
    feature: np.ndarray


def decode_inference(rows, frames, mode, phase_column="phase"):
    targets = np.array([int(r["frame"]) for r in rows])
    expected = np.arange(8, frames - 7) if mode == "offline" else np.arange(15, frames)
    if mode not in {"offline", "online"} or not np.array_equal(targets, expected):
        raise ValueError("Saved target inventory differs from the 16-frame source protocol")
    phase = np.array([float(r[phase_column]) for r in rows])
    feature = np.array([float(r["feature_raw"]) for r in rows])
    if not np.isfinite(phase).all() or not np.isfinite(feature).all() or np.any((phase < 0) | (phase >= 1)):
        raise ValueError("Expected finite features and phase predictions in [0,1)")
    return SavedInference(frames, targets, phase, feature)


def pairs_for_history(sequence, cycle, history):
    dense = np.full(sequence.frames, np.nan)
    dense[sequence.targets] = sequence.phase
    time = temporal_score(dense, cycle, history)[sequence.targets]
    return np.column_stack([sequence.feature, time])


def alarm_flags(scores, ready, threshold):
    result = np.zeros(len(scores), dtype=bool)
    streak = 0
    for index, (score, keep) in enumerate(zip(scores, ready)):
        streak = streak + 1 if keep and score > threshold else 0
        result[index] = streak >= 3
    return result


def fit_normal(normal_sequences, cycle, history, shared_history=31):
    if history not in HISTORIES or shared_history < history:
        raise ValueError("Expected an accepted history and a sufficient shared mask")
    pairs = [pairs_for_history(s, cycle, history)[common_mask(s.frames, history=shared_history)[s.targets]]
             for s in normal_sequences]
    if not pairs or any(not len(p) or not np.isfinite(p).all() for p in pairs):
        raise ValueError("Missing finite normal calibration pairs")
    return NormalCalibration().fit(np.concatenate(pairs))


def infer(sequence, calibration, cycle, history):
    pairs = pairs_for_history(sequence, cycle, history)
    score = calibration.combine(pairs)
    # Readiness uses only the available phase prefix; online tail frames remain active.
    ready = (sequence.targets >= 15 + history - 1) & np.isfinite(pairs).all(axis=1)
    flags = alarm_flags(score, ready, calibration.threshold)
    evidence = np.zeros(len(score), dtype=np.int8)
    finite = np.isfinite(pairs).all(axis=1)
    evidence[finite] = calibration.types(pairs[finite])
    return pairs, score, ready, flags, evidence


def primary_calibration_check(normal_sequences, normal_rows, source_fit, cycle):
    reference = source_fit["calibration"]["P3"]
    cal = fit_normal(normal_sequences, cycle, 5, shared_history=5)
    np.testing.assert_allclose(cal.median, reference["median"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(cal.scale, reference["mad_scale"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(cal.threshold, reference["threshold"], rtol=0, atol=1e-9)
    time_errors, score_errors = [], []
    for seq, rows in zip(normal_sequences, normal_rows):
        pairs = pairs_for_history(seq, cycle, 5)
        keep = common_mask(seq.frames)[seq.targets]
        np.testing.assert_array_equal([int(r["valid"]) for r in rows], keep.astype(int))
        expected_time = np.array([float(r["time_raw"]) for r in rows])
        expected_score = np.array([float(r["P3"]) for r in rows])
        np.testing.assert_allclose(pairs[keep, 1], expected_time[keep], rtol=0, atol=1e-12)
        actual = cal.combine(pairs)
        np.testing.assert_allclose(actual[keep], expected_score[keep], rtol=0, atol=1e-9)
        time_errors.append(float(np.max(np.abs(pairs[keep, 1] - expected_time[keep]))))
        score_errors.append(float(np.max(np.abs(actual[keep] - expected_score[keep]))))
    return cal, max(time_errors), max(score_errors)


def metric(labels, scores):
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2 or not np.isfinite(scores).all():
        raise ValueError("Expected finite evaluated scores and both binary classes")
    return {"frame_auroc": float(roc_auc_score(labels, scores)),
            "frame_ap": float(average_precision_score(labels, scores)),
            "frames": len(labels), "anomaly_frames": int(labels.sum())}


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def run_condition(source, manifest, output, model, mode, device, seed):
    original = source / model / mode / device / f"seed{seed}"
    source_metrics = json.loads((original / "metrics.json").read_text())
    source_fit = json.loads((original / "normal_fit.json").read_text())
    condition = (model, mode, device, seed)
    if (source_metrics["status"] != "complete_device_evaluation" or
        tuple(source_metrics[k] for k in ["backbone", "mode", "device", "seed"]) != condition or
        source_fit["status"] != "normal_fit_and_calibration_complete" or
        tuple(source_fit[k] for k in ["backbone", "mode", "device", "seed"]) != condition or
        source_fit["cache_identity"]["clip_frames"] != 16):
        raise ValueError("Completed matching frozen 16-frame source condition required")
    if any(r["device"] == device and r.get("split") in {"fit", "calibration"} and
           r["partition"] != "training" for r in manifest):
        raise ValueError("Normal fitting and calibration may use only training videos")
    calibration_rows = {r["sequence"]: r for r in manifest
                        if r["device"] == device and r.get("split") == "calibration"}
    fit_rows = [r for r in manifest if r["device"] == device and r.get("split") == "fit"]
    tests = {r["sequence"]: r for r in manifest if r["device"] == device and r["partition"] == "testing"}
    if set(calibration_rows) & {r["sequence"] for r in fit_rows}:
        raise ValueError("Normal fit and calibration videos overlap")
    cycle = source_fit["cycle_length_fit_median"]
    np.testing.assert_allclose(cycle, np.median([r["frames"] for r in fit_rows]), rtol=0, atol=0)
    all_normal = read_csv(original / "normal_calibration.csv")
    if {r["sequence"] for r in all_normal} != calibration_rows.keys():
        raise ValueError("Normal calibration video inventory changed")
    normal_rows = [[r for r in all_normal if r["sequence"] == key] for key in sorted(calibration_rows)]
    normal = [decode_inference(rows, calibration_rows[key]["frames"], mode, "predicted_phase")
              for key, rows in zip(sorted(calibration_rows), normal_rows)]
    primary, time_error, score_error = primary_calibration_check(normal, normal_rows, source_fit, cycle)
    sources = {name: digest(original / name) for name in ["metrics.json", "normal_fit.json", "normal_calibration.csv"]}
    # Fix every history's normal-only calibration before opening any test score files.
    calibrators = {h: fit_normal(normal, cycle, h) for h in HISTORIES}
    destinations = {h: output / f"H{h}" / model / mode / device / f"seed{seed}" for h in HISTORIES}
    for history, folder in destinations.items():
        if folder.exists():
            raise FileExistsError("Inspect an existing ablation; never overwrite it")
        (folder / "P3").mkdir(parents=True)
        cal = calibrators[history]
        meta = {"status": "normal_fit_and_calibration_complete", "backbone": model, "mode": mode,
                "device": device, "seed": seed, "history": history, "shared_comparison_history": 31,
                "calibration_mask": "t=45..N-8 inclusive", "cycle_length_fit_median": cycle,
                "score_weights": [.5, .5], "calibration_quantile": .99,
                "calibration": {"P3": {"median": cal.median.tolist(), "mad_scale": cal.scale.tolist(),
                    "threshold": cal.threshold, "component_thresholds": cal.component_thresholds.tolist()}},
                "source_normal_fit_sha256": sources["normal_fit.json"],
                "source_phase_checkpoint_sha256": source_fit["phase_checkpoint_sha256"],
                "source_cache_identity": source_fit["cache_identity"],
                "unchanged_components": ["frozen_encoder", "phase_head", "PCA", "prototypes", "neighbours", "temperature", "feature_residual"],
                "code_sha256": digest(__file__), "temporal_code_sha256": digest(Path(__file__).with_name("temporal.py")),
                "scope": "Normal-only recalibration on shared history31 mask; cached accuracy ablation, not a GPU or runtime benchmark"}
        atomic_json(folder / "normal_fit.json", meta)
        with (folder / "normal_calibration.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["sequence", "frame", "valid", "relative_phase", "predicted_phase", "feature_raw", "time_raw", "P3"])
            for key, seq in zip(sorted(calibration_rows), normal):
                pairs = pairs_for_history(seq, cycle, history)
                keep = common_mask(seq.frames, history=31)[seq.targets]
                writer.writerows(zip([key] * len(seq.targets), seq.targets, keep.astype(int),
                                     seq.targets / seq.frames, seq.phase, pairs[:, 0], pairs[:, 1], cal.combine(pairs)))
    results = {h: {"labels": [], "scores": []} for h in HISTORIES}
    if {p.stem for p in (original / "P3").glob("*.csv")} != tests.keys():
        raise ValueError("Source test video inventory differs from audited manifest")
    for key in sorted(tests, key=int):
        rows = read_csv(original / "P3" / f"{key}.csv")
        sources[f"P3/{key}.csv"] = digest(original / "P3" / f"{key}.csv")
        seq = decode_inference(rows, tests[key]["frames"], mode)
        # Verify the primary H5 protocol before performing the stricter-mask OFAT comparison.
        primary_pairs = pairs_for_history(seq, cycle, 5)
        primary_keep = common_mask(seq.frames)[seq.targets]
        primary_score = primary.combine(primary_pairs)
        expected_score = np.array([float(r["score"]) for r in rows])
        np.testing.assert_allclose(primary_score[primary_keep], expected_score[primary_keep], rtol=0, atol=1e-9)
        score_error = max(score_error, float(np.max(np.abs(primary_score[primary_keep] - expected_score[primary_keep]))))
        np.testing.assert_array_equal(alarm_flags(primary_score, primary_keep, primary.threshold),
                                      np.array([int(r["alarm"]) for r in rows], dtype=bool))
        # Annotation uncertainty affects evaluation only, never scores or alarm readiness.
        labels = np.array([float(r["label"]) for r in rows])
        if not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("Unexpected source labels")
        np.testing.assert_array_equal([int(r["valid"]) for r in rows], (primary_keep & (labels >= 0)).astype(int))
        valid = common_mask(seq.frames, history=31)[seq.targets] & (labels >= 0)
        for history, cal in calibrators.items():
            pairs, scores, ready, alarms, evidence = infer(seq, cal, cycle, history)
            results[history]["labels"].append(labels[valid].astype(np.int8))
            results[history]["scores"].append(scores[valid])
            with (destinations[history] / "P3" / f"{key}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["frame", "label", "valid", "inference_valid", "phase", "feature_raw", "time_raw", "score", "alarm", "evidence_type"])
                writer.writerows(zip(seq.targets, labels, valid.astype(int), ready.astype(int), seq.phase,
                                     pairs[:, 0], pairs[:, 1], scores, alarms.astype(int), evidence))
    proof = {"status": "passed", "backbone": model, "mode": mode, "device": device, "seed": seed,
             "primary_H5_original_mask": "t=19..N-8 inclusive", "normal_time_max_absolute_error": time_error,
             "calibration_and_test_score_max_absolute_error": score_error, "alarm_mismatches": 0,
             "test_videos_checked": len(tests), "sources": sources,
             "scope": "Replay of published residuals/phases; encoder features are not independently re-extracted"}
    for history, folder in destinations.items():
        atomic_json(folder / "source_replay_check.json", proof)
        atomic_json(folder / "metrics.json", {"status": "complete_device_evaluation", "backbone": model,
            "mode": mode, "device": device, "seed": seed, "history": history,
            "valid_mask": "t=45..N-8 inclusive, intersect original shared annotation uncertainty mask",
            "test_videos": len(tests), "variants": {"P3": metric(np.concatenate(results[history]["labels"]),
                                                                    np.concatenate(results[history]["scores"]))},
            "note": "History-only cached-score OFAT with shared calibration/evaluation mask; not the primary H5 mask or runtime performance"})
    print(f"history ablation complete: {model}/{mode}/{device}/seed{seed}; H5 replay max error={score_error:.3g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/stage02"))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/stage05/ablations/history"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("A fresh output root is required; inspect partial runs before recovery")
    manifest = json.loads(args.manifest.read_text())["sequences"]
    required = {(m, mode, d) for m in ["dinov3-l", "vjepa21-l"] for mode in ["offline", "online"]
                for d in ["R01", "R02", "R03", "R04"]}
    summary = json.loads((args.source / "device_summary.json").read_text())
    if {(r["backbone"], r["mode"], r["device"]) for r in summary["results"] if r["seeds"] == 3} != required:
        raise ValueError("The full frozen 2x2 four-device source matrix is required")
    for model, mode, device in sorted(required):
        for seed in [0, 1, 2]:
            run_condition(args.source, manifest, args.out, model, mode, device, seed)
    atomic_json(args.out / "completion.json", {"status": "complete_history_matrix", "conditions": 48,
        "histories": list(HISTORIES), "evaluated_seed_conditions": 144, "source_summary_sha256": digest(args.source / "device_summary.json"),
        "manifest_sha256": digest(args.manifest), "code_sha256": digest(__file__),
        "scope": "Frozen 2 backbones x 2 modes x 4 devices x 3 seeds x histories5/15/31; aggregation and publication checks still required"})


if __name__ == "__main__":
    main()
