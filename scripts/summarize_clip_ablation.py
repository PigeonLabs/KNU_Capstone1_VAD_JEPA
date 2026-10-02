"""Independently audit clip8 provenance/calibration/GT and paired clip16 accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ipad_jepa.alignment import align
from ipad_jepa.temporal import temporal_score
from summarize_experiments import load_run, check_pair, statistic, bootstrap, bootstrap_draws, macro_four_devices
from summarize_neighbour_ablation import digest, read, write, difference

MODELS = ("dinov3-l", "vjepa21-l")
MODES = ("offline", "online")
DEVICES = ("R01", "R02", "R03", "R04")


def targets(length, mode, frames):
    if mode not in MODES or frames not in (8, 16):
        raise ValueError("Expected an accepted mode and clip length")
    return np.arange(frames // 2, length - frames // 2 + 1) if mode == "offline" else np.arange(frames - 1, length)


def audit_trace(trace, info, mode, frames, cycle, params=None, labels=None, known=None):
    """Verify complete clips and original common-mask batch alarms, not streaming EOF."""
    ids = np.array([int(r["frame"]) for r in trace])
    np.testing.assert_array_equal(ids, targets(info["frames"], mode, frames))
    normal = labels is None
    phi = np.array([float(r["predicted_phase" if normal else "phase"]) for r in trace])
    if not np.isfinite(phi).all() or np.any((phi < 0) | (phi >= 1)):
        raise ValueError("Invalid predicted phase")
    dense = np.full(info["frames"], np.nan)
    dense[ids] = phi
    raw = np.array([[float(r["feature_raw"]), float(r["time_raw"])] for r in trace])
    np.testing.assert_allclose(raw[:, 1], temporal_score(dense, cycle, 5)[ids],
                               rtol=0, atol=1e-12, equal_nan=True)
    if not np.isfinite(raw[:, 0]).all():
        raise ValueError("Nonfinite feature score")
    common = (ids >= 19) & (ids <= info["frames"] - 8)
    if normal:
        expected_valid = common
        np.testing.assert_allclose([float(r["relative_phase"]) for r in trace],
                                   ids / info["frames"], rtol=0, atol=1e-12)
    else:
        np.testing.assert_array_equal([int(float(r["label"])) for r in trace], labels[ids])
        np.testing.assert_array_equal([int(r["inference_valid"]) for r in trace], common)
        expected_valid = common & known[ids]
    np.testing.assert_array_equal([int(r["valid"]) for r in trace], expected_valid)
    if not np.isfinite(raw[common]).all():
        raise ValueError("Nonfinite common calibration/evaluation scores")
    if params is not None:
        scores = ((raw - params["median"]) / params["mad_scale"]).mean(axis=1)
        np.testing.assert_allclose(scores, [float(r["P3" if normal else "score"]) for r in trace],
                                   rtol=0, atol=1e-9, equal_nan=True)
        if not normal:
            finite = np.isfinite(raw).all(axis=1)
            types = np.zeros(len(ids), dtype=int)
            types[finite] = ((raw[finite, 0] > params["component_thresholds"][0]).astype(int)
                            + 2 * (raw[finite, 1] > params["component_thresholds"][1]).astype(int))
            np.testing.assert_array_equal(types, [int(r["evidence_type"]) for r in trace])
            streak = 0
            for keep, score, row in zip(common, scores, trace):
                streak = streak + 1 if keep and score > params["threshold"] else 0
                if int(row["alarm"]) != int(streak >= 3):
                    raise ValueError("Batch alarm replay differs; GT must not control its streak")
    return raw[common]


def audit_calibration(folder, manifest, device, mode, frames, cycle, params):
    expected = {r["sequence"]: r for r in manifest if r["device"] == device and r.get("split") == "calibration"}
    rows = read(folder / "normal_calibration.csv")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    if set(grouped) != set(expected) or any(r["partition"] != "training" for r in expected.values()):
        raise ValueError("Normal calibration inventory differs")
    pairs = np.concatenate([audit_trace(grouped[key], expected[key], mode, frames, cycle)
                            for key in expected])
    median = np.median(pairs, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(pairs - median), axis=0), 1e-6)
    components = np.quantile(pairs, .99, axis=0)
    normalized = ((pairs - median) / scale).mean(axis=1)
    for actual, recorded in [(median, params["median"]), (scale, params["mad_scale"]),
                             (components, params["component_thresholds"]),
                             (np.quantile(normalized, .99), params["threshold"])]:
        np.testing.assert_allclose(actual, recorded, rtol=0, atol=1e-9)
    for key in expected:
        audit_trace(grouped[key], expected[key], mode, frames, cycle, params)
    return len(pairs)


def audit_clip8_source(folder, local, cache, manifest, model, mode, device, seed,
                       manifest_path=Path("results/stage00/manifest.json")):
    proof = json.loads((folder / "clip8_source_proof.json").read_text())
    phase = json.loads((folder / "phase_training.json").read_text())
    if (proof["status"] != "completed_clip8_condition" or proof["clip_frames"] != 8
            or tuple(proof[k] for k in ["backbone", "mode", "device", "seed"]) != (model, mode, device, seed)
            or phase["status"] != "complete" or phase["epochs"] != 20
            or tuple(phase[k] for k in ["backbone", "mode", "device", "seed"]) != (model, mode, device, seed)):
        raise ValueError("Incomplete or mismatched clip8 source/head")
    required_sources = {"scripts/run_clip8_matrix.py", str(manifest_path), "configs/experiment_matrix.yaml"}
    required_sources.update(f"src/ipad_jepa/{name}.py" for name in [
        "clip8_features", "backbones", "features", "temporal", "audit", "cache_data", "train_phase",
        "experiment", "memory", "torch_memory", "scoring", "alignment"])
    normalized_sources = {}
    for name, value in proof["source_sha256"].items():
        path = Path(name)
        # The running producer records its script's absolute __file__. Re-audit
        # the exact relative script in a new checkout, without requiring that host path.
        name = "scripts/run_clip8_matrix.py" if path.is_absolute() and path.name == "run_clip8_matrix.py" else name
        if name in normalized_sources or name not in required_sources or digest(name) != value:
            raise ValueError(f"Pinned source changed or inventory differs: {name}")
        normalized_sources[name] = value
    if set(normalized_sources) != required_sources:
        raise ValueError("Required pinned source inventory missing")
    tests = [r for r in manifest if r["device"] == device and r["partition"] == "testing"]
    expected_outputs = {"metrics.json", "normal_fit.json", "normal_calibration.csv", "phase_training.json", "phase_training.csv"}
    expected_outputs.update(f"{variant}/{row['sequence']}.csv" for variant in ["P0", "P1", "P2", "P3"] for row in tests)
    if set(proof["outputs_sha256"]) != expected_outputs:
        raise ValueError("Required completed output checksum inventory missing or differs")
    for name, value in proof["outputs_sha256"].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or digest(path) != value:
            raise ValueError("Completed output changed")
    for name in ["phase_training.json", "phase_training.csv"]:
        if digest(local / name) != digest(folder / name):
            raise ValueError("Published phase log differs from actual selected run")
    if digest(local / "phase_head.pt") != proof["phase_head_sha256"] or digest(local / "memory.npz") != proof["memory_sha256"]:
        raise ValueError("Actual phase head or bank checksum differs")
    curve = read(folder / "phase_training.csv")
    if len(curve) != 20 or [int(r["epoch"]) for r in curve] != list(range(1, 21)):
        raise ValueError("Full 20-epoch phase training required")
    ce = np.array([float(r["normal_calibration_ce"]) for r in curve])
    if not np.isfinite(ce).all() or phase["selected_epoch"] != int(ce.argmin()) + 1:
        raise ValueError("Selected phase head does not minimize normal calibration CE")
    np.testing.assert_allclose(ce.min(), phase["best_normal_calibration_ce"], rtol=0, atol=1e-12)
    required = {(r["partition"], r["sequence"]): r for r in manifest if r["device"] == device
                and r.get("split", "test") in {"fit", "calibration", "test"}}
    smoke = json.loads(Path(f"results/setup/{model}_clip8_smoke.json").read_text())
    if smoke["status"] != "strict_load_and_real_clip8_both_modes_passed" or smoke["backbone"] != model:
        raise ValueError("Actual selected backbone's clip8 GPU verification missing")
    recorded = {(r["partition"], r["sequence"]): r for r in proof["cache_metadata"]}
    if len(recorded) != len(proof["cache_metadata"]) or set(recorded) != set(required):
        raise ValueError("Actual cache inventory differs")
    normal_fingerprints, counts, identity = [], {"fit": 0, "calibration": 0}, None
    for key, row in required.items():
        evidence = recorded[key]
        base = cache / device / key[0] / key[1]
        meta = json.loads((base / "meta.json").read_text())
        spec = meta["spec"]
        expected = {"device": device, "partition": key[0], "sequence": key[1],
                    "backbone": model, "mode": mode, "clip_frames": 8, "fit_stride": 4,
                    "frames": row["frames"], "split": row.get("split", "test"), "image_size": 384,
                    "frame_names_sha256": row["names_sha256"], "frame_content_sha256": row["frames_content_sha256"]}
        hashed = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if (meta["status"] != "complete" or spec != evidence["spec"] or hashed != meta["fingerprint"]
                or hashed != evidence["fingerprint"] or any(spec.get(k) != v for k, v in expected.items())
                or digest(base / "meta.json") != evidence["metadata_sha256"]
                or digest(base / "targets.npy") != evidence["targets_sha256"]
                or spec["reader_sha256"] != digest("src/ipad_jepa/clip8_features.py")
                or spec["adapter_sha256"] != digest("src/ipad_jepa/backbones.py")
                or spec["base_reader_sha256"] != digest("src/ipad_jepa/features.py")
                or spec["temporal_code_sha256"] != digest("src/ipad_jepa/temporal.py")):
            raise ValueError("Cache source or eight-frame fingerprint differs")
        if any(spec[key] != smoke["spec"][key] for key in ["weights_sha256", "upstream_commit", "adapter_sha256",
               "reader_sha256", "base_reader_sha256", "temporal_code_sha256", "image_size", "feature_dtype", "preprocessing"]):
            raise ValueError("Cache differs from the actual clip8 GPU-verified encoder")
        split = spec["split"]
        if split == "fit":
            start = 4 if mode == "offline" else 0
            stop = row["frames"] - 3 if mode == "offline" else row["frames"]
            expected_ids = np.arange(start, stop, 4)
        else:
            expected_ids = targets(row["frames"], mode, 8)
        np.testing.assert_array_equal(np.load(base / "targets.npy", allow_pickle=False), expected_ids)
        if spec["rows"] != len(expected_ids) or spec["padding"] != (mode == "online" and split == "fit"):
            raise ValueError("Clip inventory or padding policy differs")
        for name, shape in [("patch", (len(expected_ids), 576, 1024)), ("global", (len(expected_ids), 1024))]:
            array = np.load(base / f"{name}.npy", allow_pickle=False, mmap_mode="r")
            if array.shape != shape or array.dtype != np.float16:
                raise ValueError("Actual eight-frame feature dimensions/dtype differ")
        current = {k: spec[k] for k in ["backbone", "mode", "weights_sha256", "upstream_commit", "adapter_sha256",
                                      "reader_sha256", "image_size", "clip_frames", "fit_stride", "preprocessing", "feature_dtype"]}
        if identity is not None and identity != current:
            raise ValueError("Mixed cache identities")
        identity = current
        if split in counts:
            if row["partition"] != "training":
                raise ValueError("Test features used for normal head training")
            normal_fingerprints.append(meta["fingerprint"])
            counts[split] += len(expected_ids)
    if sorted(normal_fingerprints) != sorted(phase["cache_fingerprints"]) or counts != {"fit": phase["fit_clips"], "calibration": phase["calibration_clips"]}:
        raise ValueError("Selected normal head's inputs differ from actual clip8 features")
    return proof, phase, identity


def audit_condition(folder, manifest, data_root, model, mode, device, seed, frames, source=None):
    meta = json.loads((folder / "metrics.json").read_text())
    normal = json.loads((folder / "normal_fit.json").read_text())
    if (meta["status"] != "complete_device_evaluation" or normal["status"] != "normal_fit_and_calibration_complete"
            or any(tuple(obj[k] for k in ["backbone", "mode", "device", "seed"]) != (model, mode, device, seed)
                   for obj in [meta, normal]) or normal["cache_identity"]["clip_frames"] != frames
            or (normal["bins"], normal["prototypes_per_bin"], normal["total_prototypes"], normal["pca_dimensions"])
            != (16, 128, 2048, 256) or normal["pca_samples"] != 50000
            or not np.isfinite(normal["temperature"]) or normal["temperature"] < 1e-6
            or normal["temperature_samples"] != 50000):
        raise ValueError("Completed condition identity or memory geometry differs")
    if source is not None:
        proof, phase, identity = source
        if (normal["cache_identity"] != identity or normal["phase_checkpoint_sha256"] != proof["phase_head_sha256"]
                or normal["phase_selected_epoch"] != phase["selected_epoch"]
                or sorted(normal["normal_cache_fingerprints"]) != sorted(phase["cache_fingerprints"])
                or normal["code_sha256"] != digest("src/ipad_jepa/experiment.py")
                or normal["memory_code_sha256"] != digest("src/ipad_jepa/memory.py")):
            raise ValueError("Normal memory differs from selected head/current evaluator")
    fits = [r for r in manifest if r["device"] == device and r.get("split") == "fit"]
    if not fits or any(r["partition"] != "training" for r in fits):
        raise ValueError("Invalid normal fit inventory")
    cycle = float(np.median([r["frames"] for r in fits]))
    np.testing.assert_allclose(cycle, normal["cycle_length_fit_median"], rtol=0, atol=0)
    params = normal["calibration"]["P3"]
    cal_count = audit_calibration(folder, manifest, device, mode, frames, cycle, params)
    if cal_count != normal["calibration_valid_frames"]:
        raise ValueError("Normal q99 frame count differs")
    expected = {r["sequence"]: r for r in manifest if r["device"] == device and r["partition"] == "testing"}
    files = {p.stem: p for p in (folder / "P3").glob("*.csv")}
    if set(files) != set(expected):
        raise ValueError("Test video inventory differs")
    hashes = {}
    for sequence, row in expected.items():
        label_file = data_root / row["label_file"]
        if digest(label_file) != row["label_sha256"]:
            raise ValueError("Actual raw test labels changed")
        labels, known, _ = align(np.load(label_file, allow_pickle=False), row["frames"])
        audit_trace(read(files[sequence]), row, mode, frames, cycle, params, labels, known)
        hashes[sequence] = {"score_sha256": digest(files[sequence]), "actual_annotation_sha256": digest(label_file)}
    run = load_run(folder, "P3", meta)
    return run, {"backbone": model, "mode": mode, "device": device, "seed": seed, "clip_frames": frames,
                 "normal_common_frames": cal_count, "threshold": params["threshold"],
                 "normal_calibration_sha256": digest(folder / "normal_calibration.csv"),
                 "normal_fit_sha256": digest(folder / "normal_fit.json"), "metrics_sha256": digest(folder / "metrics.json"),
                 "test_sources": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("root", "results/stage05/ablations/clip_frames"), ("source", "results/stage02"),
                          ("cache", "artifacts/features_clip8"), ("local", "artifacts/runs_clip8"),
                          ("manifest", "results/stage00/manifest.json"), ("data-root", "../IPAD_dataset/IPAD_dataset")]:
        parser.add_argument("--" + name, type=Path, default=Path(default))
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())["sequences"]
    required = [(m, mode, d) for m in MODELS for mode in MODES for d in DEVICES]
    complete = [key for key in required if all((args.root / "T8" / key[0] / key[1] / key[2] / f"seed{s}" / "clip8_source_proof.json").is_file() for s in (0, 1, 2))]
    if not complete or args.require_full and len(complete) != 16:
        raise ValueError("Required completed three-seed clip8 groups missing")
    # Revoke the old audit before checking changed/new evidence. A failed refresh
    # must not leave a previous success marker beside partially rewritten summaries.
    (args.root / "validation.json").unlink(missing_ok=True)
    stored, points, paired, rows, deltas, checks = {}, {}, {}, [], [], []
    for model, mode, device in complete:
        for frames in (16, 8):
            runs = []
            for seed in (0, 1, 2):
                condition = Path(model) / mode / device / f"seed{seed}"
                folder = args.source / condition if frames == 16 else args.root / "T8" / condition
                proof = None
                if frames == 8:
                    proof = audit_clip8_source(folder, args.local / condition, args.cache / model / mode,
                                              manifest, model, mode, device, seed, args.manifest)
                    with np.load(args.local / condition / "memory.npz", allow_pickle=False) as bank:
                        if (bank["mean"].shape != (1024,) or bank["components"].shape != (256, 1024)
                                or bank["prototypes"].shape != (16, 128, 256)
                                or any(not np.isfinite(bank[n]).all() for n in ["mean", "components", "prototypes"])):
                            raise ValueError("Actual refitted bank geometry differs")
                        normal = json.loads((folder / "normal_fit.json").read_text())
                        np.testing.assert_allclose([float(bank["temperature"]), float(bank["cycle_length"])],
                                                   [normal["temperature"], normal["cycle_length_fit_median"]], rtol=0, atol=0)
                run, check = audit_condition(folder, manifest, args.data_root, model, mode, device, seed, frames, proof)
                checks.append(check); runs.append(run)
            for run in runs[1:]:
                check_pair(runs[0], run)
            key = (model, mode, device, f"T{frames}")
            values, rejected = bootstrap(runs)
            point = np.mean([statistic(run, list(run)) for run in runs], axis=0)
            low, high = np.quantile(values, [.025, .975], axis=0)
            stored[key], points[key] = (runs, values), point
            paired[key] = bootstrap_draws(runs, seed=np.random.SeedSequence([2026, int(device[1:])]))
            labels = np.concatenate([v[1] for v in runs[0].values()])
            rows.append({"backbone": model, "mode": mode, "device": device, "clip_frames": frames,
                         "variant": f"T{frames}", "score_variant": "P3", "seeds": 3,
                         "test_videos": len(runs[0]), "frames": len(labels), "anomaly_frames": int(labels.sum()),
                         "auroc_mean": float(point[0]), "auroc_ci_low": float(low[0]), "auroc_ci_high": float(high[0]),
                         "ap_mean": float(point[1]), "ap_ci_low": float(low[1]), "ap_ci_high": float(high[1]),
                         "bootstrap_draws": 1000, "bootstrap_rejected": rejected})
        a, b = (model, mode, device, "T16"), (model, mode, device, "T8")
        for old, new in zip(stored[a][0], stored[b][0]):
            check_pair(old, new)
        deltas.append(difference(model, mode, "T8_minus_T16", points[b] - points[a], stored[b][1] - stored[a][1], device))
    macro = macro_four_devices(stored)
    for row in macro["results"]:
        row.update(clip_frames=int(row["variant"][1:]), score_variant="P3")
    for model in MODELS:
        for mode in MODES:
            if all((model, mode, d, "T8") in points for d in DEVICES):
                delta = np.mean([paired[model, mode, d, "T8"] - paired[model, mode, d, "T16"] for d in DEVICES], axis=0)
                point = np.mean([points[model, mode, d, "T8"] - points[model, mode, d, "T16"] for d in DEVICES], axis=0)
                macro["paired_deltas"].append(difference(model, mode, "T8_minus_T16", point, delta))
    scope = "P3; mean of three fixed seed metrics; shared t=19..N-8/GT mask; whole-video bootstrap1000; per-context normal head/PCA/bank/temperature/MAD/q99 refit; no test selection or runtime claim"
    write(args.root / "device_summary", {"scope": scope, "results": rows, "paired_deltas": deltas})
    write(args.root / "macro_summary", macro)
    validation = {"status": "passed", "matrix_complete": len(complete) == 16, "completed_groups": len(complete),
                  "thresholds_checked": len(checks), "clip8_seed_conditions_checked": len(checks) // 2,
                  "excluded_groups": [list(key) for key in required if key not in complete], "checks": checks,
                  "device_summary_sha256": digest(args.root / "device_summary.json"),
                  "macro_summary_sha256": digest(args.root / "macro_summary.json"),
                  "verifier_sha256": digest(__file__), "manifest_sha256": digest(args.manifest),
                  "shared_bootstrap_sha256": digest(Path(__file__).with_name("summarize_experiments.py")),
                  "bootstrap_metric_sha256": digest("src/ipad_jepa/bootstrap_metrics.py"), "scope": scope,
                  "clip8_gpu_smoke_sha256": {m: digest(f"results/setup/{m}_clip8_smoke.json")
                                             for m in {key[0] for key in complete}},
                  "audit_limits": "Checks actual cache metadata/targets/shapes, head and bank SHA/geometry, CE selection, CSV phase/time/calibration/alarms and actual GT. Does not recompute encoder features, GPU distances, PCA fitting or temperature distance sample."}
    (args.root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
