"""Audit actual paired dense-fit sources, spatial means, calibration and GT before aggregation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ipad_jepa.alignment import align
from ipad_jepa.representation import SEEDS, candidate_plan, capacity, fit_targets
from summarize_clip_ablation import audit_calibration, audit_trace, targets
from summarize_experiments import load_run, check_pair, statistic, bootstrap, bootstrap_draws, macro_four_devices
from summarize_neighbour_ablation import digest, read, write, difference

MODELS = ("dinov3-l", "vjepa21-l")
MODES = ("offline", "online")
DEVICES = ("R01", "R02", "R03", "R04")
REPRESENTATIONS = ("patch", "global_mean")
MODULES = ("representation", "representation_features", "representation_data", "representation_ablation",
           "backbones", "features", "temporal", "audit", "cache_data", "experiment", "memory", "torch_memory", "scoring", "alignment")


def audit_dense(root, manifest, model, mode, device):
    info = json.loads((root / "dense_fit.json").read_text())
    identity = info["identity"]
    fit = [r for r in manifest if r["device"] == device and r.get("split") == "fit"]
    if (info["status"] != "complete_dense_normal_fit" or info["device"] != device or info["seeds"] != list(SEEDS)
            or identity["backbone"] != model or identity["mode"] != mode or identity["fit_stride"] != 1
            or identity["clip_frames"] != 16 or identity["image_size"] != 384
            or identity["feature_dtype"] != "float16 from BF16 inference"
            or identity["preprocessing"] != "RGB full-frame PIL bilinear resize; ImageNet mean/std"
            or info["capacity"] != capacity(fit, mode, 1)):
        raise ValueError("Dense normal fit differs from required paired protocol")
    sources = {"reader_sha256": "representation_features", "sampling_code_sha256": "representation",
               "base_reader_sha256": "features", "adapter_sha256": "backbones"}
    if any(identity[key] != digest(f"src/ipad_jepa/{name}.py") for key, name in sources.items()):
        raise ValueError("Dense fit producer changed")
    required = {f"fit/{r['sequence']}/{name}.npy" for r in fit for name in ["targets", "global", "mean"]}
    required |= {f"seed{s}_{name}.npy" for s in SEEDS for name in ["coordinates", "phases", "patch"]}
    if set(info["outputs_sha256"]) != required:
        raise ValueError("Dense normal output inventory differs")
    for name, sha in info["outputs_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError(f"Actual dense normal output changed: {name}")
    inventory = [{"sequence": r["sequence"], "frames": r["frames"], "clips": len(fit_targets(r["frames"], mode)),
                  "frame_names_sha256": r["names_sha256"], "frame_content_sha256": r["frames_content_sha256"], "padding": mode == "online"}
                 for r in fit]
    if [{k: r.get(k) for k in inventory[0]} for r in info["fit_inventory"]] != inventory:
        raise ValueError("Actual dense normal inventory differs from audited manifest")
    for r in fit:
        ids = fit_targets(r["frames"], mode)
        np.testing.assert_array_equal(np.load(root / "fit" / r["sequence"] / "targets.npy", allow_pickle=False), ids)
        for name, shape, dtype in [("global", (len(ids), 1024), np.float16), ("mean", (len(ids), 1, 1024), np.float32)]:
            value = np.load(root / "fit" / r["sequence"] / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError("Invalid actual compact dense normal arrays")
    for seed in SEEDS:
        coordinates, phases, sampling = candidate_plan(fit, mode, seed)
        np.testing.assert_array_equal(np.load(root / f"seed{seed}_coordinates.npy", allow_pickle=False), coordinates)
        np.testing.assert_array_equal(np.load(root / f"seed{seed}_phases.npy", allow_pickle=False), phases)
        if info["candidate_sampling"][str(seed)] != sampling:
            raise ValueError("Stored balanced normal candidate quotas changed")
        values = np.load(root / f"seed{seed}_patch.npy", mmap_mode="r", allow_pickle=False)
        if values.shape != (len(coordinates), 1024) or values.dtype != np.float16:
            raise ValueError("Invalid actual normal patch candidates")
        for start in range(0, len(values), 4096):
            if not np.isfinite(values[start:start + 4096]).all():
                raise ValueError("Nonfinite stored normal patch candidates")
        for video, proof in enumerate(info["fit_inventory"]):
            if proof["candidate_rows_written"][str(seed)] != int((coordinates[:, 0] == video).sum()):
                raise ValueError("Not all planned observed normal candidate coordinates were written")
    return info, fit


def audit_mean_array(patch_path, mean_path, count):
    """Independently recompute every target-local spatial mean in bounded CPU chunks."""
    patches = np.load(patch_path, mmap_mode="r", allow_pickle=False)
    means = np.load(mean_path, mmap_mode="r", allow_pickle=False)
    if (patches.shape != (count, 576, 1024) or patches.dtype != np.float16
            or means.shape != (count, 1, 1024) or means.dtype != np.float32):
        raise ValueError("Actual spatial mean or parent patch geometry differs")
    for start in range(0, count, 32):
        chunk = patches[start:start + 32]
        if not np.isfinite(chunk).all():
            raise ValueError("Nonfinite parent local patches")
        # Deliberately do not call the producer's pooling helper.
        expected = np.mean(chunk, axis=1, dtype=np.float32)[:, None, :]
        np.testing.assert_array_equal(means[start:start + 32], expected)
    return count


def audit_cache_sources(folder, dense, original, manifest, model, mode, device, representation):
    proof = json.loads((folder / "cache_sources.json").read_text())
    selected = [r for r in manifest if r["device"] == device and
                (r.get("split") == "calibration" or r["partition"] == "testing")]
    expected = {(r["partition"], r["sequence"]): r for r in selected}
    sources = proof["original_calibration_test"]
    keys = [(s["partition"], s["sequence"]) for s in sources]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("Original calibration/test source inventory differs")
    if proof["dense_fit_metadata_sha256"] != digest(dense / "dense_fit.json"):
        raise ValueError("Dense-fit source differs from reused query caches")
    views = []
    dense_info = json.loads((dense / "dense_fit.json").read_text())
    for source in sources:
        key = (source["partition"], source["sequence"]); row = expected[key]
        parent = original / device / key[0] / key[1]
        meta = json.loads((parent / "meta.json").read_text()); spec = meta["spec"]
        import hashlib
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if (meta.get("status") != "complete" or meta.get("fingerprint") != fingerprint or source["fingerprint"] != fingerprint
                or source["spec"] != spec or digest(parent / "meta.json") != source["metadata_sha256"]
                or digest(parent / "targets.npy") != source["targets_sha256"]):
            raise ValueError("Actual original query cache provenance differs")
        expected_spec = {"device": device, "sequence": row["sequence"], "partition": row["partition"], "frames": row["frames"],
                         "split": row.get("split", "test"), "frame_names_sha256": row["names_sha256"],
                         "frame_content_sha256": row["frames_content_sha256"], "fit_stride": 4}
        if any(spec.get(k) != v for k, v in expected_spec.items()):
            raise ValueError("Reused original query cache differs from audited manifest/reader protocol")
        identity = dense_info["identity"]
        for k in ["backbone", "mode", "weights_sha256", "upstream_commit", "adapter_sha256", "image_size", "clip_frames", "preprocessing", "feature_dtype"]:
            if spec[k] != identity[k]:
                raise ValueError("Dense fit and original query encoder semantics differ")
        if spec["reader_sha256"] != identity["base_reader_sha256"]:
            raise ValueError("Original frozen reader differs from paired extraction base")
        ids = np.load(parent / "targets.npy", allow_pickle=False)
        np.testing.assert_array_equal(ids, targets(row["frames"], mode, 16))
        pooled = np.load(parent / "global.npy", mmap_mode="r", allow_pickle=False)
        if pooled.shape != (len(ids), 1024) or pooled.dtype != np.float16 or not np.isfinite(pooled).all():
            raise ValueError("Invalid shared original phase-head input")
        if representation == "global_mean":
            view = dense / "mean_views" / key[0] / key[1]
            actual = json.loads((view / "mean_source.json").read_text())
            identity_view = {"source_fingerprint": fingerprint, "source_metadata_sha256": source["metadata_sha256"],
                             "source_targets_sha256": source["targets_sha256"],
                             "pooling_code_sha256": digest("src/ipad_jepa/representation.py"),
                             "derivation_code_sha256": digest("src/ipad_jepa/representation_data.py"),
                             "representation": "FP32 spatial mean of FP16 target-local patches", "rows": len(ids)}
            if (actual != source["mean_view"] or actual["status"] != "complete_mean_view" or actual["identity"] != identity_view
                    or actual["mean_sha256"] != digest(view / "mean.npy")):
                raise ValueError("Actual spatial-mean query provenance differs")
            views.append((parent / "patch.npy", view / "mean.npy", len(ids)))
    return views


def audit_condition(folder, runs, dense, dense_info, fit_rows, original, manifest, data_root,
                    model, mode, device, seed, representation, manifest_path):
    proof = json.loads((folder / "representation_source_proof.json").read_text())
    normal = json.loads((folder / "normal_fit.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    expected = {"backbone": model, "mode": mode, "device": device, "seed": seed, "representation": representation}
    if (proof.get("status") != "completed_representation_condition" or metrics.get("status") != "complete_device_evaluation"
            or normal.get("status") != "normal_fit_and_calibration_complete"
            or any(any(obj.get(k) != v for k, v in expected.items()) for obj in [proof, normal, metrics])):
        raise ValueError("Completed representation condition identity differs")
    required_sources = {"scripts/run_representation_matrix.py", str(manifest_path), "configs/experiment_matrix.yaml"}
    required_sources.update(f"src/ipad_jepa/{name}.py" for name in MODULES)
    actual_sources = {}
    for name, sha in proof["source_sha256"].items():
        if Path(name).is_absolute() and Path(name).name == "run_representation_matrix.py":
            name = "scripts/run_representation_matrix.py"
        if name in actual_sources or name not in required_sources or digest(name) != sha:
            raise ValueError("Pinned representation source changed or duplicated")
        actual_sources[name] = sha
    if set(actual_sources) != required_sources:
        raise ValueError("Required representation source inventory is incomplete")
    tests = {r["sequence"]: r for r in manifest if r["device"] == device and r["partition"] == "testing"}
    outputs = {"metrics.json", "normal_fit.json", "normal_calibration.csv", "phase_training.json", "phase_training.csv",
               "dense_fit_source.json", "cache_sources.json"}
    outputs.update(f"{variant}/{sequence}.csv" for variant in ["P0", "P1", "P2", "P3"] for sequence in tests)
    if set(proof["outputs_sha256"]) != outputs or any(digest(folder / name) != sha for name, sha in proof["outputs_sha256"].items()):
        raise ValueError("Actual completed public output inventory/checksums differ")
    if json.loads((folder / "dense_fit_source.json").read_text()) != dense_info or normal["dense_fit_source_sha256"] != digest(dense / "dense_fit.json"):
        raise ValueError("Copied dense normal fit source differs")
    local = runs / model / mode / device / f"seed{seed}"
    phase_path = local / "shared_phase"
    phase = json.loads((folder / "phase_training.json").read_text())
    calibration = [r for r in manifest if r["device"] == device and r.get("split") == "calibration"]
    fingerprints = [json.loads((original / device / "training" / r["sequence"] / "meta.json").read_text())["fingerprint"] for r in calibration]
    expected_phase = {"backbone": model, "mode": mode, "device": device, "seed": seed, "status": "complete", "epochs": 20,
                      "shared_between_representations": True, "fit_stride": 1, "fit_clips": sum(len(fit_targets(r["frames"], mode)) for r in fit_rows),
                      "calibration_clips": sum(len(targets(r["frames"], mode, 16)) for r in calibration),
                      "fit_input_sha256": digest(dense / "dense_fit.json"), "normal_calibration_cache_fingerprints": fingerprints}
    if any(phase.get(k) != v for k, v in expected_phase.items()):
        raise ValueError("Shared phase head training provenance differs")
    for name in ["phase_training.json", "phase_training.csv"]:
        if digest(phase_path / name) != digest(folder / name):
            raise ValueError("Published phase training differs from the actual private run")
    curves = read(folder / "phase_training.csv")
    if len(curves) != 20 or [int(r["epoch"]) for r in curves] != list(range(1, 21)):
        raise ValueError("Twenty full phase epochs required")
    if not np.isfinite([[float(r[k]) for k in ["train_ce", "normal_calibration_ce", "normal_calibration_circular_mae"]] for r in curves]).all():
        raise ValueError("Nonfinite normal phase training curve")
    chosen = min(curves, key=lambda r: float(r["normal_calibration_ce"]))
    sha = digest(phase_path / "phase_head.pt")
    if (int(chosen["epoch"]) != phase["selected_epoch"] or float(chosen["normal_calibration_ce"]) != phase["best_normal_calibration_ce"]
            or proof["phase_head_sha256"] != sha or normal["phase_checkpoint_sha256"] != sha
            or normal["phase_selected_epoch"] != phase["selected_epoch"]):
        raise ValueError("Actual selected shared head differs from minimum normal calibration CE")
    queries = 576 if representation == "patch" else 1
    coordinates, _, sampling = candidate_plan(fit_rows, mode, seed, queries=queries)
    if (normal["dense_fit_identity"] != dense_info["identity"] or normal["fit_stride_both_representations"] != 1
            or (normal["bins"], normal["prototypes_per_bin"], normal["total_prototypes"], normal["pca_dimensions"]) != (16, 128, 2048, 256)
            or normal["queries_per_target"] != queries or normal["pca_sample_limit"] != 50000
            or normal["pca_actual_samples"] != min(50000, len(coordinates)) or normal["candidate_sampling"] != sampling
            or normal["normal_calibration_cache_fingerprints"] != fingerprints
            or normal["original_calibration_cache_identity"] != dense_info["original_cache_identity"]
            or normal["temperature_source"] != "Normal calibration, never fit or test"
            or normal["temperature_samples"] != min(50000, expected_phase["calibration_clips"] * queries)
            or not np.isfinite(normal["temperature"]) or normal["temperature"] < 1e-6
            or normal["code_sha256"] != digest("src/ipad_jepa/representation_ablation.py")):
        raise ValueError("Representation memory or normal sample counts differ")
    bank_path = local / representation / "memory.npz"
    if digest(bank_path) != proof["memory_sha256"]:
        raise ValueError("Actual private representation bank changed")
    cycle = float(np.median([r["frames"] for r in fit_rows]))
    if normal["cycle_length_fit_median"] != cycle:
        raise ValueError("Temporal cycle was not derived from normal fit videos")
    with np.load(bank_path, allow_pickle=False) as bank:
        if (bank["mean"].shape != (1024,) or bank["components"].shape != (256, 1024) or bank["prototypes"].shape != (16, 128, 256)
                or any(not np.isfinite(bank[k]).all() for k in ["mean", "components", "prototypes"])
                or float(bank["temperature"]) != normal["temperature"] or float(bank["cycle_length"]) != cycle):
            raise ValueError("Actual private representation bank geometry/calibration differs")
    views = audit_cache_sources(folder, dense, original, manifest, model, mode, device, representation)
    params = normal["calibration"]["P3"]
    count = audit_calibration(folder, manifest, device, mode, 16, cycle, params)
    if count != normal["calibration_valid_frames"]:
        raise ValueError("Normal common frame count differs")
    if {p.stem for p in (folder / "P3").glob("*.csv")} != set(tests):
        raise ValueError("Actual test inventory differs")
    annotations = {}
    for sequence, row in tests.items():
        label_path = data_root / row["label_file"]
        if digest(label_path) != row["label_sha256"]:
            raise ValueError("Actual raw test GT changed")
        label, known, _ = align(np.load(label_path, allow_pickle=False), row["frames"])
        audit_trace(read(folder / "P3" / f"{sequence}.csv"), row, mode, 16, cycle, params, label, known)
        annotations[sequence] = row["label_sha256"]
    run = load_run(folder, "P3", metrics)
    return run, {**expected, "normal_common_frames": count, "threshold": params["threshold"], "temperature_samples": normal["temperature_samples"],
                 "pca_actual_samples": normal["pca_actual_samples"], "phase_checkpoint_sha256": sha, "memory_sha256": proof["memory_sha256"],
                 "normal_fit_sha256": digest(folder / "normal_fit.json"), "actual_annotation_sha256": annotations,
                 "source_proof_sha256": digest(folder / "representation_source_proof.json")}, views


def audit_pair(patch, mean):
    before = json.loads((patch / "normal_fit.json").read_text()); after = json.loads((mean / "normal_fit.json").read_text())
    for key in ["seed", "backbone", "mode", "device", "dense_fit_source_sha256", "phase_checkpoint_sha256", "phase_selected_epoch",
                "cycle_length_fit_median", "normal_calibration_cache_fingerprints", "original_calibration_cache_identity"]:
        if before[key] != after[key]:
            raise ValueError("Paired representations do not share the same normal data and phase head")
    for name in ["phase_training.json", "phase_training.csv"]:
        if digest(patch / name) != digest(mean / name):
            raise ValueError("Paired representations have different phase training")
    calibration = [read(p / "normal_calibration.csv") for p in [patch, mean]]
    if [(r["sequence"], r["frame"], r["valid"]) for r in calibration[0]] != [(r["sequence"], r["frame"], r["valid"]) for r in calibration[1]]:
        raise ValueError("Paired calibration target/mask differs")
    for name in ["predicted_phase", "relative_phase", "time_raw"]:
        np.testing.assert_allclose([float(r[name]) for r in calibration[0]], [float(r[name]) for r in calibration[1]], rtol=0, atol=0, equal_nan=True)
    files = sorted((patch / "P3").glob("*.csv"))
    if {p.name for p in files} != {p.name for p in (mean / "P3").glob("*.csv")}:
        raise ValueError("Paired test inventory differs")
    for file in files:
        traces = [read(file), read(mean / "P3" / file.name)]
        for name in ["frame", "label", "valid", "inference_valid"]:
            if [r[name] for r in traces[0]] != [r[name] for r in traces[1]]:
                raise ValueError("Paired test target, GT or inference mask differs")
        for name in ["phase", "time_raw"]:
            np.testing.assert_allclose([float(r[name]) for r in traces[0]], [float(r[name]) for r in traces[1]], rtol=0, atol=0, equal_nan=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("root", "results/stage05/ablations/representation"), ("dense-fit", "artifacts/representation_dense"),
                          ("runs", "artifacts/runs_representation"), ("original-cache", "artifacts/features"),
                          ("manifest", "results/stage00/manifest.json"), ("data-root", "../IPAD_dataset/IPAD_dataset")]:
        parser.add_argument("--" + name, type=Path, default=Path(default))
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())["sequences"]
    required = [(m, mode, d) for m in MODELS for mode in MODES for d in DEVICES]
    complete = [key for key in required if all((args.root / rep / key[0] / key[1] / key[2] / f"seed{s}" / "representation_source_proof.json").is_file()
                                             for rep in REPRESENTATIONS for s in SEEDS)]
    if not complete or args.require_full and len(complete) != 16:
        raise ValueError("Required completed three-seed paired representation groups missing")
    (args.root / "validation.json").unlink(missing_ok=True)
    stored, points, paired, rows, deltas, checks, mean_checks = {}, {}, {}, [], [], [], []
    for model, mode, device in complete:
        dense = args.dense_fit / model / mode / device
        info, fit = audit_dense(dense, manifest, model, mode, device)
        original = args.original_cache / model / mode
        checked_views = set()
        for rep in REPRESENTATIONS:
            runs = []
            for seed in SEEDS:
                condition = Path(model) / mode / device / f"seed{seed}"
                folder = args.root / rep / condition
                run, check, views = audit_condition(folder, args.runs, dense, info, fit, original, manifest, args.data_root,
                                                    model, mode, device, seed, rep, args.manifest)
                checks.append(check); runs.append(run)
                for patch_path, mean_path, count in views:
                    if str(mean_path) not in checked_views:
                        observed = audit_mean_array(patch_path, mean_path, count)
                        checked_views.add(str(mean_path))
                        mean_checks.append({"backbone": model, "mode": mode, "device": device, "target_count": observed,
                                            "mean_sha256": digest(mean_path), "pooling": "Independent FP32 mean over all 576 FP16 local patches; exact equality"})
                if rep == "global_mean":
                    audit_pair(args.root / "patch" / condition, folder)
            for run in runs[1:]:
                check_pair(runs[0], run)
            key = (model, mode, device, rep)
            draws, rejected = bootstrap(runs)
            point = np.mean([statistic(r, list(r)) for r in runs], axis=0)
            low, high = np.quantile(draws, [.025, .975], axis=0)
            stored[key], points[key] = (runs, draws), point
            paired[key] = bootstrap_draws(runs, seed=np.random.SeedSequence([2026, int(device[1:])]))
            labels = np.concatenate([r[1] for r in runs[0].values()])
            rows.append({"backbone": model, "mode": mode, "device": device, "representation": rep, "variant": rep, "score_variant": "P3", "seeds": 3,
                         "test_videos": len(runs[0]), "frames": len(labels), "anomaly_frames": int(labels.sum()),
                         "auroc_mean": float(point[0]), "auroc_ci_low": float(low[0]), "auroc_ci_high": float(high[0]),
                         "ap_mean": float(point[1]), "ap_ci_low": float(low[1]), "ap_ci_high": float(high[1]), "bootstrap_draws": 1000, "bootstrap_rejected": rejected})
        a, b = (model, mode, device, "patch"), (model, mode, device, "global_mean")
        for old, new in zip(stored[a][0], stored[b][0]):
            check_pair(old, new)
        deltas.append(difference(model, mode, "global_mean_minus_patch", points[b] - points[a], stored[b][1] - stored[a][1], device))
        print(f"representation group independently verified: {model}/{mode}/{device}", flush=True)
    macro = macro_four_devices(stored)
    for row in macro["results"]:
        row.update(representation=row["variant"], score_variant="P3")
    for model in MODELS:
        for mode in MODES:
            if all((model, mode, d, "global_mean") in points for d in DEVICES):
                delta = np.mean([paired[model, mode, d, "global_mean"] - paired[model, mode, d, "patch"] for d in DEVICES], axis=0)
                point = np.mean([points[model, mode, d, "global_mean"] - points[model, mode, d, "patch"] for d in DEVICES], axis=0)
                macro["paired_deltas"].append(difference(model, mode, "global_mean_minus_patch", point, delta))
    scope = "P3; both representations fit stride1/shared head/clip16/history5; three fixed seed metrics averaged; shared t=19..N-8/GT mask; whole-video bootstrap1000; per-representation PCA/memory/calibration; no runtime claim"
    write(args.root / "device_summary", {"scope": scope, "results": rows, "paired_deltas": deltas})
    write(args.root / "macro_summary", macro)
    validation = {"status": "passed", "matrix_complete": len(complete) == 16, "completed_groups": len(complete), "thresholds_checked": len(checks),
                  "shared_phase_pairs_checked": len(checks) // 2, "actual_mean_views_recomputed": len(mean_checks), "mean_checks": mean_checks,
                  "checks": checks, "excluded_groups": [list(k) for k in required if k not in complete],
                  "device_summary_sha256": digest(args.root / "device_summary.json"), "macro_summary_sha256": digest(args.root / "macro_summary.json"),
                  "verifier_sha256": digest(__file__), "shared_trace_audit_sha256": digest(Path(__file__).with_name("summarize_clip_ablation.py")),
                  "shared_bootstrap_sha256": digest(Path(__file__).with_name("summarize_experiments.py")), "manifest_sha256": digest(args.manifest), "scope": scope,
                  "audit_limits": "Checks actual dense samples/coords/hashes, parent query metadata/targets, full mean pooling, shared head/phase/time, bank geometry/hash, normal q99 and actual GT/metrics. Does not rerun encoders, phase training, PCA/k-center fitting or GPU distance/temperature sampling. Batch common-mask alarms do not prove streaming EOF/FIFO/runtime."}
    (args.root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({k: v for k, v in validation.items() if k not in {"checks", "mean_checks"}}, indent=2))


if __name__ == "__main__":
    main()
