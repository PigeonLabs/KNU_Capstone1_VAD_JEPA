"""Independently audit B online results and paired frozen L references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ipad_jepa.backbone_audit import check_small_cache, check_phase_selection, check_bank, check_pins
from ipad_jepa.memory import balanced_counts
from summarize_clip_ablation import audit_condition
from summarize_experiments import check_pair, statistic, bootstrap, bootstrap_draws, macro_four_devices
from summarize_neighbour_ablation import digest, read, write, difference

MODELS = ("dinov3-b", "vjepa21-b")
DEVICES = ("R01", "R02", "R03", "R04")
MODULES = ("small_features", "small_backbones", "small_data", "small_train_phase", "small_experiment",
           "backbones", "features", "temporal", "audit", "cache_data", "train_phase", "experiment",
           "memory", "torch_memory", "scoring", "alignment")
IDENTITY = ("backbone", "mode", "weights_sha256", "upstream_commit", "adapter_sha256", "reader_sha256",
            "base_reader_sha256", "temporal_code_sha256", "image_size", "clip_frames", "fit_stride",
            "feature_dimension", "preprocessing", "feature_dtype")


def audit_source(folder, local, cache, manifest, model, device, seed, manifest_path):
    proof = json.loads((folder / "small_source_proof.json").read_text())
    normal = json.loads((folder / "normal_fit.json").read_text())
    phase = json.loads((folder / "phase_training.json").read_text())
    expected = {"backbone": model, "mode": "online", "device": device, "seed": seed}
    if (proof.get("status") != "completed_small_condition" or proof.get("clip_frames") != 16
            or any(proof.get(k) != v for k, v in expected.items())
            or normal.get("feature_dimension") != 768
            or proof.get("paired_large_reference") != f"results/stage02/{model.replace('-b', '-l')}/online/{device}/seed{seed}"):
        raise ValueError("Completed B condition or paired L reference differs")
    required = {"scripts/run_small_matrix.py", str(manifest_path), "configs/experiment_matrix.yaml"}
    required.update(f"src/ipad_jepa/{name}.py" for name in MODULES)
    required.update(f"results/setup/{m}_smoke.json" for m in MODELS)
    pins = check_pins(proof["source_sha256"], required, digest)
    tests = [r for r in manifest if r["device"] == device and r["partition"] == "testing"]
    outputs = {"metrics.json", "normal_fit.json", "normal_calibration.csv", "phase_training.json", "phase_training.csv"}
    outputs.update(f"{v}/{r['sequence']}.csv" for v in ["P0", "P1", "P2", "P3"] for r in tests)
    if set(proof["outputs_sha256"]) != outputs or any(digest(folder / name) != sha for name, sha in proof["outputs_sha256"].items()):
        raise ValueError("B completed public output inventory or checksum differs")
    for name in ["phase_training.json", "phase_training.csv"]:
        if digest(local / name) != digest(folder / name):
            raise ValueError("B phase logs differ from actual selected run")
    if (digest(local / "phase_head.pt") != proof["phase_head_sha256"]
            or digest(local / "memory.npz") != proof["memory_sha256"]):
        raise ValueError("Actual B head or bank checksum differs")
    smoke_path = Path(f"results/setup/{model}_smoke.json")
    smoke = json.loads(smoke_path.read_text())
    if (smoke.get("status") != "strict_load_and_real_clip_passed" or smoke.get("model") != model
            or smoke.get("mode") != "online" or smoke.get("future_frames_used") is not False
            or smoke.get("input_frame_ids") != list(range(16)) or smoke.get("target_frame") != 15
            or smoke.get("trainable_parameters") != 0 or smoke.get("local_shape") != [1, 576, 768]
            or smoke.get("global_shape") != [1, 768]
            or smoke.get("adapter_sha256") != digest("src/ipad_jepa/small_backbones.py")
            or smoke.get("reader_sha256") != digest("src/ipad_jepa/features.py")
            or smoke.get("smoke_code_sha256") != digest("scripts/smoke_small_backbone.py")):
        raise ValueError("Actual strict B GPU smoke/online input evidence differs")
    selected = {(r["partition"], r["sequence"]): r for r in manifest if r["device"] == device
                and r.get("split", "test") in {"fit", "calibration", "test"}}
    evidence = {(r["partition"], r["sequence"]): r for r in proof["cache_metadata"]}
    if len(evidence) != len(proof["cache_metadata"]) or set(evidence) != set(selected):
        raise ValueError("B cache sequence inventory differs")
    fingerprints, identity, fit_ids = [], None, []
    counts = {"fit": 0, "calibration": 0}
    for key, row in selected.items():
        base = cache / device / key[0] / key[1]
        meta = json.loads((base / "meta.json").read_text()); spec = meta["spec"]
        current = evidence[key]
        if (current["spec"] != spec or current["fingerprint"] != meta["fingerprint"]
                or digest(base / "meta.json") != current["metadata_sha256"]
                or digest(base / "targets.npy") != current["targets_sha256"]
                or spec["reader_sha256"] != digest("src/ipad_jepa/small_features.py")
                or spec["temporal_code_sha256"] != digest("src/ipad_jepa/temporal.py")):
            raise ValueError("Actual B cache fingerprint or producer differs")
        ids = np.load(base / "targets.npy", allow_pickle=False)
        patch = np.load(base / "patch.npy", allow_pickle=False, mmap_mode="r")
        context = np.load(base / "global.npy", allow_pickle=False, mmap_mode="r")
        check_small_cache(meta, row, smoke, ids, patch, context)
        current_identity = {k: spec[k] for k in IDENTITY}
        if identity is not None and identity != current_identity:
            raise ValueError("Mixed B encoder/cache identities")
        identity = current_identity
        if spec["split"] in counts:
            fingerprints.append(meta["fingerprint"]); counts[spec["split"]] += len(ids)
        if spec["split"] == "fit":
            fit_ids.append((row, ids))
    check_phase_selection(phase, read(folder / "phase_training.csv"),
                          {**expected, "fit_clips": counts["fit"], "calibration_clips": counts["calibration"],
                           "base_training_sha256": digest("src/ipad_jepa/train_phase.py")})
    if (sorted(phase["cache_fingerprints"]) != sorted(fingerprints)
            or sorted(normal["normal_cache_fingerprints"]) != sorted(fingerprints)
            or normal["cache_identity"] != identity or normal["phase_checkpoint_sha256"] != proof["phase_head_sha256"]
            or normal["phase_selected_epoch"] != phase["selected_epoch"]
            or normal["code_sha256"] != digest("src/ipad_jepa/small_experiment.py")
            or normal["base_experiment_sha256"] != digest("src/ipad_jepa/experiment.py")
            or normal["memory_code_sha256"] != digest("src/ipad_jepa/memory.py")):
        raise ValueError("B normal head/memory source differs")
    # Reproduce all normal video/bin quotas, including the RNG advance from
    # actual no-replacement candidate sampling. No raw feature values are reused.
    rng = np.random.default_rng(seed); sampling = []
    for b in range(16):
        capacities = np.array([int((np.floor(ids / r["frames"] * 16) == b).sum()) * 576 for r, ids in fit_ids])
        quotas = balanced_counts(capacities, 10000, rng)
        for cap, count in zip(capacities, quotas):
            if count:
                rng.choice(int(cap), int(count), replace=False)
        sampling.append({"phase_bin": b, "tokens": int(quotas.sum()),
                         "per_video": {r["sequence"]: int(n) for (r, _), n in zip(fit_ids, quotas)}})
    if normal["candidate_sampling"] != sampling:
        raise ValueError("B normal-only candidate quotas differ")
    with np.load(local / "memory.npz", allow_pickle=False) as bank:
        check_bank(bank, normal)
    return {"source_proof_sha256": digest(folder / "small_source_proof.json"), "source_sha256": pins,
            "phase_head_sha256": proof["phase_head_sha256"], "memory_sha256": proof["memory_sha256"],
            "smoke_sha256": digest(smoke_path), "cache_sequences_checked": len(selected),
            "normal_fit_clips": counts["fit"], "normal_calibration_clips": counts["calibration"],
            "feature_dimension": 768, "normal_candidate_bins_checked": len(sampling)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in [("root", "results/stage05/ablations/backbone_size"), ("source", "results/stage02"),
                          ("cache", "artifacts/features_small"), ("local", "artifacts/runs_small"),
                          ("manifest", "results/stage00/manifest.json"), ("data-root", "../IPAD_dataset/IPAD_dataset")]:
        p.add_argument("--" + name, type=Path, default=Path(default))
    p.add_argument("--require-full", action="store_true")
    args = p.parse_args(); manifest = json.loads(args.manifest.read_text())["sequences"]
    required = [(m, d) for m in MODELS for d in DEVICES]
    complete = [(m, d) for m, d in required if all((args.root / "B" / m / "online" / d / f"seed{s}" / "small_source_proof.json").is_file() for s in (0, 1, 2))]
    if not complete or args.require_full and len(complete) != 8:
        raise ValueError("Required completed three-seed B groups missing")
    (args.root / "validation.json").unlink(missing_ok=True)
    stored, points, paired, rows, deltas, checks = {}, {}, {}, [], [], []
    for small, device in complete:
        family = small.replace("-b", "-l")
        for size in ("L", "B"):
            runs = []
            for seed in (0, 1, 2):
                model = family if size == "L" else small
                condition = Path(model) / "online" / device / f"seed{seed}"
                folder = args.source / condition if size == "L" else args.root / "B" / condition
                source = None
                if size == "B":
                    source = audit_source(folder, args.local / condition, args.cache / model / "online",
                                          manifest, model, device, seed, args.manifest)
                run, check = audit_condition(folder, manifest, args.data_root, model, "online", device, seed, 16)
                if source is not None: check.update(source)
                check["size"] = size; checks.append(check); runs.append(run)
            for run in runs[1:]:check_pair(runs[0], run)
            key = (family, "online", device, size)
            values, rejected = bootstrap(runs)
            point = np.mean([statistic(run, list(run)) for run in runs], axis=0)
            low, high = np.quantile(values, [.025, .975], axis=0)
            stored[key], points[key] = (runs, values), point
            paired[key] = bootstrap_draws(runs, seed=np.random.SeedSequence([2026, int(device[1:])]))
            labels = np.concatenate([v[1] for v in runs[0].values()])
            rows.append({"backbone": family, "actual_backbone": family if size == "L" else small,
                         "mode": "online", "device": device, "variant": size, "score_variant": "P3", "seeds": 3,
                         "feature_dimension": 1024 if size == "L" else 768,
                         "test_videos": len(runs[0]), "frames": len(labels), "anomaly_frames": int(labels.sum()),
                         "auroc_mean": float(point[0]), "auroc_ci_low": float(low[0]), "auroc_ci_high": float(high[0]),
                         "ap_mean": float(point[1]), "ap_ci_low": float(low[1]), "ap_ci_high": float(high[1]),
                         "bootstrap_draws": 1000, "bootstrap_rejected": rejected})
        old, new = (family, "online", device, "L"), (family, "online", device, "B")
        for a, b in zip(stored[old][0], stored[new][0]):check_pair(a, b)
        deltas.append(difference(family, "online", "B_minus_L", points[new] - points[old], stored[new][1] - stored[old][1], device))
    macro = macro_four_devices(stored)
    for family in [m.replace("-b", "-l") for m in MODELS]:
        if all((family, "online", d, "B") in points for d in DEVICES):
            values = np.mean([paired[family, "online", d, "B"] - paired[family, "online", d, "L"] for d in DEVICES], axis=0)
            point = np.mean([points[family, "online", d, "B"] - points[family, "online", d, "L"] for d in DEVICES], axis=0)
            macro["paired_deltas"].append(difference(family, "online", "B_minus_L", point, values))
    scope = "Frozen online B vs L P3; separately refitted normal head/PCA/bank/temperature/MAD/q99; fixed three-seed means; shared t=19..N-8/GT; original-video paired bootstrap1000; equal-device Macro4 only with all four devices; different pretrained checkpoint/width; no runtime claim"
    write(args.root / "device_summary", {"scope": scope, "results": rows, "paired_deltas": deltas})
    write(args.root / "macro_summary", macro)
    validation = {"status": "passed", "matrix_complete": len(complete) == 8, "completed_groups": len(complete),
                  "small_seed_conditions_checked": len(checks) // 2, "thresholds_checked": len(checks),
                  "excluded_groups": [list(k) for k in required if k not in complete], "checks": checks,
                  "device_summary_sha256": digest(args.root / "device_summary.json"), "macro_summary_sha256": digest(args.root / "macro_summary.json"),
                  "verifier_sha256": digest(__file__), "source_audit_sha256": digest("src/ipad_jepa/backbone_audit.py"),
                  "shared_trace_audit_sha256": digest(Path(__file__).with_name("summarize_clip_ablation.py")),
                  "shared_bootstrap_sha256": digest(Path(__file__).with_name("summarize_experiments.py")),
                  "manifest_sha256": digest(args.manifest), "scope": scope,
                  "audit_limits": "Replays normal calibration/time/GT/masks/alarms/metrics; checks actual cache metadata, target arrays, shapes, finite context features, CE-selected head and bank hashes/geometry plus normal candidate quotas. Does not independently re-encode images, recompute all patch values, train heads/PCA/k-center or replay GPU feature distances/temperature sampling. No real-time throughput measurement."}
    (args.root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({k:v for k,v in validation.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
