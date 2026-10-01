"""Verify history ablations and aggregate paired whole-video uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from summarize_experiments import (bootstrap, bootstrap_draws, check_pair,
                                   load_run, macro_four_devices, statistic)

MODELS = ["dinov3-l", "vjepa21-l"]
MODES = ["offline", "online"]
DEVICES = ["R01", "R02", "R03", "R04"]
HISTORIES = [5, 15, 31]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delta_row(backbone, mode, comparison, point, draws, device=None):
    low, high = np.quantile(draws, [.025, .975], axis=0)
    row = {"backbone": backbone, "mode": mode, "comparison": comparison,
           "auroc_delta": float(point[0]), "auroc_delta_ci_low": float(low[0]), "auroc_delta_ci_high": float(high[0]),
           "ap_delta": float(point[1]), "ap_delta_ci_low": float(low[1]), "ap_delta_ci_high": float(high[1])}
    if device is not None:
        row["device"] = device
    return row


def write_summary(path, output):
    path.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    rows = output["results"]
    fields = [key for key in rows[0] if key != "device_counts"]
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/stage05/ablations/history"))
    parser.add_argument("--source", type=Path, default=Path("results/stage02"))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    args = parser.parse_args()
    completed = json.loads((args.root / "completion.json").read_text())
    if completed["status"] != "complete_history_matrix" or completed["evaluated_seed_conditions"] != 144:
        raise ValueError("All accepted history conditions must be complete")
    if (completed["source_summary_sha256"] != digest(args.source / "device_summary.json") or
        completed["manifest_sha256"] != digest(args.manifest) or
        completed["code_sha256"] != digest("src/ipad_jepa/history_ablation.py")):
        raise ValueError("Source summary, manifest or history implementation changed")
    manifest = json.loads(args.manifest.read_text())["sequences"]
    stored, stratified, points, references = {}, {}, {}, {}
    rows, deltas, checks = [], [], []
    proofs = {}
    for model in MODELS:
        for mode in MODES:
            for device in DEVICES:
                for history in HISTORIES:
                    runs = []
                    for seed in [0, 1, 2]:
                        folder = args.root / f"H{history}" / model / mode / device / f"seed{seed}"
                        original = args.source / model / mode / device / f"seed{seed}"
                        meta = json.loads((folder / "metrics.json").read_text())
                        normal = json.loads((folder / "normal_fit.json").read_text())
                        if (meta["status"] != "complete_device_evaluation" or meta["history"] != history or
                            (meta["backbone"], meta["mode"], meta["device"], meta["seed"]) != (model, mode, device, seed)):
                            raise ValueError("History evaluation condition differs")
                        proof = json.loads((folder / "source_replay_check.json").read_text())
                        if (proof["status"] != "passed" or proof["alarm_mismatches"] != 0 or
                            normal["history"] != history or normal["shared_comparison_history"] != 31 or
                            normal["code_sha256"] != completed["code_sha256"] or
                            normal["temporal_code_sha256"] != digest("src/ipad_jepa/temporal.py")):
                            raise ValueError("Primary H5 replay failed")
                        for name, expected in proof["sources"].items():
                            if digest(original / name) != expected:
                                raise ValueError("Original published source changed")
                        proofs[model, mode, device, seed] = proof
                        calibration = {r["sequence"]: r for r in manifest
                                       if r["device"] == device and r.get("split") == "calibration"}
                        with (folder / "normal_calibration.csv").open() as stream:
                            cal_rows = list(csv.DictReader(stream))
                        with (original / "normal_calibration.csv").open() as stream:
                            original_normal = list(csv.DictReader(stream))
                        def inventory(values):
                            result = {(r["sequence"], int(r["frame"])): r for r in values}
                            if len(result) != len(values):
                                raise ValueError("Duplicate calibration target")
                            return result
                        before, after = inventory(original_normal), inventory(cal_rows)
                        if before.keys() != after.keys():
                            raise ValueError("Calibration targets changed")
                        for key in before:
                            for column in ["feature_raw", "predicted_phase", "relative_phase"]:
                                if float(before[key][column]) != float(after[key][column]):
                                    raise ValueError("Unchanged calibration residual/phase changed")
                        if {r["sequence"] for r in cal_rows} != calibration.keys():
                            raise ValueError("Normal calibration inventory differs")
                        for r in cal_rows:
                            expected_valid = 45 <= int(r["frame"]) <= calibration[r["sequence"]]["frames"] - 8
                            if bool(int(r["valid"])) != expected_valid:
                                raise ValueError("History calibration mask differs")
                        selected = [r for r in cal_rows if r["valid"] == "1"]
                        pairs = np.array([[float(r["feature_raw"]), float(r["time_raw"])] for r in selected])
                        median = np.median(pairs, axis=0)
                        scale = np.maximum(1.4826 * np.median(np.abs(pairs - median), axis=0), 1e-6)
                        score = ((pairs - median) / scale).mean(axis=1)
                        threshold = float(np.quantile(score, .99))
                        saved = normal["calibration"]["P3"]
                        np.testing.assert_allclose(median, saved["median"], rtol=0, atol=1e-9)
                        np.testing.assert_allclose(scale, saved["mad_scale"], rtol=0, atol=1e-9)
                        np.testing.assert_allclose(score, [float(r["P3"]) for r in selected], rtol=0, atol=1e-9)
                        np.testing.assert_allclose(threshold, saved["threshold"], rtol=0, atol=1e-9)
                        checks.append({"backbone": model, "mode": mode, "device": device, "seed": seed,
                            "history": history, "normal_frames": len(selected), "threshold": threshold,
                            "normal_fit_sha256": digest(folder / "normal_fit.json"),
                            "normal_calibration_sha256": digest(folder / "normal_calibration.csv")})
                        run = load_run(folder, "P3", meta)
                        for key, (frames, _, _) in run.items():
                            source_rows = [r for r in manifest if r["device"] == device and
                                           r["partition"] == "testing" and r["sequence"] == key]
                            if len(source_rows) != 1 or np.any((frames < 45) | (frames > source_rows[0]["frames"] - 8)):
                                raise ValueError("History evaluation mask differs")
                            with (original / "P3" / f"{key}.csv").open() as stream:
                                expected_rows = [r for r in csv.DictReader(stream) if r["valid"] == "1" and int(r["frame"]) >= 45]
                            np.testing.assert_array_equal(frames, [int(r["frame"]) for r in expected_rows])
                            np.testing.assert_array_equal(run[key][1], [int(float(r["label"])) for r in expected_rows])
                        if device in references:
                            check_pair(references[device], run)
                        else:
                            references[device] = run
                        runs.append(run)
                    point = np.mean([statistic(run, list(run)) for run in runs], axis=0)
                    draws, rejected = bootstrap(runs)
                    low, high = np.quantile(draws, [.025, .975], axis=0)
                    key = (model, mode, device, f"H{history}")
                    points[key], stored[key] = point, (runs, draws)
                    stratified[key] = bootstrap_draws(runs, seed=np.random.SeedSequence([2026, int(device[1:])]))
                    labels = np.concatenate([v[1] for v in runs[0].values()])
                    rows.append({"backbone": model, "mode": mode, "device": device, "history": history,
                        "score_variant": "P3", "seeds": 3, "test_videos": len(runs[0]),
                        "frames": len(labels), "anomaly_frames": int(labels.sum()),
                        "auroc_mean": float(point[0]), "auroc_ci_low": float(low[0]), "auroc_ci_high": float(high[0]),
                        "ap_mean": float(point[1]), "ap_ci_low": float(low[1]), "ap_ci_high": float(high[1]),
                        "bootstrap_draws": 1000, "bootstrap_rejected": rejected})
    for model in MODELS:
        for mode in MODES:
            for device in DEVICES:
                baseline = (model, mode, device, "H5")
                for history in [15, 31]:
                    key = (model, mode, device, f"H{history}")
                    deltas.append(delta_row(model, mode, f"H{history}_minus_H5", points[key] - points[baseline],
                                             stored[key][1] - stored[baseline][1], device))
    macro = macro_four_devices(stored)
    if macro["incomplete_conditions"]:
        raise ValueError("History Macro4 is incomplete")
    for row in macro["results"]:
        row.update(history=int(row["variant"][1:]), score_variant="P3")
    macro_draws = {(m, mode, h): np.mean([stratified[m, mode, d, f"H{h}"] for d in DEVICES], axis=0)
                   for m in MODELS for mode in MODES for h in HISTORIES}
    valid = np.logical_and.reduce([np.isfinite(draw).all(axis=1) for draw in macro_draws.values()])
    if valid.sum() < 950:
        raise ValueError("Too many degenerate paired Macro4 history draws")
    for model in MODELS:
        for mode in MODES:
            for history in [15, 31]:
                point = np.mean([points[model, mode, d, f"H{history}"] - points[model, mode, d, "H5"] for d in DEVICES], axis=0)
                difference = (macro_draws[model, mode, history] - macro_draws[model, mode, 5])[valid]
                macro["paired_deltas"].append(delta_row(model, mode, f"H{history}_minus_H5", point, difference))
    write_summary(args.root / "device_summary", {"scope": "Same t=45..N-8 targets and normal calibration mask, 3-seed mean; whole-video CI",
                   "results": rows, "paired_deltas": deltas})
    write_summary(args.root / "macro_summary", macro)
    validation = {"status": "passed", "thresholds_checked": len(checks), "source_conditions_replayed": len(proofs),
        "primary_H5_alarm_mismatches": 0, "primary_H5_score_max_absolute_error": max(p["calibration_and_test_score_max_absolute_error"] for p in proofs.values()),
        "primary_H5_normal_time_max_absolute_error": max(p["normal_time_max_absolute_error"] for p in proofs.values()),
        "device_groups": len(rows), "macro_groups": len(macro["results"]), "paired_history_macro_differences": 8,
        "macro_bootstrap_rejected": int((~valid).sum()), "checks": checks,
        "device_summary_sha256": digest(args.root / "device_summary.json"),
        "macro_summary_sha256": digest(args.root / "macro_summary.json"),
        "completion_sha256": digest(args.root / "completion.json"), "verifier_sha256": digest(__file__),
        "bootstrap_helper_sha256": digest(Path(__file__).with_name("summarize_experiments.py")),
        "bootstrap_metric_sha256": digest("src/ipad_jepa/bootstrap_metrics.py"),
        "scope": "Normal calibration/score replay and paired cached-score statistics; not raw encoder re-extraction or runtime"}
    (args.root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
