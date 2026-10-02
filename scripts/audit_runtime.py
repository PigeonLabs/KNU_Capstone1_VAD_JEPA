"""Recalculate FIFO timing, fixed-normal P3 scores, alarms and GT metrics from actual traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ipad_jepa.alignment import align
from ipad_jepa.runtime_state import event_metrics

STAGES = ("decode_resize_ms", "encoder_ms", "feature_cast_ms", "phase_head_ms",
          "projection_search_ms", "score_alarm_ms", "service_ms")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9, equal_nan=True)


def metrics(labels, scores):
    if len(np.unique(labels)) != 2:
        return None
    return {"frame_auroc": float(roc_auc_score(labels, scores)),
            "frame_ap": float(average_precision_score(labels, scores)),
            "frames": len(labels), "anomaly_frames": int(np.sum(labels))}


def check_metrics(actual, expected):
    if expected is None:
        if actual is not None:
            raise ValueError("Metrics require both actual GT classes")
    else:
        for key, value in expected.items():
            close(actual[key], value)


def fixed_calibration(public, normal, manifest):
    rows = read(public / "normal_calibration.csv")
    expected_rows = [r for r in manifest if r["device"] == normal["device"] and r.get("split") == "calibration"]
    first = 15 if normal["mode"] == "online" else 8
    expected = [(r["sequence"], target) for r in expected_rows
                for target in range(first, r["frames"] if normal["mode"] == "online" else r["frames"] - 7)]
    actual = [(r["sequence"], int(r["frame"])) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("Actual fixed-normal calibration inventory differs")
    lengths = {r["sequence"]: r["frames"] for r in expected_rows}
    for row in rows:
        if int(row["valid"]) != int(19 <= int(row["frame"]) <= lengths[row["sequence"]] - 8):
            raise ValueError("Actual fixed-normal calibration mask differs")
    selected = [r for r in rows if r["valid"] == "1"]
    values = np.array([[float(r["feature_raw"]), float(r["time_raw"])] for r in selected])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite normal calibration pairs required")
    median = np.median(values, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(values - median), axis=0), 1e-6)
    scores = ((values - median) / scale).mean(1)
    calibration = normal["calibration"]["P3"]
    close(calibration["median"], median)
    close(calibration["mad_scale"], scale)
    close(calibration["threshold"], np.quantile(scores, .99))
    close([float(r["P3"]) for r in selected], scores)
    return calibration


def audit_trace(rows, mode, calibration, cycle, labels, known, fps=30.):
    """This score/alarm replay does not call StreamingScore or use test GT to gate alarms."""
    if not rows or len(rows) != len(labels):
        raise ValueError("All input frames and actual annotation length required")
    if [int(row["arrival_frame"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Input frames dropped, reordered or duplicated")
    phases = []
    streak = 0
    selected = []
    shared = []
    previous_emitted = 0.
    for index, row in enumerate(rows):
        arrival = index / fps
        start, emitted = float(row["service_start_seconds"]), float(row["emitted_seconds"])
        if not np.isfinite([arrival, start, emitted]).all() or start + 1e-9 < max(arrival, previous_emitted) or emitted < start:
            raise ValueError("Noncausal FIFO service/emission timestamps")
        previous_emitted = emitted
        close(float(row["arrival_seconds"]), arrival)
        close(float(row["queue_wait_ms"]), max(0, start - arrival) * 1000)
        close(float(row["service_ms"]), (emitted - start) * 1000)
        costs = np.array([float(row[key]) for key in STAGES[:-1]])
        if not np.isfinite(costs).all() or np.any(costs < 0) or costs.sum() > float(row["service_ms"]) + 1e-6:
            raise ValueError("Invalid measured component costs")
        if index < 15:
            if row["target_frame"] != "" or any(row[k] != "0" for k in ["inference_valid", "shared_metric_valid", "alarm"]):
                raise ValueError("Full clip required before target inference")
            continue
        target = index if mode == "online" else index - 7
        if int(row["target_frame"]) != target or int(row["label"]) != int(labels[target]):
            raise ValueError("Actual target or GT label differs")
        close(float(row["target_latency_ms"]), (emitted - target / fps) * 1000)
        close(float(row["lookahead_wait_ms"]), 0 if mode == "online" else 7000 / fps)
        phase, feature = float(row["phase"]), float(row["feature_raw"])
        if not np.isfinite([phase, feature]).all() or not 0 <= phase < 1:
            raise ValueError("Invalid measured phase/feature score")
        phases.append(phase)
        time_score = np.nan
        if len(phases) >= 5:
            time_score = np.mean([abs((phase - phases[-1 - lag] + .5) % 1 - .5 - lag / cycle)
                                  for lag in range(1, 5)])
        score = np.mean((np.array([feature, time_score]) - calibration["median"]) / calibration["mad_scale"])
        close(float(row["time_raw"]), time_score)
        close(float(row["score"]), score)
        eligible = target >= 19 and np.isfinite(score)
        valid = 19 <= target <= len(rows) - 8 and bool(known[target])
        if int(row["inference_valid"]) != int(eligible) or int(row["shared_metric_valid"]) != int(valid):
            raise ValueError("Operational/common GT eligibility differs")
        streak = streak + 1 if eligible and score > calibration["threshold"] else 0
        if int(row["alarm"]) != int(streak >= 3):
            raise ValueError("Fixed-normal consecutive alarm differs")
        if eligible:
            selected.append(row)
        if valid:
            shared.append(row)
    return selected, shared


def audit(folder, manifest_path, data_root):
    report = json.loads((folder / "runtime.json").read_text())
    if report["status"] != "complete_measured_replay" or not report["all_test_videos"] or report["arrival_fps"] != 30.:
        raise ValueError("Completed all-video 30 FPS actual runtime required")
    if report["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Measured manifest changed")
    suffix = Path(report["backbone"]) / report["mode"] / report["device"] / f"seed{report['seed']}"
    adapted = report["adaptation"] == "lora"
    public = Path("results/stage04" if adapted else "results/stage02") / suffix
    local = Path("artifacts/runs_lora" if adapted else "artifacts/runs") / suffix
    for source, key in [(public / "normal_fit.json", "normal_fit_sha256"),
                        (local / "phase_head.pt", "phase_head_sha256"), (local / "memory.npz", "memory_sha256")]:
        if digest(source) != report[key]:
            raise ValueError("Fixed-normal runtime component changed")
    if (report["runtime_code_sha256"] != digest("scripts/benchmark_runtime.py")
            or report["score_state_sha256"] != digest("src/ipad_jepa/runtime_state.py")):
        raise ValueError("Measured runtime source changed")
    normal = json.loads((public / "normal_fit.json").read_text())
    if any(normal[k] != report[k] for k in ["backbone", "mode", "device", "seed"]):
        raise ValueError("Fixed-normal condition differs")
    if report["weights_sha256"] != normal["cache_identity"]["weights_sha256"]:
        raise ValueError("Measured pretrained source differs")
    manifest = json.loads(manifest_path.read_text())["sequences"]
    calibration = fixed_calibration(public, normal, manifest)
    expected = [r for r in manifest if r["device"] == report["device"] and r["partition"] == "testing"]
    if [r["sequence"] for r in report["sequences"]] != [r["sequence"] for r in expected]:
        raise ValueError("Measured all-video inventory differs")
    all_rows, all_eligible, all_shared, sources = [], [], [], {}
    for info, row in zip(report["sequences"], expected):
        label_path = data_root / row["label_file"]
        if digest(label_path) != row["label_sha256"] or info["label_sha256"] != row["label_sha256"]:
            raise ValueError("Actual GT annotation changed")
        if info["frames_content_sha256"] != row["frames_content_sha256"]:
            raise ValueError("Measured raw-input provenance differs")
        labels, known, _ = align(np.load(label_path, allow_pickle=False), row["frames"])
        path = folder / f"{row['sequence']}.csv"
        rows = read(path)
        eligible, shared = audit_trace(rows, report["mode"], calibration, normal["cycle_length_fit_median"], labels, known)
        if info["input_frames"] != len(rows) or info["eligible_targets"] != len(eligible):
            raise ValueError("Declared measured inventory differs")
        elapsed = info["elapsed_seconds"]
        if elapsed < float(rows[-1]["emitted_seconds"]):
            raise ValueError("Elapsed time excludes queue drain")
        close(info["input_fps"], (len(rows) - 1) / elapsed)
        close(info["queue_wait_final_ms"], float(rows[-1]["queue_wait_ms"]))
        close(info["queue_growth_ms"], float(rows[-1]["queue_wait_ms"]) - float(rows[0]["queue_wait_ms"]))
        check_metrics(info["frame_metrics_shared_mask"], metrics(np.array([int(r["label"]) for r in shared]),
                                                                np.array([float(r["score"]) for r in shared])))
        frames = np.array([int(r["target_frame"]) for r in eligible], dtype=int)
        alarms = np.array([bool(int(r["alarm"])) for r in eligible])
        emissions = np.array([float(r["emitted_seconds"]) for r in eligible])
        mask = (frames >= 19) & (frames <= row["frames"] - 8)
        for scope, keep in [("operational_events", np.ones(len(frames), dtype=bool)), ("shared_target_events", mask)]:
            fresh = event_metrics(labels, frames[keep], alarms[keep], emissions[keep], 30.)
            if fresh != info[scope]:
                raise ValueError("Measured event accounting differs from trace replay")
        sources[str(path)] = digest(path)
        all_rows += rows
        all_eligible += eligible
        all_shared += shared
    latencies = [float(r["target_latency_ms"]) for r in all_eligible]
    queue = [float(r["queue_wait_ms"]) for r in all_rows]
    for key, values, q in [("target_latency_p50_ms", latencies, .5), ("target_latency_p95_ms", latencies, .95),
                           ("queue_wait_p50_ms", queue, .5), ("queue_wait_p95_ms", queue, .95)]:
        close(report[key], np.quantile(values, q))
    close(report["queue_wait_max_ms"], max(queue))
    close(report["queue_growth_max_ms"], max(r["queue_growth_ms"] for r in report["sequences"]))
    close(report["sustained_input_fps"], sum(r["input_frames"] - 1 for r in report["sequences"]) /
          sum(r["elapsed_seconds"] for r in report["sequences"]))
    for stage in STAGES:
        close(report["stage_mean_ms"][stage], np.mean([float(r[stage]) for r in all_eligible]))
    check_metrics(report["frame_metrics_shared_mask"], metrics(np.array([int(r["label"]) for r in all_shared]),
                                                              np.array([float(r["score"]) for r in all_shared])))
    for scope in ["operational_events", "shared_target_events"]:
        totals = report["event_totals"][scope]
        for key in ["covered_events", "events_outside_coverage", "detected_events", "missed_events",
                    "observed_end_evaluable_events", "events_detected_before_observed_end", "late_detected_events",
                    "events_without_alarm_before_observed_end", "false_alarm_episode_starts",
                    "unknown_alarm_episode_starts", "false_alarm_frames", "evaluated_normal_frames"]:
            close(totals[key], sum(row[scope][key] for row in report["sequences"]))
        delays = [event["delay_seconds"] for row in report["sequences"] for event in row[scope]["events"]
                  if event["delay_seconds"] is not None]
        for key, quantile in [("onset_delay_p50_seconds", .5), ("onset_delay_p95_seconds", .95)]:
            if not delays:
                if totals[key] is not None:
                    raise ValueError("No measured certain-onset delay available")
            else:
                close(totals[key], np.quantile(delays, quantile))
    sources.update({str(folder / "runtime.json"): digest(folder / "runtime.json"),
                    str(public / "normal_fit.json"): digest(public / "normal_fit.json"),
                    str(public / "normal_calibration.csv"): digest(public / "normal_calibration.csv")})
    return {"status": "passed_actual_runtime_trace_replay", "runtime": str(folder),
            "condition": {k: report[k] for k in ["backbone", "mode", "device", "seed", "adaptation", "precision", "implementation"]},
            "test_videos": len(expected), "input_frames": len(all_rows), "eligible_targets": len(all_eligible),
            "shared_metric_targets": len(all_shared), "normal_thresholds_recomputed": 1,
            "sources_sha256": sources, "auditor_sha256": digest(__file__),
            "manifest_sha256": digest(manifest_path),
            "limits": "Actual CSV timestamp/calibration/GT/time-score/alarm/metric replay; event accounting uses the shared frozen event definition. Raw JPEG hashes were checked by the measuring child and bound to the manifest, not re-encoded here. No independent encoder/PCA/search/VRAM replay, repeat-run CI, maximum-FPS or real-camera claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("../IPAD_dataset/IPAD_dataset"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    proof = audit(args.runtime, args.manifest, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "sources_sha256"}), flush=True)


if __name__ == "__main__":
    main()
