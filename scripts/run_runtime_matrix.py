"""Measure eligible primary conditions sequentially on an isolated GPU.

All 96 frozen/LoRA conditions remain in the plan. Missing accuracy conditions
remain pending; numerical parity failures are retained as scientific results.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODELS = ("dinov3-l", "vjepa21-l")
MODES = ("online", "offline")
DEVICES = ("R01", "R02", "R03", "R04")
SEEDS = (0, 1, 2)
WEIGHTS = {"dinov3-l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
           "vjepa21-l": "vjepa2_1_vitl_dist_vitG_384.pt"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def planned_conditions():
    return [{"adaptation": adaptation, "backbone": model, "mode": mode,
             "device": device, "seed": seed}
            for device in DEVICES for mode in MODES for model in MODELS
            for seed in SEEDS for adaptation in ("frozen", "lora")]


def implementations(condition):
    entries = [("bf16", "full"), ("bf16", "buffer")]
    if (condition["backbone"], condition["mode"]) == ("dinov3-l", "online"):
        entries += [("fp32", "full"), ("fp32", "reuse")]
    return entries


def condition_path(condition):
    return Path(condition["backbone"]) / condition["mode"] / condition["device"] / f"seed{condition['seed']}"


def paths(condition):
    suffix = condition_path(condition)
    adapted = condition["adaptation"] == "lora"
    return (Path("artifacts/runs_lora" if adapted else "artifacts/runs") / suffix,
            Path("results/stage04" if adapted else "results/stage02") / suffix,
            Path("artifacts/lora") / suffix if adapted else None)


def readiness(condition):
    run, public, training = paths(condition)
    required = [public / "metrics.json", public / "normal_fit.json",
                run / "phase_training.json", run / "phase_head.pt", run / "memory.npz"]
    if training is not None:
        required += [training / "lora_training.json", training / "selected_adapter.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"ready": False, "reason": "accuracy_or_selected_components_missing", "missing": missing}
    metrics = json.loads(required[0].read_text())
    normal = json.loads(required[1].read_text())
    phase = json.loads(required[2].read_text())
    expected = {key: condition[key] for key in ("backbone", "mode", "device", "seed")}
    if (metrics.get("status") != "complete_device_evaluation"
            or normal.get("status") != "normal_fit_and_calibration_complete"
            or any(meta.get(key) != value for meta in (metrics, normal, phase)
                   for key, value in expected.items())
            or phase.get("status") != "complete" or phase.get("epochs") != 20):
        raise ValueError(f"Existing condition metadata is incomplete or mismatched: {condition}")
    if training is not None:
        info = json.loads((training / "lora_training.json").read_text())
        if (info.get("status") != "complete_training" or info.get("completed_epochs") != 20
                or info.get("epochs") != 20 or info.get("teacher_weight") != 1.
                or any(info.get(key) != value for key, value in expected.items())):
            raise ValueError("Primary runtime requires completed teacher=1 joint LoRA training")
    return {"ready": True, "inputs_sha256": {str(path): digest(path) for path in required}}


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": fields[19]}
    except (OSError, IndexError):
        return None


def is_live(identity):
    return identity is not None and process_identity(identity["pid"]) == identity


def check_pins(pins):
    if any(digest(path) != value for path, value in pins.items()):
        raise ValueError("Pinned runtime source or fixed condition input changed")


def result_receipt(folder, condition, precision, implementation, manifest):
    info = json.loads((folder / "runtime.json").read_text())
    expected = {**condition, "precision": precision, "implementation": implementation}
    if (info.get("status") != "complete_measured_replay" or not info.get("all_test_videos")
            or info.get("arrival_fps") != 30. or info.get("test_bank_updates") is not False
            or any(info.get(key) != value for key, value in expected.items())):
        raise ValueError("Completed runtime output differs from planned primary condition")
    rows = [row for row in manifest if row["device"] == condition["device"] and row["partition"] == "testing"]
    if [row["sequence"] for row in info["sequences"]] != [row["sequence"] for row in rows]:
        raise ValueError("Every actual test video must be measured")
    for measured, actual in zip(info["sequences"], rows):
        if any(measured.get(key) != actual[key] for key in ["frames_content_sha256", "label_sha256"]):
            raise ValueError("Runtime input sources differ from the actual manifest")
        if measured["input_frames"] != actual["frames"]:
            raise ValueError("Runtime input frame count differs from the actual manifest")
    files = [folder / "runtime.json"] + [folder / f"{row['sequence']}.csv" for row in rows]
    return {"status": "completed_child_exit_zero", "output": str(folder),
            "sources_sha256": {str(path): digest(path) for path in files},
            "sustained_input_fps": info["sustained_input_fps"],
            "target_latency_p95_ms": info["target_latency_p95_ms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--cpu-python", type=Path, default=Path("artifacts/cpu-runtime/bin/python"))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/stage05/runtime"))
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/tmp/runtime_matrix.json"))
    parser.add_argument("--hold-file", type=Path, default=Path("artifacts/tmp/runtime_matrix.hold"),
                        help="While present, hold between measurements for audits/publication; never interrupt a timed video")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.ledger.parent / "isolated_runtime.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another isolated-runtime controller owns the GPU queue")
    sources = [Path(__file__), Path("scripts/benchmark_runtime.py"),
               Path("scripts/verify_runtime_parity.py"), args.manifest, Path("configs/experiment_matrix.yaml")]
    sources += [Path("src/ipad_jepa") / f"{name}.py" for name in [
        "alignment", "audit", "backbones", "adaptation", "adapted_features", "features",
        "experiment", "runtime_state", "runtime_adaptation", "streaming", "temporal",
        "torch_memory", "memory", "scoring", "cache_data"]]
    pins = {str(path): digest(path) for path in sources}
    plan = planned_conditions()
    if args.resume:
        state = json.loads(args.ledger.read_text())
        if state["plan"] != plan or state["source_sha256"] != pins:
            raise ValueError("Resume requires the same full plan and runtime sources")
        if is_live(state.get("owner")) or is_live(state.get("child")):
            raise RuntimeError("Previous controller or benchmark is still live")
        for record in state["completed_runs"].values():
            check_pins(record["sources_sha256"])
        for record in state["parity"].values():
            check_pins(record["sources_sha256"])
    else:
        if args.ledger.exists() or (args.out.exists() and any(args.out.iterdir())):
            raise ValueError("Use a fresh runtime namespace or explicitly resume its verified owner")
        state = {"plan": plan, "planned_primary_conditions": len(plan),
                 "planned_runtime_runs": sum(len(implementations(c)) for c in plan),
                 "source_sha256": pins, "completed_runs": {}, "parity": {}, "interrupted_outputs": [],
                 "scope": "All primary frozen/teacher1-LoRA seeds0/1/2; all test videos, 30 FPS FIFO, no drops. BF16 full/buffer plus separate DINO online FP32 full/reuse. Missing accuracy conditions remain pending."}
    state.update(status="running", owner=process_identity(os.getpid()), child=None)
    write(args.ledger, state)
    try:
        manifest = json.loads(args.manifest.read_text())["sequences"]
        snapshot = [(condition, readiness(condition)) for condition in plan]
        state["pending_accuracy_conditions"] = [dict(condition, readiness=ready)
                                                 for condition, ready in snapshot if not ready["ready"]]
        state["eligible_snapshot_conditions"] = sum(ready["ready"] for _, ready in snapshot)
        write(args.ledger, state)
        for condition, ready in snapshot:
            if not ready["ready"]:
                continue
            condition_root = args.out / condition["adaptation"] / condition_path(condition)
            for precision, implementation in implementations(condition):
                check_pins(pins)
                check_pins(ready["inputs_sha256"])
                folder = condition_root / precision / implementation
                key = str(folder.relative_to(args.out))
                if key in state["completed_runs"]:
                    continue
                while args.hold_file.exists():
                    state.update(status="held_between_measurements", child=None)
                    write(args.ledger, state)
                    time.sleep(10)
                state["status"] = "running"
                if folder.exists() and any(folder.iterdir()):
                    if not args.resume:
                        raise ValueError("Refusing output not owned by this runtime controller")
                    archived = Path("artifacts/runtime_interrupted") / f"recovery-{time.time_ns()}" / key
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(archived)
                    state["interrupted_outputs"].append({"output": str(folder), "preserved_as": str(archived),
                                                        "reason": "previous owner and benchmark verified absent"})
                run, public, training = paths(condition)
                upstream = Path("third_party") / ("dinov3" if condition["backbone"] == "dinov3-l" else "vjepa2")
                command = [str(args.python.absolute()), "scripts/benchmark_runtime.py",
                           "--model", condition["backbone"], "--mode", condition["mode"],
                           "--device", condition["device"], "--seed", str(condition["seed"]),
                           "--adaptation", condition["adaptation"], "--precision", precision,
                           "--implementation", implementation, "--arrival-fps", "30",
                           "--data-root", str(args.data_root), "--weights", str(Path("artifacts/weights") / WEIGHTS[condition["backbone"]]),
                           "--upstream", str(upstream), "--run", str(run), "--results", str(public),
                           "--out", str(folder), "--manifest", str(args.manifest)]
                if training is not None:
                    command += ["--lora-run", str(training)]
                state.update(current={**condition, "precision": precision, "implementation": implementation},
                             command=command, child=None)
                write(args.ledger, state)
                print(json.dumps({"status": "starting_measured_condition", "condition": state["current"]}), flush=True)
                child = subprocess.Popen(command)
                state["child"] = process_identity(child.pid)
                write(args.ledger, state)
                code = child.wait()
                state.update(child=None, actual_child_exit_code=code)
                write(args.ledger, state)
                if code != 0:
                    raise RuntimeError(f"Benchmark child exited {code}: {key}")
                check_pins(pins)
                check_pins(ready["inputs_sha256"])
                state["completed_runs"][key] = result_receipt(folder, condition, precision, implementation, manifest)
                write(args.ledger, state)
            for precision, candidate in [("bf16", "buffer"), ("fp32", "reuse")]:
                if (precision, candidate) not in implementations(condition):
                    continue
                out = condition_root / precision / "parity.json"
                key = str(out.relative_to(args.out))
                if key in state["parity"]:
                    continue
                check_pins(pins)
                command = [str(args.cpu_python.absolute()), "scripts/verify_runtime_parity.py",
                           "--reference", str(condition_root / precision / "full"),
                           "--candidate", str(condition_root / precision / candidate), "--out", str(out)]
                code = subprocess.run(command).returncode
                result = json.loads(out.read_text())
                if code not in (0, 1) or result["status"] != ("passed" if code == 0 else "failed"):
                    raise RuntimeError("Parity audit failed without a valid numerical result")
                state["parity"][key] = {"status": result["status"], "actual_child_exit_code": code,
                                         "sources_sha256": {str(out): digest(out)}}
                write(args.ledger, state)
        state["status"] = ("complete_primary_runtime_matrix" if not state["pending_accuracy_conditions"]
                           else "eligible_snapshot_measured_accuracy_conditions_pending")
        state.pop("current", None)
        write(args.ledger, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        write(args.ledger, state)
        raise


if __name__ == "__main__":
    main()
