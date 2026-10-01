"""Run the full frozen eight-frame matrix, refitting each normal-only pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

MODELS = ("dinov3-l", "vjepa21-l")
MODES = ("offline", "online")
DEVICES = ("R01", "R02", "R03", "R04")
SEEDS = (0, 1, 2)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--devices", nargs="+", choices=DEVICES, default=list(DEVICES))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/features_clip8"))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs_clip8"))
    parser.add_argument("--out", type=Path, default=Path("results/stage05/ablations/clip_frames/T8"))
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/tmp/clip8_pipeline.json"))
    args = parser.parse_args()
    for values in [args.models, args.modes, args.devices]:
        if len(values) != len(set(values)):
            raise ValueError("Duplicate matrix conditions")
    # Refuse implicit recovery. A missing completion marker does not prove a job stopped.
    for path in [args.cache, args.runs, args.out, args.ledger]:
        if path.exists():
            raise ValueError(f"Fresh output namespace required: {path}")
    sources = [Path(__file__), args.manifest, Path("configs/experiment_matrix.yaml")]
    sources += [Path("src/ipad_jepa") / f"{name}.py" for name in [
        "clip8_features", "backbones", "features", "temporal", "audit", "cache_data",
        "train_phase", "experiment", "memory", "torch_memory", "scoring", "alignment"]]
    pinned = {str(path): digest(path) for path in sources}
    manifest = json.loads(args.manifest.read_text())["sequences"]
    ledger = {"status": "running", "clip_frames": 8, "seeds": list(SEEDS),
              "planned_seed_conditions": len(args.models) * len(args.modes) * len(args.devices) * 3,
              "normal_only": True, "source_sha256": pinned, "completed": []}
    write_json(args.ledger, ledger)

    def check_sources():
        if any(digest(path) != value for path, value in pinned.items()):
            raise ValueError("Pinned matrix source changed while the experiment was active")

    def run(arguments):
        check_sources()
        subprocess.run([sys.executable, *arguments], check=True)

    try:
        for model in args.models:
            weight = Path("artifacts/weights") / {
                "dinov3-l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                "vjepa21-l": "vjepa2_1_vitl_dist_vitG_384.pt"}[model]
            upstream = Path("third_party") / ("dinov3" if model == "dinov3-l" else "vjepa2")
            for mode in args.modes:
                cache = args.cache / model / mode
                for device in args.devices:
                    ledger["current"] = {"backbone": model, "mode": mode, "device": device,
                                         "step": "extract_real_eight_frame_features"}
                    write_json(args.ledger, ledger)
                    run(["-m", "ipad_jepa.clip8_features", "--data-root", str(args.data_root),
                         "--manifest", str(args.manifest), "--model", model, "--mode", mode,
                         "--weights", str(weight), "--upstream", str(upstream),
                         "--devices", device, "--splits", "fit", "calibration", "test",
                         "--cache", str(args.cache)])
                    rows = [r for r in manifest if r["device"] == device and
                            r.get("split", "test") in {"fit", "calibration", "test"}]
                    cache_proof = []
                    for row in rows:
                        folder = cache / device / row["partition"] / row["sequence"]
                        info = json.loads((folder / "meta.json").read_text())
                        if info["status"] != "complete" or info["spec"]["clip_frames"] != 8:
                            raise ValueError("Extraction did not produce complete eight-frame features")
                        cache_proof.append({"device": device, "partition": row["partition"],
                                            "sequence": row["sequence"], "split": row.get("split", "test"),
                                            "metadata_sha256": digest(folder / "meta.json"),
                                            "targets_sha256": digest(folder / "targets.npy"),
                                            "fingerprint": info["fingerprint"], "spec": info["spec"]})
                    for seed in SEEDS:
                        condition = Path(model) / mode / device / f"seed{seed}"
                        local, public = args.runs / condition, args.out / condition
                        ledger["current"] = {"backbone": model, "mode": mode, "device": device,
                                             "seed": seed, "step": "normal_phase_head_and_memory"}
                        write_json(args.ledger, ledger)
                        run(["-m", "ipad_jepa.train_phase", "--cache", str(cache),
                             "--manifest", str(args.manifest), "--device", device,
                             "--seed", str(seed), "--epochs", "20", "--out", str(local)])
                        run(["-m", "ipad_jepa.experiment", "--cache", str(cache),
                             "--manifest", str(args.manifest), "--phase-run", str(local),
                             "--data-root", str(args.data_root), "--device", device,
                             "--seed", str(seed), "--local", str(local), "--out", str(public)])
                        check_sources()
                        metrics = json.loads((public / "metrics.json").read_text())
                        fit = json.loads((public / "normal_fit.json").read_text())
                        phase = json.loads((local / "phase_training.json").read_text())
                        if (metrics["status"] != "complete_device_evaluation"
                                or (metrics["backbone"], metrics["mode"], metrics["device"], metrics["seed"])
                                != (model, mode, device, seed) or fit["cache_identity"]["clip_frames"] != 8
                                or phase["status"] != "complete" or phase["epochs"] != 20):
                            raise ValueError("Incomplete or mismatched eight-frame condition")
                        for name in ["phase_training.json", "phase_training.csv"]:
                            shutil.copyfile(local / name, public / name)
                        proof = {"status": "completed_clip8_condition", "clip_frames": 8,
                                 "backbone": model, "mode": mode, "device": device, "seed": seed,
                                 "source_sha256": pinned, "cache_metadata": cache_proof,
                                 "phase_head_sha256": digest(local / "phase_head.pt"),
                                 "memory_sha256": digest(local / "memory.npz"),
                                 "outputs_sha256": {str(path.relative_to(public)): digest(path)
                                                    for path in sorted(public.rglob("*")) if path.is_file()},
                                 "paired_clip16_reference": str(Path("results/stage02") / condition),
                                 "comparison_mask": "t=19..N-8 inclusive, shared annotation uncertainty",
                                 "scope": "Refitted normal-only clip-context pipeline; no runtime measurement"}
                        write_json(public / "clip8_source_proof.json", proof)
                        ledger["completed"].append(str(condition))
                        write_json(args.ledger, ledger)
                        print(f"clip8 condition complete: {condition}", flush=True)
        ledger["status"] = "complete_matrix"
        ledger.pop("current", None)
        write_json(args.ledger, ledger)
    except BaseException as error:
        ledger["status"] = "failed"
        ledger["error"] = repr(error)
        write_json(args.ledger, ledger)
        raise


if __name__ == "__main__":
    main()
