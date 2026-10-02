"""Run all 96 paired dense-fit patch/spatial-mean seed conditions in isolated namespaces."""
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
REPRESENTATIONS = ("patch", "global_mean")


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
    parser.add_argument("--original-cache", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--dense-fit", type=Path, default=Path("artifacts/representation_dense"))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs_representation"))
    parser.add_argument("--out", type=Path, default=Path("results/stage05/ablations/representation"))
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/tmp/representation_pipeline.json"))
    args = parser.parse_args()
    for values in [args.models, args.modes, args.devices]:
        if len(values) != len(set(values)):
            raise ValueError("Duplicate matrix conditions")
    for path in [args.dense_fit, args.runs, args.out, args.ledger]:
        if path.exists():
            raise ValueError(f"Fresh matrix namespace required: {path}")
    sources = [Path(__file__), args.manifest, Path("configs/experiment_matrix.yaml")]
    sources += [Path("src/ipad_jepa") / f"{name}.py" for name in [
        "representation", "representation_features", "representation_data", "representation_ablation",
        "backbones", "features", "temporal", "audit", "cache_data", "experiment", "memory",
        "torch_memory", "scoring", "alignment"]]
    pinned = {str(path): digest(path) for path in sources}
    for path in [args.dense_fit, args.runs, args.out]:
        path.mkdir(parents=True)
    ledger = {"status": "running", "planned_seed_conditions": len(args.models) * len(args.modes) * len(args.devices) * 6,
              "representations": list(REPRESENTATIONS), "seeds": list(SEEDS), "fit_stride_both_representations": 1,
              "normal_only": True, "source_sha256": pinned, "completed": []}
    write_json(args.ledger, ledger)

    def check_sources():
        if any(digest(path) != sha for path, sha in pinned.items()):
            raise ValueError("Pinned representation source changed while the matrix was active")

    def run(arguments):
        check_sources()
        subprocess.run([sys.executable, *arguments], check=True)

    try:
        for model in args.models:
            weights = Path("artifacts/weights") / {"dinov3-l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                                                 "vjepa21-l": "vjepa2_1_vitl_dist_vitG_384.pt"}[model]
            upstream = Path("third_party") / ("dinov3" if model == "dinov3-l" else "vjepa2")
            for mode in args.modes:
                original = args.original_cache / model / mode
                for device in args.devices:
                    group = Path(model) / mode / device
                    dense = args.dense_fit / group
                    ledger["current"] = {"backbone": model, "mode": mode, "device": device,
                                         "step": "encode_all_dense_normal_fit_clips"}
                    write_json(args.ledger, ledger)
                    run(["-m", "ipad_jepa.representation_features", "--data-root", str(args.data_root),
                         "--manifest", str(args.manifest), "--model", model, "--mode", mode, "--device", device,
                         "--weights", str(weights), "--upstream", str(upstream), "--original-cache", str(original),
                         "--out", str(dense)])
                    for seed in SEEDS:
                        condition = group / f"seed{seed}"
                        phase = args.runs / condition / "shared_phase"
                        ledger["current"].update({"seed": seed, "step": "train_shared_normal_phase_head"})
                        ledger["current"].pop("representation", None)
                        write_json(args.ledger, ledger)
                        common = ["--dense-fit", str(dense), "--original-cache", str(original), "--manifest", str(args.manifest),
                                  "--device", device, "--seed", str(seed)]
                        run(["-m", "ipad_jepa.representation_ablation", "phase", *common, "--out", str(phase)])
                        for representation in REPRESENTATIONS:
                            public = args.out / representation / condition
                            local = args.runs / condition / representation
                            ledger["current"].update({"representation": representation, "step": "refit_normal_memory_and_evaluate"})
                            write_json(args.ledger, ledger)
                            run(["-m", "ipad_jepa.representation_ablation", "evaluate", *common,
                                 "--data-root", str(args.data_root), "--representation", representation,
                                 "--phase-run", str(phase), "--local", str(local), "--out", str(public)])
                            check_sources()
                            metrics = json.loads((public / "metrics.json").read_text())
                            normal = json.loads((public / "normal_fit.json").read_text())
                            expected = {"representation": representation, "backbone": model, "mode": mode,
                                        "device": device, "seed": seed, "status": "complete_device_evaluation"}
                            if any(metrics.get(k) != v for k, v in expected.items()) or normal["fit_stride_both_representations"] != 1:
                                raise ValueError("Mismatched or incomplete paired representation condition")
                            for name in ["phase_training.csv", "phase_training.json"]:
                                shutil.copyfile(phase / name, public / name)
                            shutil.copyfile(dense / "dense_fit.json", public / "dense_fit_source.json")
                            proof = {**expected, "status": "completed_representation_condition",
                                     "source_sha256": pinned, "phase_head_sha256": digest(phase / "phase_head.pt"),
                                     "memory_sha256": digest(local / "memory.npz"),
                                     "outputs_sha256": {str(p.relative_to(public)): digest(p) for p in sorted(public.rglob("*")) if p.is_file()},
                                     "paired_condition": str(args.out / ("global_mean" if representation == "patch" else "patch") / condition),
                                     "scope": "Normal-only dense-fit representation OFAT; frozen original calibration/test features; no real-time result"}
                            write_json(public / "representation_source_proof.json", proof)
                            ledger["completed"].append(str(Path(representation) / condition))
                            write_json(args.ledger, ledger)
                            print(f"representation condition complete: {representation}/{condition}", flush=True)
        ledger["status"] = "complete_matrix"; ledger.pop("current", None)
        write_json(args.ledger, ledger)
    except BaseException as error:
        ledger["status"] = "failed"; ledger["error"] = repr(error)
        write_json(args.ledger, ledger)
        raise


if __name__ == "__main__":
    main()
