"""Execute all seeds for a normal-only LoRA condition, rebuilding each adapted memory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(arguments):
    subprocess.run([sys.executable, *arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dinov3-l", "vjepa21-l"], required=True)
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--device", choices=["R01", "R02", "R03", "R04"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in [0, 1, 2] for seed in args.seeds):
        raise ValueError("Expected unique seeds from the accepted primary experiment")
    weights = Path("artifacts/weights") / {
        "dinov3-l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "vjepa21-l": "vjepa2_1_vitl_dist_vitG_384.pt"}[args.model]
    upstream = Path("third_party") / ("dinov3" if args.model == "dinov3-l" else "vjepa2")
    for seed in args.seeds:
        condition = Path(args.model) / args.mode / args.device / f"seed{seed}"
        local = Path("artifacts/lora") / condition
        cache = Path("artifacts/features_lora") / condition
        phase = Path("artifacts/runs_lora") / condition
        public = Path("results/stage04") / condition
        metadata = local / "lora_training.json"
        info = json.loads(metadata.read_text()) if metadata.exists() else None
        if info is not None and (info["seed"], info["backbone"], info["mode"], info["device"]) != (
            seed, args.model, args.mode, args.device):
            raise ValueError("Existing training condition differs")
        if info is None or info["status"] == "running":
            command = ["-m", "ipad_jepa.train_lora", "--model", args.model, "--mode", args.mode,
                "--device", args.device, "--seed", str(seed), "--data-root", str(args.data_root),
                "--teacher-cache", f"artifacts/features/{args.model}/{args.mode}",
                "--weights", str(weights), "--upstream", str(upstream), "--out", str(local)]
            if info is not None:
                # A running marker alone is not evidence that a process stopped. The caller
                # must verify that no live job owns this condition before invoking recovery.
                if not (local / "resume.pt").is_file():
                    raise ValueError("No complete epoch checkpoint available for recovery")
                command.append("--resume")
            run(command)
        elif info["status"] != "complete_training":
            raise ValueError("Pilot or invalid training cannot become a primary condition")
        run(["-m", "ipad_jepa.adapted_features", "--run", str(local), "--weights", str(weights),
            "--upstream", str(upstream), "--data-root", str(args.data_root), "--cache", str(cache),
            "--phase-out", str(phase)])
        public.mkdir(parents=True, exist_ok=True)
        for filename in ["lora_training.csv", "lora_training.json"]:
            shutil.copyfile(local / filename, public / filename)
        # The evaluator independently validates adapted cache/head fingerprints and rebuilds
        # PCA, prototypes and normal calibration; it never reuses the frozen tensor bank.
        run(["-m", "ipad_jepa.experiment", "--cache", str(cache), "--phase-run", str(phase),
             "--data-root", str(args.data_root), "--device", args.device, "--seed", str(seed),
             "--out", str(public), "--local", str(phase)])


if __name__ == "__main__":
    main()
