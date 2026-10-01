"""Publish measured GPU LoRA probes and tensor checks; never publish tensors."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/lora_pilots"))
    parser.add_argument("--out", type=Path, default=Path("results/stage04/pilots"))
    args = parser.parse_args()
    count = 0
    for path in sorted(args.runs.glob("*/*/*/seed*/lora_training.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("status") != "pilot_only":
            continue
        checkpoint = torch.load(path.parent / "pilot_adapter.pt", map_location="cpu", weights_only=True)
        if any(metadata.get(k) != value for k, value in checkpoint["protocol"].items()):
            raise ValueError("Pilot metadata/protocol differ")
        state = checkpoint["adapter"]
        b_matrices = {k: float(v.float().norm()) for k, v in state.items()
                      if k.endswith(("q_b.weight", "v_b.weight"))}
        if len(b_matrices) != 8 or any(v <= 0 for v in b_matrices.values()):
            raise ValueError("Zero-initialized q/v B matrices were not all updated")
        if any(not value.isfinite().all() for value in state.values()):
            raise ValueError("Nonfinite trained adapter")
        result = {**metadata, "pilot_checkpoint_sha256": None,
            "zero_initialized_qv_b_norms_after_update": b_matrices,
            "all_eight_qv_b_matrices_updated": True,
            "base_parameter_updates": "Excluded from optimizer; wrapper unit tests check frozen base/k",
            "scope": "One optimizer step, eight normal clips; no anomaly metrics or sustained FPS"}
        from ipad_jepa.features import file_hash
        result["pilot_checkpoint_sha256"] = file_hash(path.parent / "pilot_adapter.pt")
        destination = args.out / path.parent.relative_to(args.runs)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "lora_training.json").write_text(json.dumps(result, indent=2) + "\n")
        count += 1
    print(json.dumps({"exported_pilots": count, "tensors_published": False}))


if __name__ == "__main__":
    main()
