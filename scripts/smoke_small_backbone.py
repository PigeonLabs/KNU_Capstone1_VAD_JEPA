"""Strict-load one small encoder and test a full real past-only sixteen-frame normal clip."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from ipad_jepa.audit import inspect_frames
from ipad_jepa.features import ClipDataset, file_hash
from ipad_jepa.small_backbones import SmallBackbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dinov3-b", "vjepa21-b"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Fresh GPU smoke output required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for the actual model validation")
    row = next(r for r in json.loads(args.manifest.read_text())["sequences"]
               if r["device"] == "R01" and r["sequence"] == "01" and r.get("split") == "fit")
    current = inspect_frames(args.data_root / row["relative_directory"])
    if any(current[k] != row[k] for k in ["frames", "names_sha256", "frames_content_sha256"]):
        raise ValueError("Actual normal input differs from audited manifest")
    dataset = ClipDataset(args.data_root / row["relative_directory"], np.array([15]), "online", frames=16, padding=False)
    video, target = dataset[0]
    model = SmallBackbone(args.model, args.upstream, args.weights).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        local, pooled = model(video[None].cuda())
    if local.shape != (1, 576, 768) or pooled.shape != (1, 768) or not local.isfinite().all() or not pooled.isfinite().all():
        raise ValueError("Actual B encoder feature shape or values differ")
    source = Path(__file__).resolve().parents[1] / "src/ipad_jepa"
    result = {"model": args.model, "mode": "online", "status": "strict_load_and_real_clip_passed",
              "weight_sha256": file_hash(args.weights), "adapter_sha256": file_hash(source / "small_backbones.py"),
              "reader_sha256": file_hash(source / "features.py"), "smoke_code_sha256": file_hash(Path(__file__)),
              "upstream_commit": subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"]).decode().strip(),
              "normal_source": {"device": "R01", "sequence": "01", "split": "fit", "frames_content_sha256": row["frames_content_sha256"]},
              "clip_frames": 16, "target_frame": target, "input_frame_ids": list(range(16)), "future_frames_used": False,
              "local_shape": list(local.shape), "global_shape": list(pooled.shape),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": "BF16 inference, FP32 model parameters",
              "scope": "Actual frozen model/clip causality, strict weights, shapes and finite values only; no AUROC/AP/throughput/latency claim"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
