"""Normal-only joint q/v LoRA + phase training against immutable frozen dense caches."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ipad_jepa.adaptation import (install_adapters, adapter_training_mode,
    adapter_state, load_adapter_state, dense_teacher_loss, accumulation_divisor)
from ipad_jepa.audit import inspect_frames
from ipad_jepa.backbones import Backbone, PhaseHead
from ipad_jepa.cache_data import sequences
from ipad_jepa.features import ClipDataset, file_hash
from ipad_jepa.temporal import circular_phase


class NormalTeacherClips(Dataset):
    def __init__(self, cached, root, split):
        if split not in {"fit", "calibration"} or any(
            s.row["partition"] != "training" or s.row.get("split") != split for s in cached
        ):
            raise ValueError("LoRA may use only normal fit/calibration videos")
        self.cached, self.clips, self.index = cached, [], []
        for video, sequence in enumerate(cached):
            row, spec = sequence.row, sequence.meta["spec"]
            folder = root / row["relative_directory"]
            actual = inspect_frames(folder)
            if (actual["frames_content_sha256"] != row["frames_content_sha256"] or
                actual["names_sha256"] != row["names_sha256"]):
                raise ValueError("Normal raw frames changed after teacher extraction")
            self.clips.append(ClipDataset(folder, sequence.targets, spec["mode"],
                size=spec["image_size"], frames=spec["clip_frames"],
                padding=spec["padding"], cache_frames=16))
            self.index.extend((video, i) for i in range(len(sequence.targets)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        video, position = self.index[index]
        sequence = self.cached[video]
        clip, target = self.clips[video][position]
        phase = target / sequence.row["frames"]
        # Copy a bounded token array from mmap; teacher never participates in autograd.
        teacher = torch.from_numpy(np.array(sequence.patches[position], dtype=np.float32))
        if not teacher.isfinite().all():
            raise ValueError("Nonfinite frozen teacher features")
        return clip, int(np.floor(200 * phase)), teacher, phase


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_curves(out, curves):
    temporary = out / "lora_training.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    temporary.replace(out / "lora_training.csv")


def validate_normal(model, head, loader):
    adapter_training_mode(model, False)
    head.eval()
    total, count, predicted, actual = 0., 0, [], []
    with torch.inference_mode():
        for clip, label, _, phase in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, global_features = model(clip.cuda(non_blocking=True))
                logits = head(global_features)
            total += torch.nn.functional.cross_entropy(
                logits.float(), label.cuda(), reduction="sum").item()
            count += len(label)
            predicted.append(logits.float().softmax(1).cpu().numpy())
            actual.append(phase.numpy())
    phase = circular_phase(np.concatenate(predicted))
    error = np.abs((phase - np.concatenate(actual) + .5) % 1 - .5)
    return total / count, float(error.mean())


def train(args):
    if args.epochs < 1 or args.workers < 0 or args.accumulation < 1:
        raise ValueError("Invalid training counts")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("Pilot max-steps must be positive")
    if args.resume and args.max_steps is not None:
        raise ValueError("A pilot cannot resume a full experiment")
    if args.teacher_weight < 0 or not np.isfinite(args.teacher_weight):
        raise ValueError("Invalid teacher weight")
    out = args.out
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh output directory or explicit --resume")
    rows = [r for r in json.loads(args.manifest.read_text())["sequences"]
            if r["device"] == args.device]
    fit = sequences(args.teacher_cache, rows, "fit")
    calibration = sequences(args.teacher_cache, rows, "calibration")
    identity = fit[0].identity
    if any(s.identity != identity for s in fit + calibration):
        raise ValueError("Mixed teacher identities")
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"]).decode().strip()
    source = Path(__file__).parent
    expected = {"backbone": args.model, "mode": args.mode,
        "weights_sha256": file_hash(args.weights), "upstream_commit": upstream_commit,
        "adapter_sha256": file_hash(source / "backbones.py"),
        "reader_sha256": file_hash(source / "features.py"),
        "image_size": 384, "clip_frames": 16, "fit_stride": 4,
        "preprocessing": "RGB full-frame PIL bilinear resize; ImageNet mean/std",
        "feature_dtype": "float16 from BF16 inference"}
    if identity != expected:
        raise ValueError("Teacher cache differs from requested frozen encoder/protocol")
    protocol = {"seed": args.seed, "device": args.device, "backbone": args.model,
        "mode": args.mode, "epochs": args.epochs, "micro_batch": 1,
        "accumulation": args.accumulation, "teacher_weight": args.teacher_weight,
        "teacher_identity": identity,
        "teacher_fingerprints": sorted(s.meta["fingerprint"] for s in fit + calibration),
        "rank": 8, "alpha": 16, "dropout": .05, "blocks": 4,
        "encoder_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4,
        "precision": "bf16", "torch": str(torch.__version__),
        "trainer_sha256": file_hash(Path(__file__)),
        "adaptation_sha256": file_hash(source / "adaptation.py"),
        "max_steps": args.max_steps,
        "dense_loss": "Mean patchwise squared L2 of channel-normalized vectors",
        "selection": "Minimum normal calibration phase CE; no anomaly labels"}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    fit_data = NormalTeacherClips(fit, args.data_root, "fit")
    calibration_data = NormalTeacherClips(calibration, args.data_root, "calibration")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(fit_data, batch_size=1, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=True)
    validation = DataLoader(calibration_data, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    model = Backbone(args.model, args.upstream, args.weights, args.mode).cuda()
    encoder_parameters = install_adapters(model)
    if encoder_parameters != 131072:
        raise ValueError("Unexpected ViT-L adapter parameter count")
    head = PhaseHead().cuda()
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    curves, selected_epoch, best, updates, prior_seconds = [], 0, float("inf"), 0, 0.
    if args.resume:
        checkpoint = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if checkpoint["protocol"] != protocol or checkpoint["status"] != "running":
            raise ValueError("Resume source/protocol/status differs")
        load_adapter_state(model, checkpoint["adapter"])
        head.load_state_dict(checkpoint["head"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        curves, best = checkpoint["curves"], checkpoint["best"]
        selected_epoch, updates = checkpoint["selected_epoch"], checkpoint["updates"]
        prior_seconds = checkpoint["seconds"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        generator.set_state(checkpoint["loader_rng"])
    out.mkdir(parents=True, exist_ok=True)
    metadata = {**protocol, "status": "running", "fit_clips": len(fit_data),
        "calibration_clips": len(calibration_data), "encoder_trainable_parameters": encoder_parameters,
        "head_trainable_parameters": sum(p.numel() for p in head.parameters()),
        "training": "Normal-only joint phase head and q/v LoRA; fixed cached teacher; no decoder"}
    atomic_json(out / "lora_training.json", metadata)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(len(curves), args.epochs):
        adapter_training_mode(model, True)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        ce_sum, dense_sum, count = 0., 0., 0
        for i, (clip, label, teacher, _) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dense, global_features = model(clip.cuda(non_blocking=True))
                logits = head(global_features)
                ce = torch.nn.functional.cross_entropy(logits, label.cuda())
            distillation = dense_teacher_loss(dense, teacher.cuda(non_blocking=True))
            loss = ce.float() + args.teacher_weight * distillation
            if not loss.isfinite():
                raise ValueError("Nonfinite LoRA training loss")
            (loss / accumulation_divisor(i, len(loader), args.accumulation)).backward()
            ce_sum += ce.item()
            dense_sum += distillation.item()
            count += 1
            if (i + 1) % args.accumulation == 0 or i + 1 == len(loader):
                gradients = [p.grad for group in optimizer.param_groups for p in group["params"]
                             if p.grad is not None]
                if not gradients or any(not g.isfinite().all() for g in gradients):
                    raise ValueError("Invalid LoRA gradients")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if args.max_steps is not None and updates >= args.max_steps:
                    pilot = {**metadata, "status": "pilot_only", "updates": updates,
                        "processed_clips": count, "train_ce": ce_sum / count,
                        "train_dense_l2": dense_sum / count,
                        "seconds": time.perf_counter() - start,
                        "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
                        "note": "GPU training probe only; no full epoch, selection, or anomaly metrics"}
                    atomic_save(out / "pilot_adapter.pt", {"adapter": adapter_state(model),
                        "head": head.state_dict(), "protocol": protocol})
                    atomic_json(out / "lora_training.json", pilot)
                    print(json.dumps(pilot), flush=True)
                    return pilot
        validation_ce, validation_mae = validate_normal(model, head, validation)
        entry = {"epoch": epoch + 1, "train_ce": ce_sum / count,
            "train_dense_l2": dense_sum / count,
            "normal_calibration_ce": validation_ce,
            "normal_calibration_circular_mae": validation_mae, "clips": count,
            "updates": updates}
        if not np.isfinite(list(entry.values())).all():
            raise ValueError("Nonfinite LoRA validation")
        curves.append(entry)
        if validation_ce < best:
            best, selected_epoch = validation_ce, epoch + 1
            atomic_save(out / "selected_adapter.pt", {"adapter": adapter_state(model),
                "head": head.state_dict(), "protocol": protocol, "selected_epoch": selected_epoch})
            atomic_save(out / "phase_head.pt", head.state_dict())
        seconds = prior_seconds + time.perf_counter() - start
        checkpoint = {"protocol": protocol, "status": "running", "adapter": adapter_state(model),
            "head": head.state_dict(), "optimizer": optimizer.state_dict(), "curves": curves,
            "best": best, "selected_epoch": selected_epoch, "updates": updates, "seconds": seconds,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "loader_rng": generator.get_state()}
        atomic_save(out / "resume.pt", checkpoint)
        write_curves(out, curves)
        metadata.update(completed_epochs=len(curves), selected_epoch=selected_epoch,
            best_normal_calibration_ce=best, updates=updates, seconds=seconds,
            peak_vram_mib=torch.cuda.max_memory_allocated() / 1024**2)
        atomic_json(out / "lora_training.json", metadata)
        print(json.dumps(entry), flush=True)
    metadata["status"] = "complete_training"
    atomic_json(out / "lora_training.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--model", choices=["dinov3-l", "vjepa21-l"], required=True)
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--device", choices=["R01", "R02", "R03", "R04"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--teacher-weight", type=float, default=1.)
    parser.add_argument("--max-steps", type=int, help="Pilot only; never a full training result")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    train(args)


if __name__ == "__main__":
    main()
