"""Rebuild selected LoRA features and bind the selected joint head to their provenance."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from ipad_jepa.adaptation import install_adapters, load_adapter_state, adapter_training_mode
from ipad_jepa.backbones import Backbone, PhaseHead
from ipad_jepa.cache_data import sequences
from ipad_jepa.features import extract_sequence, file_hash
from ipad_jepa.train_lora import atomic_json, atomic_save


def selected_protocol(run):
    metadata = json.loads((run / "lora_training.json").read_text())
    if (metadata.get("status") != "complete_training" or metadata.get("epochs") != 20 or
        metadata.get("completed_epochs") != 20 or metadata.get("max_steps") is not None):
        raise ValueError("Selected LoRA inference requires completed 20-epoch training")
    checkpoint = torch.load(run / "selected_adapter.pt", map_location="cpu", weights_only=True)
    protocol = checkpoint["protocol"]
    if any(metadata.get(k) != v for k, v in protocol.items()):
        raise ValueError("Selected adapter metadata/protocol differs")
    if checkpoint["selected_epoch"] != metadata["selected_epoch"]:
        raise ValueError("Selected epoch differs")
    with (run / "lora_training.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    if [int(r["epoch"]) for r in curves] != list(range(1, 21)):
        raise ValueError("Incomplete LoRA epoch curve")
    selected = min(curves, key=lambda r: float(r["normal_calibration_ce"]))
    if (int(selected["epoch"]) != metadata["selected_epoch"] or
        float(selected["normal_calibration_ce"]) != metadata["best_normal_calibration_ce"]):
        raise ValueError("Selected checkpoint differs from normal calibration criterion")
    source = Path(__file__).parent
    if (protocol["trainer_sha256"] != file_hash(source / "train_lora.py") or
        protocol["adaptation_sha256"] != file_hash(source / "adaptation.py")):
        raise ValueError("LoRA source changed; retain the selected implementation revision")
    return metadata, checkpoint


def rebuild(args):
    metadata, checkpoint = selected_protocol(args.run)
    teacher = metadata["teacher_identity"]
    source = Path(__file__).parent
    actual = {"weights_sha256": file_hash(args.weights),
        "adapter_sha256": file_hash(source / "backbones.py"),
        "reader_sha256": file_hash(source / "features.py"),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"]).decode().strip()}
    if any(teacher[k] != v for k, v in actual.items()):
        raise ValueError("Encoder/reader differs from selected adaptation teacher")
    model = Backbone(metadata["backbone"], args.upstream, args.weights, metadata["mode"]).cuda()
    install_adapters(model)
    load_adapter_state(model, checkpoint["adapter"])
    adapter_training_mode(model, False)
    selected_hash = file_hash(args.run / "selected_adapter.pt")
    fingerprint = {**teacher, "schema_version": 2,
        "torch": str(torch.__version__), "adaptation": "qv_lora_last4",
        "adaptation_seed": metadata["seed"], "selected_adapter_sha256": selected_hash,
        "adaptation_source_sha256": metadata["adaptation_sha256"],
        "adapted_reader_sha256": file_hash(Path(__file__)),
        "selected_epoch": metadata["selected_epoch"],
        "adapter_sha256": hashlib.sha256(json.dumps({
            "base": teacher["adapter_sha256"], "source": metadata["adaptation_sha256"],
            "selected": selected_hash}, sort_keys=True).encode()).hexdigest()}
    rows = [r for r in json.loads(args.manifest.read_text())["sequences"]
            if r["device"] == metadata["device"]]
    # Always build fit and calibration before test; a fresh bank must use adapted features.
    for split in ["fit", "calibration", "diagnostic", "test"]:
        if split not in args.splits:
            continue
        for row in rows:
            if row.get("split", "test") == split:
                extract_sequence(model, row, args.data_root, args.cache, fingerprint,
                                 args.batch_size, args.workers, teacher["fit_stride"])
    fit = sequences(args.cache, rows, "fit")
    calibration = sequences(args.cache, rows, "calibration")
    if fit[0].identity != calibration[0].identity:
        raise ValueError("Adapted normal identities differ")
    phase_out = args.phase_out
    phase_out.mkdir(parents=True, exist_ok=True)
    info = {"status": "complete", "seed": metadata["seed"], "epochs": 20,
        "selected_epoch": metadata["selected_epoch"], "backbone": metadata["backbone"],
        "mode": metadata["mode"], "device": metadata["device"],
        "training": "Joint LoRA + phase head; no post-adaptation head retraining",
        "selection": metadata["selection"], "selected_adapter_sha256": selected_hash,
        "teacher_cache_fingerprints": metadata["teacher_fingerprints"],
        "cache_fingerprints": [s.meta["fingerprint"] for s in fit + calibration]}
    previous = phase_out / "phase_training.json"
    if previous.exists() and json.loads(previous.read_text()) != info:
        raise ValueError("Phase output contains another adaptation identity")
    head = PhaseHead()
    head.load_state_dict(checkpoint["head"], strict=True)
    atomic_save(phase_out / "phase_head.pt", head.state_dict())
    atomic_json(previous, info)
    print(json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Separate cache namespace per seed")
    parser.add_argument("--phase-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--splits", nargs="+", choices=["fit", "calibration", "diagnostic", "test"],
                        default=["fit", "calibration", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid extraction counts")
    rebuild(args)


if __name__ == "__main__":
    main()
