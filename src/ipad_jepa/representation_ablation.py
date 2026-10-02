"""Train one shared dense-fit phase head and compare patch versus local spatial-mean memory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ipad_jepa.alignment import POLICY, align
from ipad_jepa.backbones import PhaseHead
from ipad_jepa.cache_data import sequences
from ipad_jepa.experiment import VARIANTS, alarms, combined, metrics, predict, score_sequence, temperature_sample
from ipad_jepa.features import file_hash
from ipad_jepa.memory import PrototypeMemory
from ipad_jepa.representation_data import dense_candidates, mean_sequence, validate_dense
from ipad_jepa.representation_features import write_json
from ipad_jepa.scoring import NormalCalibration
from ipad_jepa.temporal import circular_phase, common_mask
from ipad_jepa.torch_memory import TorchMemory


def train_phase(dense, original_cache, rows, out, seed, epochs=20, batch=256):
    if out.exists():
        raise ValueError("Fresh shared phase namespace required")
    info, fit, calibration = validate_dense(dense, rows, original_cache)
    values, labels = [], []
    for row in fit:
        folder = dense / "fit" / row["sequence"]
        targets = np.load(folder / "targets.npy", allow_pickle=False)
        values.append(np.load(folder / "global.npy", allow_pickle=False).astype(np.float32))
        labels.append(np.floor(200 * targets / row["frames"]).astype(np.int64))
    x, y = np.concatenate(values), np.concatenate(labels)
    vx = np.concatenate([s.global_features.astype(np.float32) for s in calibration])
    vphi = np.concatenate([s.targets / s.row["frames"] for s in calibration])
    vy = np.floor(200 * vphi).astype(np.int64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(vx)) or epochs != 20:
        raise ValueError("Finite normal-only inputs and the full twenty epochs required")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = PhaseHead().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    out.mkdir(parents=True)
    curves, best, selected_epoch = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train(); loss_sum = 0.0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(bx.cuda())
                loss = torch.nn.functional.cross_entropy(logits, by.cuda())
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(bx)
        model.eval(); validation_loss = 0.0; probabilities = []
        with torch.inference_mode():
            for i in range(0, len(vx), batch):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(torch.from_numpy(vx[i:i + batch]).cuda())
                validation_loss += float(torch.nn.functional.cross_entropy(
                    logits.float(), torch.from_numpy(vy[i:i + batch]).cuda(), reduction="sum"))
                probabilities.append(logits.float().softmax(1).cpu().numpy())
        validation_loss /= len(vx)
        phase = circular_phase(np.concatenate(probabilities))
        entry = {"epoch": epoch + 1, "train_ce": loss_sum / len(x), "normal_calibration_ce": validation_loss,
                 "normal_calibration_circular_mae": float(np.abs((phase - vphi + .5) % 1 - .5).mean())}
        if not np.all(np.isfinite(list(entry.values()))):
            raise ValueError("Nonfinite phase-head training statistics")
        curves.append(entry)
        if validation_loss < best:
            best, selected_epoch = validation_loss, epoch + 1
            torch.save(model.state_dict(), out / "phase_head.pt")
        print(json.dumps(entry), flush=True)
    with (out / "phase_training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(curves)
    result = {"status": "complete", "seed": seed, "epochs": epochs, "selected_epoch": selected_epoch,
              "backbone": info["identity"]["backbone"], "mode": info["identity"]["mode"], "device": info["device"],
              "fit_clips": len(x), "calibration_clips": len(vx), "phase_batch_size": batch,
              "fit_input_sha256": file_hash(dense / "dense_fit.json"),
              "normal_calibration_cache_fingerprints": [s.meta["fingerprint"] for s in calibration],
              "best_normal_calibration_ce": best, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "selection": "Minimum normal calibration CE; first epoch wins ties; no anomaly labels",
              "phase_input": "Same whole-context frozen pooled feature for patch and global_mean",
              "fit_stride": 1, "calibration_source": "Original frozen sixteen-frame caches; complete target stride 1",
              "shared_between_representations": True, "code_sha256": file_hash(Path(__file__)),
              "seconds": time.perf_counter() - started}
    write_json(out / "phase_training.json", result)
    return result


def evaluate(dense, original_cache, rows, phase_run, out, local, seed, data_root, representation):
    if out.exists() or local.exists():
        raise ValueError("Fresh representation evaluation and private bank namespaces required")
    started = time.perf_counter()
    info, fit_rows, calibration = validate_dense(dense, rows, original_cache, seed)
    phase_meta = json.loads((phase_run / "phase_training.json").read_text())
    identity = info["identity"]
    expected = {"seed": seed, "backbone": identity["backbone"], "mode": identity["mode"], "device": info["device"],
                "fit_input_sha256": file_hash(dense / "dense_fit.json"), "status": "complete", "epochs": 20,
                "normal_calibration_cache_fingerprints": [s.meta["fingerprint"] for s in calibration],
                "shared_between_representations": True, "fit_stride": 1}
    if any(phase_meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Shared phase checkpoint is not the correct completed dense normal-fit run")
    with (phase_run / "phase_training.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    if len(curves) != 20 or [int(r["epoch"]) for r in curves] != list(range(1, 21)):
        raise ValueError("Full phase training curve required")
    chosen = min(curves, key=lambda r: float(r["normal_calibration_ce"]))
    if (int(chosen["epoch"]) != phase_meta["selected_epoch"]
            or float(chosen["normal_calibration_ce"]) != phase_meta["best_normal_calibration_ce"]):
        raise ValueError("Phase selection differs from minimum normal calibration CE")
    head = PhaseHead().cuda().eval()
    head.load_state_dict(torch.load(phase_run / "phase_head.pt", map_location="cpu", weights_only=True), strict=True)
    x, phi, groups, sampling = dense_candidates(dense, fit_rows, identity["mode"], seed, representation)
    actual_pca_samples = min(50000, len(x))
    memory = PrototypeMemory(seed=seed).fit(x, phi, groups)
    del x, phi, groups
    scorer = TorchMemory(memory).cuda().eval()
    views = dense / "mean_views"
    if representation == "global_mean":
        calibration = [mean_sequence(s, views) for s in calibration]
    elif representation != "patch":
        raise ValueError("Unknown representation")
    cal_phase = [predict(s, head) for s in calibration]
    memory.temperature, n_temperature = temperature_sample(calibration, [p[0] for p in cal_phase], scorer, seed)
    scorer.temperature = memory.temperature
    cycle = float(np.median([r["frames"] for r in fit_rows]))
    cal_pairs = {v: [] for v in VARIANTS}; normal_outputs = []
    for s, (phase, dense_phase) in zip(calibration, cal_phase):
        raw = score_sequence(s, phase, dense_phase, scorer, cycle)
        valid = common_mask(s.row["frames"])[s.targets]
        normal_outputs.append((s, phase, raw, valid))
        for v in VARIANTS:
            pairs = raw[v][valid]
            if not np.all(np.isfinite(pairs)):
                raise ValueError("Invalid normal calibration scores")
            cal_pairs[v].append(pairs)
    calibrators = {}
    for v in VARIANTS:
        pairs = np.concatenate(cal_pairs[v])
        calibrators[v] = NormalCalibration().fit(pairs)
        calibrators[v].threshold = float(np.quantile(combined(calibrators[v], pairs, v), .99))
    local.mkdir(parents=True); out.mkdir(parents=True)
    np.savez(local / "memory.npz", mean=memory.pca.mean_, components=memory.pca.components_,
             prototypes=memory.prototypes, temperature=memory.temperature, cycle_length=cycle)
    metadata = {"status": "normal_fit_and_calibration_complete", "representation": representation,
                "seed": seed, "device": info["device"], "backbone": identity["backbone"], "mode": identity["mode"],
                "dense_fit_identity": identity, "dense_fit_source_sha256": file_hash(dense / "dense_fit.json"),
                "original_calibration_cache_identity": calibration[0].identity,
                "normal_calibration_cache_fingerprints": [s.meta["fingerprint"] for s in calibration],
                "phase_checkpoint_sha256": file_hash(phase_run / "phase_head.pt"),
                "phase_selected_epoch": phase_meta["selected_epoch"], "cycle_length_fit_median": cycle,
                "bins": 16, "prototypes_per_bin": 128, "total_prototypes": 2048, "pca_dimensions": 256,
                "pca_sample_limit": 50000, "pca_actual_samples": actual_pca_samples, "candidate_sampling": sampling,
                "pca_explained_variance_ratio": float(memory.pca.explained_variance_ratio_.sum()),
                "queries_per_target": 576 if representation == "patch" else 1,
                "fit_stride_both_representations": 1, "temperature": memory.temperature,
                "temperature_samples": n_temperature, "temperature_source": "Normal calibration, never fit or test",
                "calibration_valid_frames": sum(map(len, cal_pairs["P3"])),
                "calibration": {v: {"median": c.median.tolist(), "mad_scale": c.scale.tolist(), "threshold": c.threshold,
                                    "component_thresholds": c.component_thresholds.tolist()} for v, c in calibrators.items()},
                "code_sha256": file_hash(Path(__file__)), "scope": "Paired dense-fit representation OFAT; no runtime measurement"}
    write_json(out / "normal_fit.json", metadata)
    with (out / "normal_calibration.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sequence", "frame", "valid", "relative_phase", "predicted_phase", "feature_raw", "time_raw", *VARIANTS])
        for s, phase, raw, valid in normal_outputs:
            scores = [combined(calibrators[v], raw[v], v) for v in VARIANTS]
            writer.writerows(zip([s.row["sequence"]] * len(phase), s.targets, valid.astype(int),
                                 s.targets / s.row["frames"], phase, raw["P2"][:, 0], raw["P2"][:, 1], *scores))
    print(f"representation normal memory fixed: {identity['backbone']}/{identity['mode']}/{info['device']}/seed{seed}/{representation}", flush=True)
    # Test arrays/GT are opened only after both the head and this bank's calibration are fixed.
    test = sequences(original_cache, rows, "test")
    if test[0].identity != calibration[0].identity:
        raise ValueError("Original test and calibration encoder identities differ")
    if representation == "global_mean":
        test = [mean_sequence(s, views) for s in test]
    results = {v: {"labels": [], "scores": []} for v in VARIANTS}
    sensitivity = {v: {o: {"labels": [], "scores": []} for o in [-1, 0, 1]} for v in VARIANTS}
    alignment_notes, cache_proof = [], []
    for s in calibration + test:
        entry = {"partition": s.row["partition"], "sequence": s.row["sequence"],
                 "fingerprint": s.meta["fingerprint"], "spec": s.meta["spec"],
                 "metadata_sha256": file_hash(s.folder / "meta.json"), "targets_sha256": file_hash(s.folder / "targets.npy")}
        if representation == "global_mean":
            entry["mean_view"] = s.mean_source
        cache_proof.append(entry)
    for s in test:
        row = s.row; label_path = data_root / row["label_file"]
        if file_hash(label_path) != row["label_sha256"]:
            raise ValueError("Actual test GT changed")
        label, known, candidates = align(np.load(label_path, allow_pickle=False), row["frames"])
        phase, dense_phase = predict(s, head)
        raw = score_sequence(s, phase, dense_phase, scorer, cycle)
        inference_valid = common_mask(row["frames"])[s.targets]
        valid = inference_valid & known[s.targets]
        if row["alignment_status"] != "matched":
            alignment_notes.append({"sequence": row["sequence"], "unknown_common_frames": int((inference_valid & ~known[s.targets]).sum())})
        for v, pairs in raw.items():
            cal = calibrators[v]; scores = combined(cal, pairs, v)
            if not np.all(np.isfinite(scores[valid])):
                raise ValueError("Invalid test scores")
            alarm = alarms(scores, cal.threshold, inference_valid)
            types = np.zeros(len(scores), dtype=int)
            finite = np.all(np.isfinite(pairs), axis=1); types[finite] = cal.types(pairs[finite])
            folder = out / v; folder.mkdir(exist_ok=True)
            with (folder / f"{row['sequence']}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["frame", "label", "valid", "inference_valid", "phase", "feature_raw", "time_raw", "score", "alarm", "evidence_type"])
                writer.writerows(zip(s.targets, label[s.targets], valid.astype(int), inference_valid.astype(int), phase,
                                     pairs[:, 0], pairs[:, 1], scores, alarm.astype(int), types))
            results[v]["labels"].append(label[s.targets][valid]); results[v]["scores"].append(scores[valid])
            for offset in [-1, 0, 1]:
                gt = candidates.get(offset, candidates[0])[s.targets]
                keep = inference_valid & (gt >= 0)
                sensitivity[v][offset]["labels"].append(gt[keep]); sensitivity[v][offset]["scores"].append(scores[keep])
        print(f"representation test: {representation}/{identity['backbone']}/{identity['mode']}/{row['device']}/{row['sequence']}", flush=True)
    summary = {"status": "complete_device_evaluation", "representation": representation, "seed": seed,
               "backbone": identity["backbone"], "mode": identity["mode"], "device": info["device"], "test_videos": len(test),
               "valid_mask": "t=19..N-8 inclusive", "alignment_policy": POLICY, "unresolved_alignment_sequences": alignment_notes,
               "variants": {v: metrics(np.concatenate(r["labels"]), np.concatenate(r["scores"])) for v, r in results.items()},
               "alignment_sensitivity": {v: {str(o): metrics(np.concatenate(r["labels"]), np.concatenate(r["scores"]))
                                                 for o, r in offsets.items()} for v, offsets in sensitivity.items()} if alignment_notes else {},
               "seconds": time.perf_counter() - started,
               "note": "Device frame metrics; common-mask batch alarms only; no streaming EOF or real-time claim"}
    write_json(out / "cache_sources.json", {"original_calibration_test": cache_proof,
                                           "dense_fit_metadata_sha256": file_hash(dense / "dense_fit.json")})
    write_json(out / "metrics.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["phase", "evaluate"])
    parser.add_argument("--dense-fit", type=Path, required=True)
    parser.add_argument("--original-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    parser.add_argument("--device", choices=["R01", "R02", "R03", "R04"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-run", type=Path)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--representation", choices=["patch", "global_mean"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [r for r in json.loads(args.manifest.read_text())["sequences"] if r["device"] == args.device]
    if args.step == "phase":
        result = train_phase(args.dense_fit, args.original_cache, rows, args.out, args.seed)
    else:
        if any(value is None for value in [args.phase_run, args.local, args.data_root, args.representation]):
            raise ValueError("Evaluation requires phase run, private bank, raw GT root and representation")
        result = evaluate(args.dense_fit, args.original_cache, rows, args.phase_run, args.out, args.local,
                          args.seed, args.data_root, args.representation)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
