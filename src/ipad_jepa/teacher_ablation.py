"""Protocol checks for the new zero-weight teacher-loss LoRA treatment."""
import numpy as np


def check_zero_training(meta, curves, model, mode, device, seed):
    expected = {"status":"complete_training", "backbone":model, "mode":mode, "device":device,
                "seed":seed, "teacher_weight":0., "epochs":20, "completed_epochs":20,
                "micro_batch":1, "accumulation":8, "rank":8, "alpha":16, "dropout":.05,
                "blocks":4, "encoder_lr":1e-4, "head_lr":1e-3, "precision":"bf16", "max_steps":None}
    if any(meta.get(k) != value for k,value in expected.items()):
        raise ValueError("Complete zero-weight normal LoRA treatment required")
    if len(curves) != 20 or [int(r["epoch"]) for r in curves] != list(range(1,21)):
        raise ValueError("Full 20-epoch treatment curve required")
    values = np.array([[float(r[k]) for k in ["train_ce","train_dense_l2","normal_calibration_ce","normal_calibration_circular_mae"]] for r in curves])
    if not np.isfinite(values).all() or any(int(r["clips"]) != meta["fit_clips"] for r in curves):
        raise ValueError("Invalid complete normal training statistics")
    chosen = int(values[:,2].argmin())
    if meta.get("selected_epoch") != chosen+1 or meta.get("best_normal_calibration_ce") != float(values[chosen,2]):
        raise ValueError("Zero-weight head must minimize normal calibration CE")


def expected_cache_bytes(rows, mode):
    if mode not in {"offline","online"} or not rows:
        raise ValueError("Accepted nonempty mode/device rows required")
    count = 0
    for row in rows:
        split = row.get("split","test")
        if split == "diagnostic":continue
        if (split not in {"fit","calibration","test"} or row["frames"] < 16
                or row["partition"] != ("testing" if split == "test" else "training")):
            raise ValueError("Invalid normal/test cache inventory")
        start = 0 if mode == "online" and split == "fit" else (15 if mode == "online" else 8)
        stop = row["frames"] if mode == "online" else row["frames"]-7
        count += len(range(start,stop,4 if split == "fit" else 1))
    # Actual FP16 local576x1024 and context1024 plus int64 targets. Allow
    # metadata/NPY headers and temporary private model/head/bank files separately.
    return count*(2*(576*1024+1024)+8)+len(rows)*1024**2
