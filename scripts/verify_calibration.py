"""Recompute every published normal q99 threshold from its source score CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/stage02"))
    parser.add_argument("--manifest", type=Path, default=Path("results/stage00/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())["sequences"]
    checks = []
    for marker in sorted(args.root.glob("*/*/*/seed*/metrics.json")):
        metrics = json.loads(marker.read_text())
        if metrics["status"] != "complete_device_evaluation":
            continue
        expected = {r["sequence"]: r["frames"] for r in manifest if
            r["device"] == metrics["device"] and r["partition"] == "training" and
            r.get("split") == "calibration"}
        csv_path, fit_path = marker.parent / "normal_calibration.csv", marker.parent / "normal_fit.json"
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        if {r["sequence"] for r in rows} != expected.keys():
            raise ValueError("Calibration video inventory differs from normal-only split")
        for row in rows:
            frame = int(row["frame"])
            keep = 19 <= frame <= expected[row["sequence"]] - 8
            if bool(int(row["valid"])) != keep:
                raise ValueError("Calibration valid mask differs from common frame protocol")
        fit = json.loads(fit_path.read_text())
        for variant in metrics["variants"]:
            scores = np.array([float(r[variant]) for r in rows if r["valid"] == "1"])
            if not len(scores) or not np.isfinite(scores).all():
                raise ValueError("Nonfinite or missing normal calibration scores")
            actual = float(np.quantile(scores, .99))
            saved = fit["calibration"][variant]["threshold"]
            if not np.isclose(actual, saved, atol=1e-9, rtol=0):
                raise ValueError("Published threshold differs from normal q99")
            checks.append({"backbone": metrics["backbone"], "mode": metrics["mode"],
                "device": metrics["device"], "seed": metrics["seed"], "variant": variant,
                "normal_frames": len(scores), "stored_threshold": saved,
                "recomputed_threshold": actual, "absolute_error": abs(actual - saved),
                "calibration_csv_sha256": digest(csv_path), "normal_fit_sha256": digest(fit_path)})
    if not checks:
        raise ValueError("No completed experiments to validate")
    result = {"status": "passed", "thresholds_checked": len(checks), "absolute_tolerance": 1e-9,
        "method": "Normal calibration split and common mask checked; q99 of published calibrated scores",
        "scope": "Thresholds only; does not verify raw encoder features or real-time performance",
        "checks": checks}
    (args.root / "calibration_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "thresholds_checked": len(checks)}))


if __name__ == "__main__":
    main()
