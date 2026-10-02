"""Independent representation audits reject changed pooling, phase heads and causal masks."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_representation_ablation import audit_mean_array, audit_pair


def write_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def pair(tmp_path):
    paths = [tmp_path / "patch", tmp_path / "global_mean"]
    fit = {"seed": 0, "backbone": "dinov3-l", "mode": "online", "device": "R01", "dense_fit_source_sha256": "dense",
           "phase_checkpoint_sha256": "shared", "phase_selected_epoch": 3, "cycle_length_fit_median": 160,
           "normal_calibration_cache_fingerprints": ["cal"], "original_calibration_cache_identity": {"reader": "original"}}
    for i, path in enumerate(paths):
        path.mkdir()
        (path / "normal_fit.json").write_text(json.dumps({**fit, "representation": path.name}))
        (path / "phase_training.json").write_text('{"shared": true}')
        (path / "phase_training.csv").write_text('epoch,normal_calibration_ce\n1,2\n')
        calibration = [{"sequence": "01", "frame": str(t), "valid": "1", "relative_phase": str(t / 160),
                        "predicted_phase": str(t / 160), "time_raw": "0.1", "feature_raw": str(i + .5)} for t in [19, 20]]
        write_csv(path / "normal_calibration.csv", calibration)
        test = [{"frame": str(t), "label": str(j), "valid": "1", "inference_valid": "1", "phase": str(t / 160),
                 "time_raw": "0.1", "feature_raw": str(i + .5), "score": str(i + .3)} for j, t in enumerate([19, 20])]
        write_csv(path / "P3/01.csv", test)
    return paths


def test_independent_pooling_reads_all_patches_and_rejects_changed_mean(tmp_path):
    patches = np.zeros((2, 576, 1024), dtype=np.float16)
    patches[0, :288] = 2; patches[0, 288:] = 4; patches[1] = -5
    means = np.stack([np.full((1, 1024), 3, dtype=np.float32), np.full((1, 1024), -5, dtype=np.float32)])
    np.save(tmp_path / "patch.npy", patches); np.save(tmp_path / "mean.npy", means)
    assert audit_mean_array(tmp_path / "patch.npy", tmp_path / "mean.npy", 2) == 2
    means[1, 0, 1023] += 1
    np.save(tmp_path / "mean.npy", means)
    with pytest.raises(AssertionError):
        audit_mean_array(tmp_path / "patch.npy", tmp_path / "mean.npy", 2)


def test_independent_pooling_rejects_context_feature_or_wrong_dtype(tmp_path):
    np.save(tmp_path / "patch.npy", np.zeros((2, 576, 1024), dtype=np.float16))
    np.save(tmp_path / "mean.npy", np.zeros((2, 1024), dtype=np.float16))
    with pytest.raises(ValueError, match="geometry"):
        audit_mean_array(tmp_path / "patch.npy", tmp_path / "mean.npy", 2)


def test_pair_allows_changed_feature_scores_with_shared_phase_and_time(tmp_path):
    before, after = pair(tmp_path)
    audit_pair(before, after)


def test_pair_rejects_independently_changed_phase_head(tmp_path):
    before, after = pair(tmp_path)
    fit = json.loads((after / "normal_fit.json").read_text()); fit["phase_checkpoint_sha256"] = "different"
    (after / "normal_fit.json").write_text(json.dumps(fit))
    with pytest.raises(ValueError, match="same normal data and phase head"):
        audit_pair(before, after)


@pytest.mark.parametrize("field", ["phase", "inference_valid"])
def test_pair_rejects_changed_prediction_or_gt_dependent_inference_mask(tmp_path, field):
    before, after = pair(tmp_path)
    with (after / "P3/01.csv").open() as stream:
        trace = list(csv.DictReader(stream))
    trace[0][field] = "0"
    write_csv(after / "P3/01.csv", trace)
    with pytest.raises((ValueError, AssertionError)):
        audit_pair(before, after)
