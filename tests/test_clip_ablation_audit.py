from pathlib import Path
import sys

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
from summarize_clip_ablation import audit_trace, audit_calibration


def fixture(mode="online", frames=8, test=False):
    n = 40
    start = frames - 1 if mode == "online" else frames // 2
    stop = n if mode == "online" else n - frames // 2 + 1
    ids = np.arange(start, stop)
    feature = .2 + (ids % 7) / 10
    common = (ids >= 19) & (ids <= n - 8)
    pairs = np.stack([feature[common], np.zeros(common.sum())], axis=1)
    median = np.median(pairs, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(pairs - median), axis=0), 1e-6)
    params = {"median": median.tolist(), "mad_scale": scale.tolist(),
              "component_thresholds": np.quantile(pairs, .99, axis=0).tolist(),
              "threshold": float(np.quantile(((pairs - median) / scale).mean(axis=1), .99))}
    labels, known = np.zeros(n, dtype=int), np.ones(n, dtype=bool)
    labels[21:26] = 1
    known[23] = False
    rows = []
    for i, frame in enumerate(ids):
        time_raw = 0. if i >= 4 else float("nan")
        raw = np.array([feature[i], time_raw])
        score = float(((raw - median) / scale).mean())
        row = {"frame": str(frame), "feature_raw": str(feature[i]), "time_raw": str(time_raw),
               "valid": str(int(common[i] and (known[frame] if test else True)))}
        if test:
            row.update(phase=str(frame / n), label=str(labels[frame]), inference_valid=str(int(common[i])),
                       score=str(score), alarm="0", evidence_type="0")
        else:
            row.update(sequence="01", predicted_phase=str(frame / n), relative_phase=str(frame / n), P3=str(score))
        rows.append(row)
    return rows, {"frames": n, "sequence": "01", "device": "R01", "partition": "training", "split": "calibration"}, params, labels, known


@pytest.mark.parametrize("mode,frames", [("offline", 8), ("online", 8), ("offline", 16), ("online", 16)])
def test_auditor_accepts_both_complete_contexts_and_same_normal_mask(mode, frames):
    rows, info, params, _, _ = fixture(mode, frames)
    pairs = audit_trace(rows, info, mode, frames, 40, params)
    assert len(pairs) == 14
    np.testing.assert_allclose(pairs[:, 1], 0, rtol=0, atol=1e-12)


def test_auditor_rejects_extra_accuracy_frames_missing_target_and_wrong_phase_time():
    rows, info, _, _, _ = fixture()
    rows[0]["valid"] = "1"
    with pytest.raises(AssertionError):
        audit_trace(rows, info, "online", 8, 40)
    rows[0]["valid"] = "0"
    with pytest.raises(AssertionError):
        audit_trace(rows[1:], info, "online", 8, 40)
    rows[15]["time_raw"] = ".1"
    with pytest.raises(AssertionError):
        audit_trace(rows, info, "online", 8, 40)


def test_annotation_uncertainty_never_resets_the_batch_alarm():
    rows, info, params, labels, known = fixture(test=True)
    params["threshold"] = -100
    streak = 0
    for row in rows:
        keep = row["inference_valid"] == "1"
        streak = streak + 1 if keep else 0
        row["alarm"] = str(int(streak >= 3))
        raw = np.array([float(row["feature_raw"]), float(row["time_raw"])])
        row["evidence_type"] = str(int((raw > params["component_thresholds"]) @ [1, 2])) if np.isfinite(raw).all() else "0"
    audit_trace(rows, info, "online", 8, 40, params, labels, known)
    row = next(row for row in rows if row["frame"] == "23")
    assert row["valid"] == "0" and row["alarm"] == "1"
    row["alarm"] = "0"
    with pytest.raises(ValueError, match="GT must not control"):
        audit_trace(rows, info, "online", 8, 40, params, labels, known)


def test_normal_q99_audit_rejects_test_video_and_changed_threshold(tmp_path):
    import csv
    rows, info, params, _, _ = fixture()
    with (tmp_path / "normal_calibration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    assert audit_calibration(tmp_path, [info], "R01", "online", 8, 40, params) == 14
    params["threshold"] += .01
    with pytest.raises(AssertionError):
        audit_calibration(tmp_path, [info], "R01", "online", 8, 40, params)
    info["partition"] = "testing"
    with pytest.raises(ValueError, match="Normal calibration"):
        audit_calibration(tmp_path, [info], "R01", "online", 8, 40, params)
