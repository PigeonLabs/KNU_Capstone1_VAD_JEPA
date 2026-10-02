import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_runtime_matrix import implementations, planned_conditions, result_receipt


def test_runtime_plan_retains_every_model_mode_device_seed_and_adaptation():
    plan = planned_conditions()
    assert len(plan) == 96
    identities = {tuple(c[k] for k in ["backbone", "mode", "device", "seed", "adaptation"]) for c in plan}
    assert len(identities) == 96
    assert sum(len(implementations(c)) for c in plan) == 240
    for adaptation in ["frozen", "lora"]:
        assert sum(len(implementations(c)) for c in plan if c["adaptation"] == adaptation) == 120
    for condition in plan:
        assert implementations(condition)[:2] == [("bf16", "full"), ("bf16", "buffer")]
        if ("fp32", "reuse") in implementations(condition):
            assert (condition["backbone"], condition["mode"]) == ("dinov3-l", "online")
        assert ("bf16", "reuse") not in implementations(condition)


def test_runtime_completion_rejects_subset_video_claim(tmp_path):
    import json
    condition = planned_conditions()[0]
    report = {**condition, "status": "complete_measured_replay", "all_test_videos": True,
              "precision": "bf16", "implementation": "full", "arrival_fps": 30.,
              "test_bank_updates": False, "sequences": [{"sequence": "01"}]}
    (tmp_path / "runtime.json").write_text(json.dumps(report))
    manifest = [{"device": "R01", "partition": "testing", "sequence": sequence} for sequence in ["01", "02"]]
    with pytest.raises(ValueError, match="Every actual test video"):
        result_receipt(tmp_path, condition, "bf16", "full", manifest)


def test_runtime_completion_rejects_wrong_source_frame_inventory(tmp_path):
    import json
    condition = planned_conditions()[0]
    row = {"sequence": "01", "input_frames": 39, "frames_content_sha256": "frames", "label_sha256": "labels"}
    report = {**condition, "status": "complete_measured_replay", "all_test_videos": True,
              "precision": "bf16", "implementation": "full", "arrival_fps": 30.,
              "test_bank_updates": False, "sequences": [row]}
    (tmp_path / "runtime.json").write_text(json.dumps(report))
    manifest = [{"device": "R01", "partition": "testing", "sequence": "01", "frames": 40,
                 "frames_content_sha256": "frames", "label_sha256": "labels"}]
    with pytest.raises(ValueError, match="input frame count"):
        result_receipt(tmp_path, condition, "bf16", "full", manifest)
    report["sequences"][0]["input_frames"] = 40
    report["sequences"][0]["label_sha256"] = "other_labels"
    (tmp_path / "runtime.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="input sources"):
        result_receipt(tmp_path, condition, "bf16", "full", manifest)
