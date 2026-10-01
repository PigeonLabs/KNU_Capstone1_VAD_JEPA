import numpy as np
import pytest
import torch
from PIL import Image

from ipad_jepa.clip8_features import sequence_spec, check_outputs
from ipad_jepa.features import ClipDataset
from ipad_jepa.temporal import clip_indices, common_mask


def row(split):
    return {"frames": 40, "device": "R01", "sequence": "01", "partition": "training",
            "split": split, "names_sha256": "names", "frames_content_sha256": "contents"}


@pytest.mark.parametrize("mode,split,expected,padding", [
    ("offline", "fit", list(range(4, 37, 4)), False),
    ("offline", "calibration", list(range(4, 37)), False),
    ("online", "fit", list(range(0, 40, 4)), True),
    ("online", "calibration", list(range(7, 40)), False),
    ("online", "test", list(range(7, 40)), False),
])
def test_clip8_boundaries_and_primary_comparison_mask(mode, split, expected, padding):
    targets, spec = sequence_spec(row(split), mode, {"clip_frames": 8, "mode": mode})
    assert targets.tolist() == expected
    assert spec["padding"] is padding
    for target in targets:
        ids = clip_indices(int(target), 40, mode, 8, padding)
        assert len(ids) == 8
        if mode == "online":
            assert ids.max() <= target
    if split != "fit":
        # Accuracy remains paired to the original 16-frame protocol, including its right boundary.
        assert targets[common_mask(40)[targets]].tolist() == list(range(19, 33))


def test_actual_dataset_reads_eight_frames_and_only_normal_fit_can_pad(tmp_path):
    for index in range(12):
        Image.fromarray(np.full((3, 3, 3), index * 10, dtype=np.uint8)).save(tmp_path / f"{index}.jpg")
    dataset = ClipDataset(tmp_path, np.array([7, 11]), "online", size=16, frames=8)
    video, target = dataset[1]
    assert video.shape == (3, 8, 16, 16) and target == 11
    means = video.mean((0, 2, 3))
    assert bool(torch.all(torch.diff(means) > 0))
    assert torch.equal(video[:, -1], dataset.frame(11))
    assert torch.equal(video[:, 0], dataset.frame(4))
    with pytest.raises(ValueError, match="Incomplete clip"):
        ClipDataset(tmp_path, np.array([0]), "online", size=16, frames=8)[0]
    padded, _ = ClipDataset(tmp_path, np.array([0]), "online", size=16, frames=8, padding=True)[0]
    assert all(torch.equal(padded[:, 0], padded[:, i]) for i in range(8))


def test_reject_clip16_identity_or_different_mode():
    with pytest.raises(ValueError, match="fingerprint"):
        sequence_spec(row("fit"), "online", {"clip_frames": 16, "mode": "online"})
    with pytest.raises(ValueError, match="fingerprint"):
        sequence_spec(row("fit"), "online", {"clip_frames": 8, "mode": "offline"})


def test_output_gate_rejects_nonfinite_or_incompatible_backbone():
    patches, global_features = torch.zeros(1, 576, 1024), torch.zeros(1, 1024)
    check_outputs(patches, global_features, 1)
    patches[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        check_outputs(patches, global_features, 1)
    with pytest.raises(ValueError, match="dimensions"):
        check_outputs(torch.zeros(1, 576, 768), torch.zeros(1, 768), 1)
