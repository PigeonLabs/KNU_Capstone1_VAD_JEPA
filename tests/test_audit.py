import numpy as np
import pytest
from PIL import Image
from ipad_jepa.audit import split_cycles, inspect_frames
from ipad_jepa.upload_guard import inspect


@pytest.mark.parametrize("n,expected", [(34,(23,5,6)), (30,(21,5,4)), (22,(15,4,3)), (25,(17,4,4))])
def test_splits_are_disjoint_and_repeatable(n, expected):
    ids = [f"{i:03}" for i in range(n)]
    groups = split_cycles(ids, calibration_count=expected[1])
    assert tuple(map(len, groups.values())) == expected
    assert sorted(sum(groups.values(), [])) == ids
    assert groups == split_cycles(ids[::-1], calibration_count=expected[1])
    assert groups != split_cycles(ids, seed=43, calibration_count=expected[1])


def test_numeric_order_and_missing_frames(tmp_path):
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        Image.new("RGB", (256,256)).save(tmp_path / f"{i}.jpg")
    result = inspect_frames(tmp_path)
    assert result["last_frame"] == "10.jpg"
    (tmp_path / "5.jpg").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        inspect_frames(tmp_path)


@pytest.mark.parametrize("name", ["weights/model.pth", "data/frame.jpg", "cache/features.npz", ".env"])
def test_upload_guard_blocks_artifacts(tmp_path, name):
    from pathlib import Path
    relative = Path(name)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    assert inspect(path, relative)
