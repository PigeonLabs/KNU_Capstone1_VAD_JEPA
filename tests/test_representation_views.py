import json
from types import SimpleNamespace

import numpy as np
import pytest

from ipad_jepa.representation_data import mean_sequence


def sequence(tmp_path):
    folder = tmp_path / "original"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"fingerprint": "original"}))
    np.save(folder / "targets.npy", np.array([19, 20], dtype=np.int64))
    patches = np.zeros((2, 576, 1024), dtype=np.float16)
    patches[0, :288] = 2; patches[0, 288:] = 4; patches[1] = 7
    return SimpleNamespace(row={"partition": "training", "sequence": "01"},
                           targets=np.array([19, 20]), meta={"fingerprint": "original"}, folder=folder,
                           patches=patches, global_features=np.full((2, 1024), -99, dtype=np.float16),
                           identity={"reader_sha256": "original_reader"})


def test_mean_view_preserves_original_phase_input_and_provenance(tmp_path):
    original = sequence(tmp_path)
    view = mean_sequence(original, tmp_path / "views")
    assert view.global_features is original.global_features
    assert view.identity is original.identity and view.meta is original.meta
    assert view.patches.shape == (2, 1, 1024) and view.patches.dtype == np.float32
    assert np.all(view.patches[0] == 3) and np.all(view.patches[1] == 7)
    assert np.all(view.global_features == -99)
    again = mean_sequence(original, tmp_path / "views")
    assert np.array_equal(view.patches, again.patches)
    assert again.mean_source["identity"]["source_fingerprint"] == "original"


def test_mean_view_rejects_altered_source_and_altered_queries(tmp_path):
    original = sequence(tmp_path)
    root = tmp_path / "views"
    mean_sequence(original, root)
    original.meta = {"fingerprint": "changed"}
    with pytest.raises(ValueError, match="Stale"):
        mean_sequence(original, root)
    original.meta = {"fingerprint": "original"}
    data = np.load(root / "training/01/mean.npy", mmap_mode="r+")
    data[0, 0, 0] = 123; data.flush(); del data
    with pytest.raises(ValueError, match="changed"):
        mean_sequence(original, root)


def test_partial_mean_view_is_not_implicitly_overwritten(tmp_path):
    original = sequence(tmp_path)
    (tmp_path / "views/training/01").mkdir(parents=True)
    with pytest.raises(ValueError, match="Partial"):
        mean_sequence(original, tmp_path / "views")
