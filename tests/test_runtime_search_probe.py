"""Verify diagnostic search arithmetic independently of actual GPU context capture."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_runtime_search import search_capture, file_hash
from ipad_jepa.torch_memory import TorchMemory


def test_probe_source_hash_accepts_script_and_saved_json_string_paths(tmp_path):
    import hashlib
    path = tmp_path / 'input.json'
    path.write_bytes(b'{"status":"failed"}')
    assert file_hash(str(path)) == file_hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def memory():
    rng = np.random.default_rng(41)
    components = np.asfortranarray(rng.normal(size=(3, 4)).astype(np.float32))
    prototypes = rng.normal(size=(4, 3, 3)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=-1, keepdims=True)
    scorer = TorchMemory(SimpleNamespace(bins=4, dimensions=3, temperature=.2,
        pca=SimpleNamespace(mean_=np.zeros(4, dtype=np.float32), components_=components),
        prototypes=prototypes))
    features = torch.tensor(rng.normal(size=(2, 43, 4)).astype(np.float32))
    return scorer, features


def test_exact_search_capture_and_global_ids_with_wraparound_bins():
    scorer, features = memory()
    phase = torch.tensor([0., .999], dtype=torch.float32)
    with torch.inference_mode():
        result = search_capture(scorer, features, phase)
    torch.testing.assert_close(result['fast_frame_score'], scorer.frame_scores(features, phase), rtol=0, atol=0)
    torch.testing.assert_close(result['phase_bins'], torch.tensor([[0, 1, 3], [0, 2, 3]]))
    assert result['fast_top_patch_ids'].shape == (2, 3)  # ceil(43*.05)
    for batch, ids in enumerate(result['fast_ids']):
        assert set((ids // 3).flatten().tolist()) <= set(result['phase_bins'][batch].tolist())


def test_stable_rerank_matches_independent_numpy_explicit_distances():
    scorer, features = memory()
    phase = torch.tensor([.3, .8])
    with torch.inference_mode():
        result = search_capture(scorer, features, phase)
    z = result['query'].numpy().astype(np.float64)
    bank = scorer.prototypes[result['phase_bins']].numpy().reshape(2, 9, 3).astype(np.float64)
    distance = ((z[:, :, None] - bank[:, None]) ** 2).sum(-1)
    local = np.argsort(distance, axis=-1)[..., :5]
    expected = np.take_along_axis(distance, local, -1)
    np.testing.assert_allclose(result['stable_distances'].numpy(), expected, rtol=1e-14, atol=1e-14)
    bins = result['phase_bins'].numpy()
    global_ids = (bins[:, :, None] * 3 + np.arange(3)).reshape(2, 9)
    np.testing.assert_array_equal(result['stable_ids'].numpy(), global_ids[np.arange(2)[:, None, None], local])
    weights = np.exp(-expected / .2 - (-expected / .2).max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    neighbours = bank[np.arange(2)[:, None, None], local]
    scores = ((z - (neighbours * weights[..., None]).sum(-2)) ** 2).sum(-1)
    np.testing.assert_allclose(result['stable_frame_score'].numpy(), np.sort(scores, axis=1)[:, -3:].mean(1),
                               rtol=1e-13, atol=1e-13)


def test_cancellation_diagnostic_keeps_fast_score_and_fp32_query_unchanged():
    bank = np.array([[[1., (index + 1) * 1e-5] for index in range(3)]
                     for _ in range(4)], dtype=np.float32)
    scorer = TorchMemory(SimpleNamespace(bins=4, dimensions=2, temperature=.2,
        pca=SimpleNamespace(mean_=np.zeros(2, dtype=np.float32), components_=np.eye(2, dtype=np.float32)),
        prototypes=bank))
    features, phase = torch.tensor([[[1., 0.]]]), torch.tensor([.1])
    original = scorer.frame_scores(features, phase).clone()
    with torch.inference_mode():
        result = search_capture(scorer, features, phase)
    assert result['distance_rounding_max_abs'] > 0
    assert torch.all(result['fast_distances'] == 0)
    assert torch.all(result['stable_distances'] > 0)
    assert result['query'].dtype == torch.float32
    assert result['stable_distances'].dtype == torch.float64
    torch.testing.assert_close(scorer.frame_scores(features, phase), original, rtol=0, atol=0)


@pytest.mark.parametrize('which', ['double_features', 'double_phase', 'nan_features', 'bad_phase', 'wrong_phase_shape'])
def test_incompatible_capture_inputs_are_rejected(which):
    scorer, features = memory()
    phase = torch.tensor([.3, .8])
    if which == 'double_features': features = features.double()
    if which == 'double_phase': phase = phase.double()
    if which == 'nan_features': features[0, 0, 0] = float('nan')
    if which == 'bad_phase': phase[0] = 1.
    if which == 'wrong_phase_shape': phase = phase[:1]
    with pytest.raises(ValueError, match='Finite FP32'):
        search_capture(scorer, features, phase)


def test_capture_refuses_a_changed_primary_scorer(monkeypatch):
    scorer, features = memory()
    original = scorer.frame_scores
    monkeypatch.setattr(scorer, 'frame_scores', lambda *a, **k: original(*a, **k) + .01)
    with pytest.raises(ValueError, match='unchanged scorer.frame_scores'):
        search_capture(scorer, features, torch.tensor([.3, .8]))
