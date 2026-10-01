from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ipad_jepa.history_ablation import SavedInference
from ipad_jepa.memory import PrototypeMemory
from ipad_jepa.neighbour_ablation import infer_pairs, recalibrate, score_cached_sequences, scored
from ipad_jepa.torch_memory import TorchMemory


def make_scorer(seed):
    rng = np.random.default_rng(seed)
    memory = PrototypeMemory(dimensions=8, temperature=.3)
    memory.pca = SimpleNamespace(mean_=rng.normal(size=12).astype(np.float32),
        components_=np.eye(8, 12, dtype=np.float32))
    memory.prototypes = rng.normal(size=(16, 128, 8)).astype(np.float32)
    memory.prototypes /= np.linalg.norm(memory.prototypes, axis=-1, keepdims=True)
    return TorchMemory(memory)


def test_paired_seed_batching_uses_each_fixed_bank_and_phase():
    rng = np.random.default_rng(7)
    sequence = SimpleNamespace(targets=np.arange(8), patches=rng.normal(size=(8, 4, 12)).astype(np.float16))
    contexts = {s: {'scorer': make_scorer(s)} for s in [0, 1, 2]}
    phase = {s: rng.uniform(size=8) for s in contexts}
    actual = score_cached_sequences(sequence, contexts, phase, 'cpu', batch=3)
    for seed, context in contexts.items():
        for k in [1, 5, 10]:
            with torch.inference_mode():
                expected = context['scorer'].frame_scores(torch.from_numpy(sequence.patches.astype(np.float32)),
                    torch.from_numpy(phase[seed].astype(np.float32)), k=k).numpy()
            np.testing.assert_allclose(actual[seed][k], expected, rtol=1e-5, atol=1e-6)
    assert not np.array_equal(actual[0][5], actual[1][5])


def test_k1_is_existing_hard_neighbour_score():
    scorer = make_scorer(4)
    x = torch.arange(48, dtype=torch.float32).reshape(1, 4, 12) / 48
    phase = torch.tensor([.7])
    with torch.inference_mode():
        nearest = scorer.nearest(x, phase, k=1)[2][..., 0]
        actual = scorer.frame_scores(x, phase, k=1)
    torch.testing.assert_close(actual, nearest.max(dim=1).values)


def test_online_prefix_and_eof_are_inference_only():
    n = 75; targets = np.arange(15, n)
    phase = np.mod(targets / 100 + .04 * np.sin(targets), 1)
    sequence = SavedInference(n, targets, phase, np.zeros(len(targets)))
    normal = infer_pairs(sequence, np.linspace(.1, .6, len(targets)), 100)
    cal = recalibrate([normal], [sequence]); cal.threshold = -1e6
    whole = scored(sequence, normal, cal)
    partial = SavedInference(60, targets[targets < 60], phase[targets < 60], np.zeros(sum(targets < 60)))
    prefix = infer_pairs(partial, np.linspace(.1, .6, len(targets))[targets < 60], 100)
    short = scored(partial, prefix, cal)
    for index in range(4):
        np.testing.assert_array_equal(whole[index][targets < 60], short[index])
    assert whole[1][-7:].all() and whole[2][-7:].all()
    assert not whole[1][:4].any()
    assert not whole[2][:6].any() and whole[2][6]


def test_invalid_bank_seed_alignment_is_rejected():
    sequence = SimpleNamespace(targets=np.arange(3), patches=np.zeros((3, 4, 12), dtype=np.float16))
    with pytest.raises(ValueError, match='paired seed'):
        score_cached_sequences(sequence, {0: {'scorer': make_scorer(0)}}, {1: np.ones(3)}, 'cpu')


def test_restoring_bank_preserves_serialized_pca_layout(tmp_path):
    from ipad_jepa.neighbour_ablation import load_scorer
    components = np.zeros((256, 1024), dtype=np.float32, order='F')
    prototypes = np.zeros((16, 128, 256), dtype=np.float32)
    path = tmp_path / 'memory.npz'
    np.savez(path, mean=np.zeros(1024, dtype=np.float32), components=components,
        prototypes=prototypes, temperature=1., cycle_length=100.)
    scorer = load_scorer(path, {'temperature': 1., 'cycle_length_fit_median': 100.}, 'cpu')
    assert scorer.components.stride() == tuple(s // 4 for s in components.strides)
    assert scorer.prototypes.stride() == tuple(s // 4 for s in prototypes.strides)
