"""Sampling, leakage and mean semantics for the paired dense-fit ablation."""
import numpy as np
import pytest

from ipad_jepa.representation import candidate_plan, capacity, fit_targets, local_spatial_mean


def rows():
    return [{"device": "R01", "partition": "training", "split": "fit", "sequence": f"{i:02d}", "frames": 320}
            for i in range(16)]


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_dense_capacity_solves_mean_candidate_shortage(mode):
    fit = rows()
    assert not capacity(fit, mode, 4)["enough_for_128_per_bin"]
    assert capacity(fit, mode, 1)["enough_for_128_per_bin"]
    targets = fit_targets(320, mode)
    assert np.array_equal(targets, np.arange(8, 313) if mode == "offline" else np.arange(320))


@pytest.mark.parametrize("queries", [1, 576])
def test_candidate_coordinates_are_observed_unique_balanced_and_reproducible(queries):
    fit = rows()
    coords, phase, sampling = candidate_plan(fit, "online", 2, limit=160, queries=queries)
    again = candidate_plan(fit, "online", 2, limit=160, queries=queries)
    assert np.array_equal(coords, again[0]) and np.array_equal(phase, again[1])
    assert len(np.unique(coords, axis=0)) == len(coords)
    assert np.all((coords[:, 1] >= 0) & (coords[:, 1] < 320))
    assert np.all((coords[:, 2] >= 0) & (coords[:, 2] < queries))
    assert np.array_equal(phase, coords[:, 1] / 320)
    assert all(s["tokens"] == 160 and set(s["per_video"].values()) == {10} for s in sampling)
    assert np.bincount(np.floor(phase * 16).astype(int), minlength=16).tolist() == [160] * 16


def test_mean_uses_target_local_spatial_patches_in_float32():
    patches = np.zeros((2, 576, 1024), dtype=np.float16)
    patches[0, :288] = 2
    patches[0, 288:] = 4
    patches[1] = -5
    result = local_spatial_mean(patches)
    assert result.shape == (2, 1, 1024) and result.dtype == np.float32
    assert np.all(result[0] == 3) and np.all(result[1] == -5)
    patches[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        local_spatial_mean(patches)


def test_sampling_rejects_calibration_test_duplicate_and_insufficient_fit():
    for partition, split in [("testing", "test"), ("training", "calibration"), ("training", "diagnostic")]:
        bad = rows(); bad[0] = {**bad[0], "partition": partition, "split": split}
        with pytest.raises(ValueError, match="normal fit"):
            candidate_plan(bad, "online", 0)
    with pytest.raises(ValueError, match="Duplicate"):
        candidate_plan(rows() + [rows()[0]], "online", 0)
    with pytest.raises(ValueError, match="Insufficient"):
        candidate_plan(rows()[:1], "online", 0)


def test_sampled_patch_store_matches_full_normal_sampler_without_full_dense_cache():
    fit = rows()
    coordinates, phase, sampling = candidate_plan(fit, "offline", 1, limit=160)
    # Independently enumerate video/phase capacities and replay RNG quotas/choices.
    from ipad_jepa.memory import balanced_counts
    rng = np.random.default_rng(1)
    expected = []
    for b in range(16):
        ids = [np.flatnonzero(np.floor(fit_targets(r["frames"], "offline") / r["frames"] * 16).astype(int) == b) for r in fit]
        counts = balanced_counts(np.array([len(i) * 576 for i in ids]), 160, rng)
        for video, (ci, count) in enumerate(zip(ids, counts)):
            chosen = rng.choice(len(ci) * 576, int(count), replace=False)
            expected.extend(zip(np.full(int(count), video), ci[chosen // 576], chosen % 576))
    assert np.array_equal(coordinates, np.array(expected))
    assert len(phase) == sum(s["tokens"] for s in sampling)
