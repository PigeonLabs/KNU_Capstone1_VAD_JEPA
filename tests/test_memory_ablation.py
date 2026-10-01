import numpy as np
import pytest
from ipad_jepa.memory import PrototypeMemory
from ipad_jepa.memory_ablation import prefix_memory


@pytest.mark.parametrize('bins', [4, 8])
def test_larger_kcenter_prefix_matches_independently_fitted_smaller_banks(bins):
    """Budget extension must not perturb later bins' RNG draws or PCA candidates."""
    rng = np.random.default_rng(14)
    features = rng.normal(size=(1024, 12)).astype(np.float32)
    phase = np.tile(np.arange(128) / 128, 8)
    groups = np.repeat(np.arange(8), 128)
    options = dict(bins=bins, dimensions=8, seed=6, pca_samples=512, candidate_limit=128)
    largest = PrototypeMemory(per_bin=16, **options).fit(features, phase, groups)
    for count in [4, 8, 16]:
        independent = PrototypeMemory(per_bin=count, **options).fit(features, phase, groups)
        prefix = prefix_memory(largest, count)
        np.testing.assert_array_equal(prefix.pca.mean_, independent.pca.mean_)
        np.testing.assert_array_equal(prefix.pca.components_, independent.pca.components_)
        np.testing.assert_array_equal(prefix.prototypes, independent.prototypes)
        assert prefix.prototypes.strides == independent.prototypes.strides
    original = largest.prototypes.copy()
    prefix_memory(largest, 4).prototypes[:] = 0
    np.testing.assert_array_equal(largest.prototypes, original)


@pytest.mark.parametrize('count', [0, 17])
def test_prefix_rejects_invalid_budget_without_repeating_prototypes(count):
    memory = PrototypeMemory(per_bin=16)
    with pytest.raises(ValueError, match='prefix'):
        prefix_memory(memory, count)
