import numpy as np
import pytest
import torch
from ipad_jepa.memory import PrototypeMemory,balanced_counts,balanced_indices
from ipad_jepa.torch_memory import TorchMemory


def test_balanced_quota_redistribution_and_video_sampling():
    rng=np.random.default_rng(5)
    assert balanced_counts(np.array([1,100,100]),21,rng).tolist()==[1,10,10]
    groups=np.repeat(np.arange(3),[5,100,100])
    selected=balanced_indices(groups,35,rng)
    assert len(np.unique(selected))==35
    assert np.bincount(groups[selected]).tolist()==[5,15,15]


@pytest.mark.parametrize('global_search',[False,True])
@pytest.mark.parametrize('k',[1,5])
def test_torch_scores_match_numpy_reference(global_search,k):
    rng=np.random.default_rng(8)
    x=rng.normal(size=(120,8)).astype(np.float32)
    phases=np.repeat(np.arange(4)/4+.01,30)
    memory=PrototypeMemory(bins=4,per_bin=7,dimensions=5).fit(x,phases)
    memory.temperature=.19
    scorer=TorchMemory(memory)
    query=x[:30].reshape(3,10,8)
    phi=np.array([.001,.999,.5],dtype=np.float32)
    expected=np.array([memory.frame_score(q,float(p),k,global_search) for q,p in zip(query,phi)])
    actual=scorer.frame_scores(torch.from_numpy(query),torch.from_numpy(phi),k,global_search).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)
