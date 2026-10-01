import numpy as np
import pytest
from ipad_jepa.alignment import align


def test_matched_labels_keep_all_indices():
    gt=np.array([0,0,1,1,0])
    label,known,candidates=align(gt,5)
    np.testing.assert_array_equal(label,gt)
    assert known.all() and list(candidates)==[0]


@pytest.mark.parametrize('frames',[9,11])
def test_one_frame_mismatch_masks_transition_uncertainty(frames):
    gt=np.array([0]*4+[1]*4+[0]*2)
    label,known,candidates=align(gt,frames)
    assert not known[3] and not known[4] and not known[7] and not known[8]
    assert label[2]==0 and label[5]==1
    # A boundary at any location causing a single insertion/deletion yields
    # t/t-1 or t/t+1 label indices: consensus stays identical under every edit.
    for offset,values in candidates.items():
        np.testing.assert_array_equal(values[known],label[known])
    assert np.all(label[~known]==-1)
    with pytest.raises(ValueError,match='only supports'):
        align(gt,13)
