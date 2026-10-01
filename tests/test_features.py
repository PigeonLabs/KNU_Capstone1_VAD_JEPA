import numpy as np
from PIL import Image
from ipad_jepa.features import ClipDataset, anchors


def test_online_fit_covers_initial_phase_but_test_never_pads():
    assert anchors(100,"online","fit")[0]==0
    assert anchors(100,"online","test")[0]==15
    assert anchors(100,"offline","test").tolist()==list(range(8,93))


def test_actual_reader_does_not_read_future(tmp_path):
    for i in range(20):
        Image.new("RGB",(8,8),(i,i,i)).save(tmp_path/f"{i:03}.jpg")
    data=ClipDataset(tmp_path,np.array([15]),"online",size=16)
    before,_=data[0]
    Image.new("RGB",(8,8),(255,0,0)).save(tmp_path/"016.jpg")
    after,_=ClipDataset(tmp_path,np.array([15]),"online",size=16)[0]
    np.testing.assert_array_equal(before.numpy(),after.numpy())
    assert before.shape==(3,16,16,16)
