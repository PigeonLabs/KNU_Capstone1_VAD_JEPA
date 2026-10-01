import hashlib
import json
import numpy as np
import pytest
from ipad_jepa.train_phase import load_normal


def row(sequence='01', split='fit', partition='training'):
    return {'device':'R01','sequence':sequence,'partition':partition,'split':split,
            'frames':100,'frames_content_sha256':'contents','names_sha256':'names'}


def make_cache(folder, split='fit', weights='weights'):
    folder.mkdir(parents=True)
    spec={'split':split,'device':'R01','sequence':folder.name,'partition':'training',
          'frames':100,'frame_content_sha256':'contents','frame_names_sha256':'names',
          'backbone':'model','mode':'offline','weights_sha256':weights,'upstream_commit':'commit',
          'adapter_sha256':'adapter','reader_sha256':'reader','image_size':384,
          'clip_frames':16,'fit_stride':4,'rows':2}
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    (folder/'meta.json').write_text(json.dumps({'spec':spec,'status':'complete','fingerprint':fingerprint}))
    np.save(folder/'targets.npy',[8,12])
    np.save(folder/'global.npy',np.ones((2,1024),dtype=np.float16))


def test_phase_loader_excludes_test_and_diagnostic(tmp_path):
    make_cache(tmp_path/'R01/training/01')
    rows=[row(),row('02','diagnostic'),row('03','fit','testing')]
    x,y,phase=load_normal(tmp_path,rows,'fit')
    assert x.shape==(2,1024)
    assert y.tolist()==[16,24]
    np.testing.assert_allclose(phase,[.08,.12])


def test_phase_loader_rejects_cache_split_mismatch(tmp_path):
    make_cache(tmp_path/'R01/training/01',split='calibration')
    with pytest.raises(ValueError,match='Invalid normal'):
        load_normal(tmp_path,[row()],'fit')


def test_phase_loader_rejects_changed_source_and_mixed_weights(tmp_path):
    make_cache(tmp_path/'R01/training/01')
    changed=row(); changed['frames_content_sha256']='changed'
    with pytest.raises(ValueError,match='match manifest'):
        load_normal(tmp_path,[changed],'fit')
    make_cache(tmp_path/'R01/training/02',weights='other')
    with pytest.raises(ValueError,match='Mixed feature'):
        load_normal(tmp_path,[row(),row('02')],'fit')
