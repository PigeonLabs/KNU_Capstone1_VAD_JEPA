import importlib.util
import json
from pathlib import Path
import sys
import pytest


@pytest.fixture
def downloader():
    path=Path(__file__).parents[1]/'scripts/download_models.py'
    spec=importlib.util.spec_from_file_location('download_models',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrupt_download_is_not_promoted_to_checkpoint(tmp_path,monkeypatch,downloader):
    class Response:
        status=200
        headers={'Content-Length':'3'}
        remaining=b'bad'
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,count):
            data=self.remaining; self.remaining=b''
            return data
    monkeypatch.setattr(downloader,'urlopen',lambda *args,**kwargs:Response())
    with pytest.raises(ValueError,match='SHA256'):
        downloader.download('dinov3-l',tmp_path,True)
    target=tmp_path/downloader.MODELS['dinov3-l'][1]
    assert not target.exists()
    assert target.with_suffix('.pth.part').read_bytes()==b'bad'


def test_cached_download_preserves_actual_source(tmp_path,monkeypatch,downloader):
    weights=tmp_path/'weights'; weights.mkdir()
    filename=downloader.MODELS['dinov3-l'][1]
    (weights/filename).write_bytes(b'checkpoint')
    metadata=tmp_path/'metadata'; metadata.mkdir()
    digest='8aa4cbdd'+'0'*56
    recorded={'source':downloader.USER_PROVIDED_DINO,'source_type':'user_provided_mirror',
              'sha256':digest,'status':'hash_prefix_verified'}
    (metadata/'dinov3-l.json').write_text(json.dumps(recorded))
    monkeypatch.setattr(downloader,'sha256',lambda path:digest)
    monkeypatch.setattr(sys,'argv',['download_models','--model','dinov3-l','--out',str(weights),'--metadata',str(metadata)])
    downloader.main()
    reused=json.loads((metadata/'dinov3-l.json').read_text())
    assert reused['source']==recorded['source']
    assert reused['source_type']=='user_provided_mirror'
    assert reused['reused_local_file']
