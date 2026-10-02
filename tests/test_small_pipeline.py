"""B cache boundaries reject L features, stale inputs and non-normal fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ipad_jepa.features import file_hash
from ipad_jepa.small_data import FeatureSequence, sequences, normal_candidates
from ipad_jepa.small_features import sequence_spec
from ipad_jepa.small_train_phase import load_normal

SOURCE = Path(__file__).resolve().parents[1]/'src/ipad_jepa'


def cache_fixture(tmp_path, split='fit', sequence='01', dimension=768):
    row = dict(device='R01', sequence=sequence, partition='training', frames=32,
               split=split, names_sha256='audited-names', frames_content_sha256='audited-content')
    identity = dict(backbone='dinov3-b', mode='online', clip_frames=16, feature_dimension=768,
                    weights_sha256='verified-weights', upstream_commit='fixed-upstream', image_size=384,
                    adapter_sha256=file_hash(SOURCE/'small_backbones.py'), reader_sha256=file_hash(SOURCE/'small_features.py'),
                    base_reader_sha256=file_hash(SOURCE/'features.py'), temporal_code_sha256=file_hash(SOURCE/'temporal.py'),
                    preprocessing='RGB full-frame PIL bilinear resize; ImageNet mean/std', feature_dtype='float16 from BF16 inference')
    targets, spec = sequence_spec(row, 'online', identity)
    folder = tmp_path/'R01'/'training'/sequence
    folder.mkdir(parents=True)
    np.save(folder/'targets.npy', targets, allow_pickle=False)
    np.save(folder/'patch.npy', np.zeros((len(targets), 576, dimension), np.float16), allow_pickle=False)
    np.save(folder/'global.npy', np.full((len(targets), dimension), 2, np.float16), allow_pickle=False)
    write_meta(folder, spec)
    return row, folder, spec


def write_meta(folder, spec):
    (folder/'meta.json').write_text(json.dumps(dict(status='complete', spec=spec,
        fingerprint=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())))


def test_b_phase_input_keeps_normal_split_and_original_phase_bins(tmp_path):
    fit, _, _ = cache_fixture(tmp_path)
    cal, _, _ = cache_fixture(tmp_path, 'calibration', '02')
    # A test row with no cache must never be opened while loading normal inputs.
    test = dict(fit, partition='testing', split='test', sequence='missing-test')
    x, labels, positions = load_normal(tmp_path, [fit, cal, test], 'fit')
    assert x.shape == (8, 768)
    np.testing.assert_array_equal(positions, np.arange(0,32,4)/32)
    np.testing.assert_array_equal(labels, np.floor(200*positions).astype(np.int64))
    vx, _, vpositions = load_normal(tmp_path, [fit, cal, test], 'calibration')
    assert vx.shape == (17, 768)
    np.testing.assert_array_equal(vpositions, np.arange(15,32)/32)


def test_b_cache_rejects_actual_l_width_even_with_b_metadata(tmp_path):
    row, _, _ = cache_fixture(tmp_path, dimension=1024)
    with pytest.raises(ValueError, match='dimensions'):
        FeatureSequence(tmp_path, row)


def test_b_cache_rejects_changed_target_order(tmp_path):
    row, folder, _ = cache_fixture(tmp_path)
    np.save(folder/'targets.npy', np.arange(1,33,4), allow_pickle=False)
    with pytest.raises(ValueError, match='targets'):
        FeatureSequence(tmp_path, row)


def test_b_cache_rejects_stale_reader_even_with_consistent_fingerprint(tmp_path):
    row, folder, spec = cache_fixture(tmp_path)
    write_meta(folder, dict(spec, base_reader_sha256='wrong-reader'))
    with pytest.raises(ValueError, match='provenance'):
        FeatureSequence(tmp_path, row)


def test_b_cache_rejects_mixed_model_provenance(tmp_path):
    first, _, _ = cache_fixture(tmp_path)
    second, folder, spec = cache_fixture(tmp_path, sequence='02')
    write_meta(folder, dict(spec, backbone='vjepa21-b'))
    with pytest.raises(ValueError, match='mixed'):
        sequences(tmp_path, [first, second], 'fit')


def test_b_candidates_reject_normal_calibration_in_fit(tmp_path):
    row, _, _ = cache_fixture(tmp_path, 'calibration')
    value = FeatureSequence(tmp_path, row)
    with pytest.raises(ValueError, match='normal fit'):
        normal_candidates([value])
