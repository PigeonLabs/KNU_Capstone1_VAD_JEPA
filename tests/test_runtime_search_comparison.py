"""Check independent arithmetic and reject incomplete/mismatched probe evidence."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_runtime_search_probe import (CONDITION, neighbor_changes, score_with_ids,
                                             validate_pins, verify_capture_pair, check_arrays)


def test_rank_permutation_is_not_a_membership_change():
    a = np.array([[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]])
    b = np.array([[[1, 0, 2, 4, 3], [5, 6, 7, 8, 10]]])
    assert neighbor_changes(a, b) == {'ordered_changed_patches': 2, 'membership_changed_patches': 1}


def test_explicit_projection_reselects_top_patches_and_preserves_id_permutation():
    # Closed-form equal prototype projection: residuals are exactly query^2.
    bank = np.zeros((6, 1), dtype=np.float32)
    query = np.arange(43, dtype=np.float32).reshape(1, 43, 1)
    ids = np.broadcast_to(np.arange(5), (1, 43, 5)).copy()
    result = score_with_ids(query, bank, ids, .2)
    np.testing.assert_array_equal(result['top_ids'], [[42, 41, 40]])  # ceil(43*.05)
    assert result['scores'][0] == (42**2 + 41**2 + 40**2) / 3
    np.testing.assert_array_equal(score_with_ids(query, bank, ids[..., ::-1], .2)['scores'], result['scores'])
    query[0, 0, 0] = 100
    assert score_with_ids(query, bank, ids, .2)['top_ids'][0, 0] == 0


def test_two_components_sum_to_stable_delta_without_freezing_top_patch_ids():
    bank = np.arange(10, dtype=np.float32).reshape(10, 1)
    a = np.zeros((1, 21, 1), dtype=np.float32)
    b = a.copy(); b[0, 4, 0] = 2
    ids_a = np.broadcast_to(np.arange(5), (1, 21, 5)).copy()
    ids_b = ids_a.copy(); ids_b[0, 4] = np.arange(5, 10)
    full = score_with_ids(a, bank, ids_a, 1.)['scores'][0]
    fixed = score_with_ids(b, bank, ids_a, 1.)['scores'][0]
    reuse = score_with_ids(b, bank, ids_b, 1.)['scores'][0]
    assert (fixed - full) + (reuse - fixed) == pytest.approx(reuse - full, abs=1e-14)
    assert score_with_ids(b, bank, ids_b, 1.)['top_ids'][0, 0] == 4


@pytest.mark.parametrize('fault', ['duplicate_ids', 'negative_ids', 'out_of_bank', 'nonfinite_query', 'zero_temperature'])
def test_invalid_counterfactual_inputs_rejected(fault):
    query = np.zeros((1, 3, 2), dtype=np.float32)
    bank = np.zeros((6, 2), dtype=np.float32)
    ids = np.broadcast_to(np.arange(5), (1, 3, 5)).copy(); temperature = .2
    if fault == 'duplicate_ids': ids[..., 4] = 0
    if fault == 'negative_ids': ids[..., 4] = -1
    if fault == 'out_of_bank': ids[..., 4] = 6
    if fault == 'nonfinite_query': query[0, 0, 0] = np.nan
    if fault == 'zero_temperature': temperature = 0
    with pytest.raises(ValueError): score_with_ids(query, bank, ids, temperature)


def fixture_records():
    diagnostic = {'failed_targets': [{'sequence': '03', 'target_frame': 269,
                                      'reference_feature_raw': .4, 'candidate_feature_raw': .401,
                                      'reference_phase': .75, 'candidate_phase': .75}]}
    pins = {'original': 'hash'}
    records = {}
    for implementation, prefix in [('full', 'reference'), ('reuse', 'candidate')]:
        original = diagnostic['failed_targets'][0]
        raw = original[prefix + '_feature_raw']
        row = {'sequence': '03', 'target_frame': 269, 'feature_raw': raw, 'original_feature_raw': raw,
               'phase': .75, 'original_phase': .75, 'feature_raw_difference_from_original': 0.,
               'original_raw_within_fixed_gate': True, 'original_phase_within_fixed_gate': True}
        records[implementation] = {'status': 'complete_untimed_gpu_capture', 'implementation': implementation,
                                   'condition': deepcopy(CONDITION), 'original_parity_status': 'failed',
                                   'warmup_forward_calls': 20, 'cuda_matmul_tf32': False, 'cudnn_tf32': False,
                                   'torch': 'same', 'gpu': 'same', 'source_sha256': pins.copy(), 'captures': [row]}
    return records, diagnostic, pins


def test_complete_same_source_target_pair_accepted():
    records, diagnostic, pins = fixture_records()
    inventory, merged = verify_capture_pair(records, diagnostic, pins)
    assert set(inventory['full']) == {('03', 269)}
    assert merged == pins


@pytest.mark.parametrize('fault', ['missing_capture', 'duplicate_target', 'missing_pin', 'conflicting_pin',
                                  'different_gpu', 'tf32', 'raw_mismatch', 'phase_mismatch', 'wrong_original'])
def test_untrustworthy_capture_pair_rejected(fault):
    records, diagnostic, pins = fixture_records(); candidate = records['reuse']
    if fault == 'missing_capture': candidate['captures'] = []
    if fault == 'duplicate_target': candidate['captures'] *= 2
    if fault == 'missing_pin': candidate['source_sha256'] = {}
    if fault == 'conflicting_pin': candidate['source_sha256']['original'] = 'altered'
    if fault == 'different_gpu': candidate['gpu'] = 'different'
    if fault == 'tf32': candidate['cuda_matmul_tf32'] = True
    if fault == 'raw_mismatch':
        candidate['captures'][0]['feature_raw'] = .6
        candidate['captures'][0]['feature_raw_difference_from_original'] = .6 - .401
    if fault == 'phase_mismatch': candidate['captures'][0]['phase'] = .5
    if fault == 'wrong_original': candidate['captures'][0]['original_feature_raw'] = .5
    with pytest.raises(ValueError): verify_capture_pair(records, diagnostic, pins)


def test_changed_input_hash_rejected(tmp_path):
    import hashlib
    path = tmp_path / 'source'; path.write_bytes(b'original')
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_pins(pins)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed source'): validate_pins(pins)


@pytest.fixture(scope='module')
def captured_schema():
    import torch
    from types import SimpleNamespace
    from ipad_jepa.torch_memory import TorchMemory
    from probe_runtime_search import search_capture
    rng = np.random.default_rng(91)
    prototypes = rng.normal(size=(16, 128, 256)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=-1, keepdims=True)
    scorer = TorchMemory(SimpleNamespace(bins=16, dimensions=256, temperature=.5,
                         pca=SimpleNamespace(mean_=np.zeros(1024, dtype=np.float32),
                                             components_=np.eye(256, 1024, dtype=np.float32)),
                         prototypes=prototypes))
    local = torch.tensor(rng.normal(size=(1, 576, 1024)).astype(np.float32))
    with torch.inference_mode():
        result = search_capture(scorer, local, torch.tensor([.75]))
    arrays = {k: v.numpy() for k, v in result.items()}
    arrays.update(local_features=local.numpy(), global_features=local.mean(1).numpy())
    return arrays, prototypes


def test_schema_and_full_candidate_FP64_search_independently_replayed(captured_schema):
    arrays, prototypes = captured_schema
    assert check_arrays(arrays, prototypes, .5, .75) < 1e-12


@pytest.mark.parametrize('fault', ['missing_array', 'wrong_bins', 'nonfinite', 'wrong_distance', 'nonnearest_ids'])
def test_altered_capture_arrays_rejected(captured_schema, fault):
    original, prototypes = captured_schema; arrays = {k: v.copy() for k, v in original.items()}
    if fault == 'missing_array': arrays.pop('query')
    if fault == 'wrong_bins': arrays['phase_bins'][0, 0] = 0
    if fault == 'nonfinite': arrays['query'][0, 0, 0] = np.nan
    if fault == 'wrong_distance': arrays['stable_distances'][0, 0, 0] += .001
    if fault == 'nonnearest_ids':
        selected = set(arrays['stable_ids'][0, 0])
        arrays['stable_ids'][0, 0, 0] = next(i for i in range(11*128, 12*128) if i not in selected)
    with pytest.raises(ValueError): check_arrays(arrays, prototypes, .5, .75)
