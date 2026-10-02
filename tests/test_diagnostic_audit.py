"""Diagnostic replay rejects leakage/tampering; bootstrap clusters by original video."""
import numpy as np
import pytest
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_diagnostics import source_inventory

from ipad_jepa.diagnostic_audit import (FIELDS, audit_trace, class_f1, clustered_statistics,
                                      normalized_confusion, equal_device_confusion)
from ipad_jepa.diagnostics import scenarios
from ipad_jepa.temporal import temporal_score


def fixture(mode):
    n = 96
    intervention = next(s for s in scenarios(dict(device='R01', sequence='02',
        partition='training', split='diagnostic', frames=n)) if s.name == 'stall_8__occlusion_1')
    ids = np.arange(15, n) if mode == 'online' else np.arange(8, n-7)
    phase = (ids/100) % 1
    dense = np.full(n, np.nan); dense[ids] = phase
    raw = np.stack([np.where(intervention.labels()[ids]&1, 4., .5),
                    temporal_score(dense, 100, 5)[ids]], axis=1)
    params = dict(median=[1., 0.], mad_scale=[2., .1], component_thresholds=[2., .2])
    predicted = np.zeros(len(ids), int); finite = np.isfinite(raw).all(1)
    predicted[finite] = (raw[finite, 0] > 2).astype(int) + 2*(raw[finite, 1] > .2)
    values = zip(ids, ((ids>=19)&(ids<=n-8)).astype(int), phase, raw[:,0], raw[:,1],
                 ((raw-params['median'])/params['mad_scale']).mean(1), predicted,
                 intervention.labels()[ids])
    rows = [dict(zip(FIELDS, map(str, row))) for row in values]
    return rows, n, intervention.metadata(), params


@pytest.mark.parametrize('mode', ['online', 'offline'])
def test_replay_checks_complete_targets_and_framewise_partial_overlap(mode):
    rows, n, recipe, params = fixture(mode)
    matrix = audit_trace(rows, n, mode, recipe, 100, params)
    assert matrix.sum() == n-26
    assert matrix.sum(1)[3] == 8 and matrix.sum(1)[1] == 24
    with pytest.raises(AssertionError):
        audit_trace(rows[1:], n, mode, recipe, 100, params)


@pytest.mark.parametrize('field', ['valid','phase','time_raw','score','evidence_type','intervention_type'])
def test_trace_tampering_is_rejected(field):
    rows, n, recipe, params = fixture('online')
    rows[20][field] = str(float(rows[20][field])+1)
    with pytest.raises((AssertionError, ValueError)):
        audit_trace(rows, n, 'online', recipe, 100, params)


def test_original_video_clusters_share_draws_across_seeds():
    matrices = np.zeros((3,2,4,4), dtype=np.int64)
    matrices[:,0] = np.eye(4,dtype=np.int64)*10
    matrices[:,1,0,1] = 20
    point, classes, boot, class_boot = clustered_statistics(matrices, 1000, 42)
    np.testing.assert_allclose(classes, class_f1(matrices.sum(1)).mean(0))
    assert point == classes.mean()
    # Two ORIGINAL videos produce just three possible pooled mixtures, although
    # each video can contain arbitrarily many correlated intervention scenarios.
    assert len(np.unique(boot)) == 3
    np.testing.assert_allclose(boot, class_boot.mean(1))
    again = clustered_statistics(matrices, 1000, 42)
    np.testing.assert_array_equal(boot, again[2])


def test_four_classes_remain_in_macro_with_zero_denominator():
    matrix = np.zeros((4,4), dtype=np.int64); matrix[0,0]=10
    np.testing.assert_array_equal(class_f1(matrix), [1.,0.,0.,0.])
    with pytest.raises(ValueError):
        class_f1(matrix.astype(float))


def test_macro_confusion_normalizes_each_device_before_equal_weight_average():
    devices = np.zeros((4,4,4),dtype=float)
    devices[0] = np.eye(4)*10000
    for i in (1,2,3):
        devices[i] = np.roll(np.eye(4),1,axis=1)
    actual = equal_device_confusion(devices)
    np.testing.assert_allclose(np.diag(actual), [.25]*4)
    np.testing.assert_allclose(actual.sum(1), [1.]*4)
    assert not np.allclose(actual,normalized_confusion(devices.sum(0)))


def test_confusion_display_keeps_empty_truth_rows_at_zero():
    matrix=np.zeros((4,4));matrix[0,1]=10
    np.testing.assert_array_equal(normalized_confusion(matrix)[0],[0.,1.,0.,0.])
    assert np.isfinite(normalized_confusion(matrix)).all()
    with pytest.raises(ValueError):equal_device_confusion(np.zeros((3,4,4)))


@pytest.mark.parametrize('mutation', ['missing', 'changed'])
def test_source_inventory_uses_checkout_paths_and_rejects_missing_or_changed_pins(tmp_path, monkeypatch, mutation):
    names = ('diagnostic_evaluation','diagnostics','backbones','features','audit',
             'temporal','scoring','torch_memory','memory')
    required = [Path(f'src/ipad_jepa/{name}.py') for name in names]
    manifest = Path('results/stage00/manifest.json')
    required += [manifest, Path('configs/experiment_matrix.yaml')]
    pins = {}
    for relative in required:
        actual = tmp_path/relative
        actual.parent.mkdir(parents=True, exist_ok=True)
        actual.write_bytes(b'original source\n')
        # Absolute recorded source paths refer to a DIFFERENT producer checkout.
        key = '/previous/checkout/'+str(relative) if relative.suffix=='.py' else str(relative)
        pins[key] = hashlib.sha256(actual.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    assert set(source_inventory({'source_sha256':pins}, manifest)) == {str(p) for p in required}
    if mutation == 'missing':
        del pins[str(manifest)]
    else:
        (tmp_path/required[0]).write_bytes(b'changed protocol\n')
    with pytest.raises(ValueError):
        source_inventory({'source_sha256':pins}, manifest)
