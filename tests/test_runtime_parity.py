import importlib.util
import json
from pathlib import Path
import pytest


def module():
    spec = importlib.util.spec_from_file_location('runtime_parity', Path(__file__).parents[1]/'scripts/verify_runtime_parity.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def rows():
    return [{'arrival_frame': str(i), 'target_frame': str(i), 'inference_valid': '1',
             'label': '-1' if i == 2 else '0', 'shared_metric_valid': '0' if i == 2 else '1',
             'phase': '.999999', 'feature_raw': '2', 'time_raw': '.01', 'score': '3', 'alarm': '0'}
            for i in range(4)]


def test_runtime_parity_checks_unknown_gt_and_online_tail_alarms():
    mod = module(); a = rows(); b = rows()
    assert mod.compare_rows(a, b)['passed']
    b[2]['alarm'] = '1'
    result = mod.compare_rows(a, b)
    assert not result['passed'] and result['alarm_mismatches'] == 1
    b = rows(); b[-1]['score'] = '4'
    assert not mod.compare_rows(a, b)['passed']


def test_runtime_parity_uses_circular_phase_but_fixed_score_tolerance():
    mod = module(); a = rows(); b = rows(); b[0]['phase'] = '.000001'
    assert mod.compare_rows(a, b)['passed']
    b[0]['score'] = '3.001'
    assert not mod.compare_rows(a, b)['passed']


def test_runtime_parity_refuses_missing_or_remasked_targets():
    mod = module(); a = rows(); b = rows()
    with pytest.raises(ValueError, match='arrival inventory'): mod.compare_rows(a, b[:-1])
    b[-1]['inference_valid'] = '0'
    with pytest.raises(ValueError, match='inventories differ'): mod.compare_rows(a, b)
    b = rows(); a[0]['score'] = 'nan'
    with pytest.raises(ValueError, match='Nonfinite'): mod.compare_rows(a, b)


def test_runtime_parity_refuses_partial_or_different_conditions(tmp_path):
    mod = module(); a = tmp_path/'full'; b = tmp_path/'reuse'
    a.mkdir(); b.mkdir()
    for folder in [a, b]:
        (folder/'runtime.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='Completed'): mod.compare(a, b)
    for folder in [a, b]:
        (folder/'runtime.json').write_text(json.dumps({'status': 'complete_measured_replay', 'mode': 'online'}))
    with pytest.raises(ValueError, match='provenance differ'): mod.compare(a, b)
