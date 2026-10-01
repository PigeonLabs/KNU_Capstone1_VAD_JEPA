import importlib.util
import csv
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


def test_runtime_parity_binds_selected_adapter_memory_and_input_sources(tmp_path):
    mod = module(); folders = [tmp_path/'full', tmp_path/'buffer']
    common = {'status': 'complete_measured_replay', 'backbone': 'dinov3-l', 'mode': 'online',
        'device': 'R01', 'seed': 0, 'variant': 'P3', 'precision': 'bf16', 'arrival_fps': 30.,
        'normal_fit_sha256': 'normal-fit', 'phase_head_sha256': 'selected-head',
        'memory_sha256': 'rebuilt-memory', 'weights_sha256': 'base',
        'runtime_code_sha256': 'runtime', 'score_state_sha256': 'state',
        'cuda_matmul_tf32': False, 'cudnn_tf32': False, 'manifest_sha256': 'manifest',
        'adaptation': 'lora', 'selected_adapter_sha256': 'selected-adapter',
        'lora_training_sha256': 'completed-training', 'teacher_weight': 1.,
        'runtime_adaptation_sha256': 'adapter-loader', 'all_test_videos': False,
        'alarm_warmup_target': 19, 'lookahead_frames': 0, 'test_bank_updates': False,
        'sequences': [{'sequence': '03', 'input_frames': 40, 'eligible_targets': 21,
                       'frames_content_sha256': 'frames', 'label_sha256': 'labels'}]}
    trace = [{'arrival_frame': str(i), 'target_frame': str(i) if i >= 15 else '',
        'inference_valid': str(int(i >= 19)), 'label': '-1' if i == 25 else '0',
        'shared_metric_valid': str(int(19 <= i <= 32 and i != 25)),
        'phase': '.2', 'feature_raw': '2', 'time_raw': '.01', 'score': '3', 'alarm': '0'}
        for i in range(40)]
    for folder, implementation in zip(folders, ['full', 'buffer']):
        folder.mkdir()
        (folder/'runtime.json').write_text(json.dumps({**common, 'implementation': implementation}))
        with (folder/'03.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader(); writer.writerows(trace)
    assert mod.compare(*folders)['status'] == 'passed'
    assert mod.compare(*folders)['condition']['selected_adapter_sha256'] == 'selected-adapter'
    for field in ['selected_adapter_sha256', 'memory_sha256', 'runtime_adaptation_sha256']:
        changed = {**common, 'implementation': 'buffer', field: 'different'}
        (folders[1]/'runtime.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='provenance differ'): mod.compare(*folders)
    changed = {**common, 'implementation': 'buffer', 'sequences': [
        {**common['sequences'][0], 'frames_content_sha256': 'other-frames'}]}
    (folders[1]/'runtime.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='test sources differ'): mod.compare(*folders)
