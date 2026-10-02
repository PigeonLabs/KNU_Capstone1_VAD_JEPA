"""Reporting gates retain actual failed equivalence and reject mismatched provenance."""
import json
from pathlib import Path

import pytest
import plot_runtime_pairs as reporting


def measured_pairs(tmp_path, monkeypatch):
    root = tmp_path / 'condition'
    reports, proofs, comparisons = {}, {}, {}
    for precision, candidate in [('bf16', 'buffer'), ('fp32', 'reuse')]:
        for implementation in ['full', candidate]:
            folder = root / precision / implementation
            folder.mkdir(parents=True)
            report = {key: 'same' for key in ['backbone', 'mode', 'device', 'adaptation', 'variant',
                      'normal_fit_sha256', 'phase_head_sha256', 'memory_sha256', 'weights_sha256',
                      'manifest_sha256', 'runtime_code_sha256', 'score_state_sha256',
                      'selected_adapter_sha256']}
            report.update(seed=0, arrival_fps=30., teacher_weight=1., all_test_videos=True,
                          precision=precision, implementation=implementation)
            proof = {'status': 'passed_actual_runtime_trace_replay', 'test_videos': 15,
                     'sources_sha256': {}}
            (folder / 'runtime.json').write_text(json.dumps(report))
            (folder / 'trace_audit.json').write_text(json.dumps(proof))
            reports[(precision, implementation)] = report
            proofs[folder] = proof
        result = {'status': 'passed' if precision == 'bf16' else 'failed',
                  'sequences': [{'sequence': '03', 'passed': precision == 'bf16', 'alarm_mismatches': 0}]}
        (root / precision / 'parity.json').write_text(json.dumps(result))
        comparisons[root / precision] = result
    monkeypatch.setattr(reporting, 'audit', lambda folder, *_: proofs[folder])
    monkeypatch.setattr(reporting, 'compare', lambda reference, candidate: comparisons[reference.parent])
    return root, reports, proofs


def test_failed_scores_remain_failed_despite_matching_alarms(tmp_path, monkeypatch):
    root, _, _ = measured_pairs(tmp_path, monkeypatch)
    _, _, parity, _ = reporting.verified_pairs(root, Path('manifest'), Path('data'))
    assert parity['fp32']['status'] == 'failed'
    assert parity['fp32']['sequences'][0]['alarm_mismatches'] == 0


def test_report_refuses_relabelled_failed_parity(tmp_path, monkeypatch):
    root, _, _ = measured_pairs(tmp_path, monkeypatch)
    path = root / 'fp32/parity.json'
    result = json.loads(path.read_text())
    result['status'] = 'passed'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='preserve the actual failed gate'):
        reporting.verified_pairs(root, Path('manifest'), Path('data'))


def test_report_refuses_cross_precision_checkpoint_mix(tmp_path, monkeypatch):
    root, reports, _ = measured_pairs(tmp_path, monkeypatch)
    for implementation in ['full', 'reuse']:
        report = reports[('fp32', implementation)]
        report['phase_head_sha256'] = 'different-selected-head'
        (root / 'fp32' / implementation / 'runtime.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='different selected conditions'):
        reporting.verified_pairs(root, Path('manifest'), Path('data'))
