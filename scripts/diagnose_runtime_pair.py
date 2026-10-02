"""Replay a failed FP32 runtime pair without changing gates or claiming encoder equivalence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ipad_jepa.cache_retirement import atomic_record
from verify_runtime_parity import compare, digest as parity_digest, GATES


def digest(path):
    return parity_digest(Path(path))


def phase_bins(phases, bins):
    # benchmark_runtime constructs a default FP32 phase tensor before TorchMemory.
    phases = np.asarray(phases, dtype=np.float32)
    if bins < 3 or not np.isfinite(phases).all() or np.any((phases < 0) | (phases >= 1)):
        raise ValueError('Valid FP32 runtime phases and at least three bins required')
    return (phases * np.float32(bins)).astype(np.int64)


def normalized_delta_components(reference, candidate, scale):
    reference, candidate, scale = [np.asarray(value, dtype=float) for value in (reference, candidate, scale)]
    if (reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 2
            or scale.shape != (2,) or not np.isfinite(np.r_[reference.ravel(), candidate.ravel(), scale]).all()
            or np.any(scale <= 0)):
        raise ValueError('Finite paired feature/time raw scores and positive calibration scales required')
    return .5 * (candidate - reference) / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--normal-fit', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--measurement-sources', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = args.root / 'full', args.root / 'reuse'
    published = json.loads((args.root / 'parity.json').read_text())
    replay = compare(reference, candidate)
    if replay != published or replay['status'] != 'failed' or replay['condition']['precision'] != 'fp32':
        raise ValueError('Exact published failed FP32 pair required')
    meta = json.loads((reference / 'runtime.json').read_text())
    pinned = json.loads(args.measurement_sources.read_text())
    if (pinned['status'] != 'passed_actual_live_runtime_controller_source_and_completed_output_check'
            or digest('scripts/benchmark_runtime.py') != meta['runtime_code_sha256']
            or digest('src/ipad_jepa/runtime_state.py') != meta['score_state_sha256']
            or pinned['source_sha256'].get('src/ipad_jepa/torch_memory.py') != digest('src/ipad_jepa/torch_memory.py')
            or any(digest(path) != expected for path, expected in pinned['source_sha256'].items())):
        raise ValueError('Current scorer differs from actual measurement source pins')
    for folder in (reference, candidate):
        matches = [run for run in pinned['completed_runs'].values() if str(folder / 'runtime.json') in run['sources_sha256']]
        if len(matches) != 1 or matches[0]['status'] != 'completed_child_exit_zero':
            raise ValueError('Exact completed measured pair missing from source snapshot')
        if any(digest(path) != expected for path, expected in matches[0]['sources_sha256'].items()):
            raise ValueError('Original completed measured output changed')
    if digest(args.normal_fit) != meta['normal_fit_sha256'] or digest(args.bank) != meta['memory_sha256']:
        raise ValueError('Paired normal calibration or prototype bank changed')
    normal = json.loads(args.normal_fit.read_text())
    bins = normal['bins']
    with np.load(args.bank, allow_pickle=False) as bank:
        if bank['prototypes'].shape[0] != bins:
            raise ValueError('Actual bank bin inventory differs')
    scale = normal['calibration']['P3']['mad_scale']
    paths = [args.root / 'parity.json', args.normal_fit, args.measurement_sources,
             Path(__file__).resolve().relative_to(Path.cwd().resolve()),
             Path('scripts/verify_runtime_parity.py'), Path('scripts/benchmark_runtime.py'),
             Path('src/ipad_jepa/torch_memory.py'), Path('src/ipad_jepa/runtime_state.py')]
    sources = {str(path): digest(path) for path in paths}
    videos, failures = [], []
    all_bins_changed = alarm_mismatches = eligible_total = 0
    max_delta_error = 0.
    for sequence in [row['sequence'] for row in replay['sequences']]:
        paired = []
        for folder in (reference, candidate):
            path = folder / (sequence + '.csv')
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            paired.append([row for row in rows if row['inference_valid'] == '1'])
            sources[str(path)] = digest(path)
        a, b = paired
        left, right = [np.array([[float(row['feature_raw']), float(row['time_raw'])] for row in rows]) for rows in paired]
        phi_a, phi_b = [phase_bins([float(row['phase']) for row in rows], bins) for rows in paired]
        contributions = normalized_delta_components(left, right, scale)
        score_delta = np.array([float(y['score']) - float(x['score']) for x, y in zip(a, b)])
        error = float(np.max(np.abs(score_delta - contributions.sum(1))))
        if error > 1e-9:
            raise ValueError('P3 difference not reproduced by fixed normal feature/time scales')
        raw_failed = ~np.isclose(left[:, 0], right[:, 0], rtol=GATES['raw_rtol'], atol=GATES['raw_atol'])
        changed = int(np.sum(phi_a != phi_b))
        alarms = sum(x['alarm'] != y['alarm'] for x, y in zip(a, b))
        eligible_total += len(a); all_bins_changed += changed; alarm_mismatches += alarms
        max_delta_error = max(max_delta_error, error)
        videos.append({'sequence': sequence, 'eligible_targets': len(a), 'phase_bank_bin_changes': changed,
                       'raw_feature_gate_failed_targets': int(raw_failed.sum()), 'alarm_mismatches': alarms,
                       'max_score_delta_replay_error': error})
        for index in np.flatnonzero(raw_failed):
            index = int(index)
            tolerance = GATES['raw_atol'] + GATES['raw_rtol'] * abs(right[index, 0])
            failures.append({'sequence': sequence, 'target_frame': int(a[index]['target_frame']),
                             'reference_phase': float(a[index]['phase']), 'candidate_phase': float(b[index]['phase']),
                             'reference_bank_bin': int(phi_a[index]), 'candidate_bank_bin': int(phi_b[index]),
                             'reference_feature_raw': float(left[index, 0]), 'candidate_feature_raw': float(right[index, 0]),
                             'feature_raw_difference': float(right[index, 0] - left[index, 0]),
                             'raw_gate_tolerance_at_candidate': float(tolerance),
                             'raw_gate_exceedance_ratio': float(abs(right[index, 0] - left[index, 0]) / tolerance),
                             'score_difference': float(score_delta[index]),
                             'feature_contribution_to_score_difference': float(contributions[index, 0]),
                             'time_contribution_to_score_difference': float(contributions[index, 1])})
    if not failures:
        raise ValueError('No raw feature gate failures to diagnose')
    for folder in (reference, candidate):
        path = folder / 'trace_audit.json'
        proof = json.loads(path.read_text())
        if proof['status'] != 'passed_actual_runtime_trace_replay':
            raise ValueError('Prior actual trace audit required')
        for name, expected in proof['sources_sha256'].items():
            if digest(name) != expected:
                raise ValueError('Independently audited trace source changed')
            sources[name] = expected
        if proof['auditor_sha256'] != digest('scripts/audit_runtime.py'):
            raise ValueError('Trace auditor changed')
        sources[str(path)] = digest(path)
    fig, axis = plt.subplots(figsize=(9, 4.9), layout='constrained')
    bars = axis.bar(range(len(failures)), [row['raw_gate_exceedance_ratio'] for row in failures],
                    color='#c16c40', width=.55)
    axis.bar_label(bars, fmt='%.1f×', padding=5)
    axis.axhline(1, color='#496577', linestyle='--', linewidth=1, label='Fixed gate limit (1×)')
    axis.set(xticks=range(len(failures)), xticklabels=[f"Video {row['sequence']} / t={row['target_frame']}" for row in failures],
             ylabel='Absolute raw error / fixed tolerance', ylim=(0, max(row['raw_gate_exceedance_ratio'] for row in failures) * 1.25))
    axis.spines[['top', 'right']].set_visible(False); axis.legend(frameon=False)
    axis.grid(axis='y', alpha=.18); axis.set_axisbelow(True)
    fig.suptitle('DINOv3-L / online / R01 / seed 0 / FP32\nFull vs feature reuse: failed raw-score targets', fontsize=13)
    fig.supxlabel(f'{eligible_total:,} eligible targets / {len(videos)} videos; unchanged gates.\n'
                  f'{all_bins_changed} FP32 phase-bank bin changes; {alarm_mismatches} alarm mismatches. Encoder/query/neighbor cause remains unverified.', fontsize=9)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(args.figure.with_suffix('.' + extension), dpi=180)
    plt.close(fig)
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('Diagnostic inputs changed')
    result = {'status': 'replayed_failed_pair_and_localized_trace_divergence', 'parity_status': 'failed',
              'condition': replay['condition'], 'gates': GATES, 'videos_checked': len(videos),
              'eligible_targets': eligible_total, 'phase_bank_bins': bins, 'phase_bank_bin_changes': all_bins_changed,
              'alarm_mismatches': alarm_mismatches, 'raw_feature_failed_targets': len(failures),
              'P3_mad_scales': scale, 'max_score_delta_replay_error': max_delta_error,
              'video_checks': videos, 'failed_targets': failures, 'source_sha256': sources,
              'actual_bank_sha256': digest(args.bank),
              'figures': {str(args.figure.with_suffix('.' + ext)): digest(args.figure.with_suffix('.' + ext)) for ext in ('png', 'svg')},
              'finding': 'Phase gating uses FP32 bin IDs only. No bin change was observed, and P3 score deltas reproduce feature/time contributions under fixed calibration. Raw feature differences require encoder/query/search investigation.',
              'limits': 'Trace-only replay plus current bank/calibration hashes and actual scorer source review; no encoder features, transformed queries, neighbor IDs/distances or ties captured. No GPU rerun, cause attribution, relaxed gate, streaming equivalence or new runtime measurement.'}
    atomic_record(args.root / 'failure_diagnostics.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_sha256', 'video_checks', 'figures']}, indent=2))


if __name__ == '__main__':
    main()
