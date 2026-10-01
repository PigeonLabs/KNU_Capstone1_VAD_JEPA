"""Compare completed runtime scores/alarms before claiming streaming equivalence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


GATES = {'phase_circular_max_abs': 1e-5, 'raw_rtol': 1e-5, 'raw_atol': 1e-7,
         'score_rtol': 1e-5, 'score_atol': 1e-4, 'alarm_mismatches': 0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_rows(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError('Different or empty arrival inventory')
    fields = ['arrival_frame', 'target_frame', 'inference_valid', 'label', 'shared_metric_valid']
    if any(any(a[key] != b[key] for key in fields) for a, b in zip(reference, candidate)):
        raise ValueError('Arrival/target/eligibility/label inventories differ')
    if [int(row['arrival_frame']) for row in reference] != list(range(len(reference))):
        raise ValueError('Dropped or reordered input frames')
    selected = [i for i, row in enumerate(reference) if row['inference_valid'] == '1']
    if not selected:
        raise ValueError('No eligible runtime targets')
    stats = {}; passed = True
    for field in ['phase', 'feature_raw', 'time_raw', 'score']:
        a = np.array([float(reference[i][field]) for i in selected])
        b = np.array([float(candidate[i][field]) for i in selected])
        if not np.isfinite(np.r_[a, b]).all():
            raise ValueError('Nonfinite eligible scores')
        delta = (b-a+.5) % 1-.5 if field == 'phase' else b-a
        stats[field] = {'max_abs_difference': float(np.max(np.abs(delta))),
                        'relative_l2_difference': float(np.linalg.norm(delta)/max(np.linalg.norm(a), 1e-12))}
        if field == 'phase':
            keep = bool(np.max(np.abs(delta)) <= GATES['phase_circular_max_abs'])
        else:
            keep = bool(np.allclose(a, b, rtol=GATES['score_rtol' if field == 'score' else 'raw_rtol'],
                                    atol=GATES['score_atol' if field == 'score' else 'raw_atol']))
        stats[field]['passed'] = keep; passed &= keep
    if any(row['alarm'] not in {'0', '1'} for rows in [reference, candidate] for row in rows):
        raise ValueError('Invalid alarm flags')
    mismatches = sum(reference[i]['alarm'] != candidate[i]['alarm'] for i in selected)
    return {'passed': bool(passed and mismatches == 0), 'input_frames': len(reference),
            'eligible_targets': len(selected), 'alarm_mismatches': mismatches, 'components': stats}


def compare(reference, candidate):
    a = json.loads((reference/'runtime.json').read_text())
    b = json.loads((candidate/'runtime.json').read_text())
    if any(meta.get('status') != 'complete_measured_replay' for meta in [a, b]):
        raise ValueError('Completed measured runtime results required')
    fields = ['backbone', 'mode', 'device', 'seed', 'variant', 'precision', 'arrival_fps',
              'normal_fit_sha256', 'phase_head_sha256', 'memory_sha256', 'weights_sha256',
              'runtime_code_sha256', 'score_state_sha256', 'cuda_matmul_tf32', 'cudnn_tf32',
              'manifest_sha256',
              'all_test_videos', 'alarm_warmup_target', 'lookahead_frames', 'test_bank_updates']
    if any(a.get(key) != b.get(key) or key not in a for key in fields):
        raise ValueError('Paired runtime conditions/provenance differ')
    if a['implementation'] != 'full' or b['implementation'] not in {'buffer', 'reuse'}:
        raise ValueError('Reference must be full; candidate must be buffer or reuse')
    if b['implementation'] == 'reuse' and (b['backbone'], b['mode'], b['precision']) != ('dinov3-l', 'online', 'fp32'):
        raise ValueError('Only the separately reported DINO online FP32 reuse is eligible')
    sequences = [row['sequence'] for row in a['sequences']]
    if not sequences or len(set(sequences)) != len(sequences) or sequences != [row['sequence'] for row in b['sequences']]:
        raise ValueError('Runtime video inventories differ')
    results = []; sources = []
    for sequence in sequences:
        declared_pair = [next(row for row in meta['sequences'] if row['sequence'] == sequence) for meta in [a, b]]
        first, second = declared_pair
        if any(first.get(key) != second.get(key) or key not in first
               for key in ['frames_content_sha256', 'label_sha256']):
            raise ValueError('Runtime test sources differ or are unbound')
        paths = [folder/f'{sequence}.csv' for folder in [reference, candidate]]
        rows = []
        for path in paths:
            with path.open() as stream:
                rows.append(list(csv.DictReader(stream)))
        result = compare_rows(*rows)
        for declared in declared_pair:
            if result['input_frames'] != declared['input_frames'] or result['eligible_targets'] != declared['eligible_targets']:
                raise ValueError('Trace inventory differs from completed metadata')
        results.append({'sequence': sequence, **result})
        sources.append({'sequence': sequence, 'reference_sha256': digest(paths[0]), 'candidate_sha256': digest(paths[1])})
    return {'status': 'passed' if all(row['passed'] for row in results) else 'failed',
            'scope': 'All emitted eligible targets, including online tail and unknown GT; no GT exclusion or test tuning',
            'condition': {key: a[key] for key in ['backbone', 'mode', 'device', 'seed', 'precision', 'all_test_videos']},
            'reference_implementation': a['implementation'], 'candidate_implementation': b['implementation'],
            'gates': GATES, 'sequences': results, 'trace_sources': sources,
            'reference_runtime_sha256': digest(reference/'runtime.json'),
            'candidate_runtime_sha256': digest(candidate/'runtime.json'),
            'verifier_sha256': digest(Path(__file__)),
            'note': 'A pass establishes this measured pair only; it does not establish equivalence to batch4 accuracy caches or another precision.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.out)
    print(json.dumps({'status': result['status'], 'videos_checked': len(result['sequences'])}))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
