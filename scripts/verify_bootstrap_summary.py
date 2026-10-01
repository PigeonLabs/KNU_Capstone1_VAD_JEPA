"""Compare optimized summaries with saved, hash-bound sklearn reference outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_value(reference, actual, context, errors):
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or reference.keys() != actual.keys():
            raise ValueError(f'{context}: field inventory differs')
        for key in reference:
            compare_value(reference[key], actual[key], f'{context}.{key}', errors)
    elif isinstance(reference, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError(f'{context}: non-finite or non-numeric value')
        difference = abs(reference-actual)
        if difference > 1e-12:
            raise ValueError(f'{context}: difference {difference} exceeds 1e-12')
        errors.append(difference)
    elif reference != actual or type(reference) is not type(actual):
        raise ValueError(f'{context}: value differs')


def compare_rows(reference, actual, keys, context, errors):
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(row[field] for field in keys)
            if key in result:
                raise ValueError(f'{context}: duplicate condition {key}')
            result[key] = row
        return result
    before, after = index(reference), index(actual)
    if not before.keys() <= after.keys():
        raise ValueError(f'{context}: reference conditions are missing')
    for key, row in before.items():
        compare_value(row, after[key], f'{context}.{key}', errors)
    return {'reference_rows_compared': len(before), 'current_rows': len(after),
            'additional_rows': len(after)-len(before)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/stage02'))
    args = parser.parse_args()
    evidence = args.root/'bootstrap_validation'
    record = json.loads((evidence/'validation.json').read_text())
    for name, expected in record['reference_sources'].items():
        if digest(evidence/name) != expected:
            raise ValueError(f'Reference checksum differs: {name}')
    original = subprocess.run(
        ['git', 'show', record['reference_code_commit']+':scripts/summarize_experiments.py'],
        check=True, capture_output=True).stdout
    if hashlib.sha256(original).hexdigest() != record['reference_code_sha256']:
        raise ValueError('Reference implementation checksum differs')
    module = Path('src/ipad_jepa/bootstrap_metrics.py')
    if digest(module) != record['optimized_metric_sha256']:
        raise ValueError('Optimized metric differs from the tested implementation')
    real = record['real_draw_check']
    if real['status'] != 'passed':
        raise ValueError('Real-data draw comparison has not passed')
    for name, expected in real['sources'].items():
        source = args.root/real['relative_root']/name
        if not source.resolve().is_relative_to(args.root.resolve()) or digest(source) != expected:
            raise ValueError(f'Real-data comparison source differs: {name}')
    if 'native_single_seed_summary_check' in record:
        native = record['native_single_seed_summary_check']
        original = subprocess.run(['git', 'show', native['reference_commit']+':'+native['reference_path']],
                                  check=True, capture_output=True).stdout
        if hashlib.sha256(original).hexdigest() != native['reference_sha256']:
            raise ValueError('Native reference summary checksum differs')
        current = Path(native['reference_path']).read_bytes()
        if hashlib.sha256(current).hexdigest() != native['current_sha256']:
            raise ValueError('Native current summary checksum differs')
        before, after, native_errors = json.loads(original), json.loads(current), []
        for field in ['status', 'device', 'scope', 'seed', 'seeds', 'test_videos', 'bootstrap',
                      'bootstrap_rejected', 'sources', 'limitation', 'variants', 'B1_minus_B0']:
            compare_value(before[field], after[field], 'native.'+field, native_errors)
        native.update(status='passed', float_values_compared=len(native_errors),
                      maximum_absolute_difference=max(native_errors, default=0))
    errors, counts, current_sources = [], {}, {}
    for name, device in [('device_summary.json', True), ('macro_summary.json', False)]:
        before = json.loads((evidence/('reference_'+name)).read_text())
        after = json.loads((args.root/name).read_text())
        for field in ['scope', 'bootstrap']:
            compare_value(before[field], after[field], f'{name}.{field}', errors)
        if device:
            compare_value(before['macro_summary'], after['macro_summary'], name+'.macro_summary', errors)
        identity = ['backbone', 'mode']+(['device'] if device else [])
        for field, dimension in [('results', 'variant'), ('paired_deltas', 'comparison')]:
            counts[name+'/'+field] = compare_rows(before[field], after[field], identity+[dimension],
                                                 name+'/'+field, errors)
        current_sources[name] = digest(args.root/name)
    record.update(status='reference_subset_summary_comparison_passed',
                  summary_comparison={'absolute_tolerance': 1e-12,
                      'float_values_compared': len(errors), 'maximum_absolute_difference': max(errors, default=0),
                      'inventories': counts,
                      'scope': 'All saved reference rows, means, percentile CIs, paired differences, seed/video/frame counts and rejected-draw counts. Additional completed conditions may be present; their old outputs are unavailable.'},
                  optimized_sources={**current_sources,
                      'scripts/summarize_experiments.py': digest(Path('scripts/summarize_experiments.py')),
                      'src/ipad_jepa/bootstrap_metrics.py': digest(module),
                      'scripts/verify_bootstrap_summary.py': digest(Path(__file__))})
    (evidence/'validation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record['summary_comparison'], indent=2))


if __name__ == '__main__':
    main()
