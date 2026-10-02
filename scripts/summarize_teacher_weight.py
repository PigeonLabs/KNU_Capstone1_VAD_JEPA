"""Independently replay actual teacher 0/1 LoRA pairs; never use frozen as T1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ipad_jepa.lora_audit import ready_group
from ipad_jepa.teacher_weight_audit import check_pair_controls
from summarize_lora_matrix import audit_lora, MODELS, MODES, DEVICES, SEEDS, VARIANTS
from summarize_experiments import load_run, check_pair, statistic, bootstrap_draws, macro_four_devices
from summarize_neighbour_ablation import digest, read, write, difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        'zero-root': 'results/stage05/ablations/teacher_weight/T0',
        'zero-training': 'artifacts/lora_teacher0',
        'zero-cache': 'artifacts/features_lora_teacher0',
        'zero-local': 'artifacts/runs_lora_teacher0',
        'one-root': 'results/stage04', 'one-training': 'artifacts/lora',
        'one-cache': 'artifacts/features_lora', 'one-local': 'artifacts/runs_lora',
        'frozen': 'results/stage02', 'frozen-local': 'artifacts/runs',
        'manifest': 'results/stage00/manifest.json',
        'data-root': '../IPAD_dataset/IPAD_dataset',
        'out': 'results/stage05/ablations/teacher_weight/paired_audit',
    }
    for name, default in defaults.items():
        parser.add_argument('--' + name, type=Path, default=Path(default))
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    # A distinct namespace is required for every treatment and its evidence.
    for kind in ('root', 'training', 'cache', 'local'):
        if getattr(args, 'zero_' + kind).resolve() == getattr(args, 'one_' + kind).resolve():
            raise ValueError('Teacher treatments must use distinct namespaces')
    manifest = json.loads(args.manifest.read_text())['sequences']
    required = [(m, mode, d, s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS]
    completed = []
    for key in required:
        condition = Path(key[0])/key[1]/key[2]/f'seed{key[3]}'
        valid = True
        for kind in ('zero', 'one'):
            folder = getattr(args, kind + '_root')/condition
            for name, status in [('metrics.json', 'complete_device_evaluation'),
                                 ('lora_training.json', 'complete_training')]:
                source = folder/name
                if not source.is_file() or json.loads(source.read_text()).get('status') != status:
                    valid = False
        if valid:
            completed.append(key)
    if not completed or (args.require_full and len(completed) != 48):
        raise ValueError('Actual completed teacher 0/1 pairs required; no frozen substitution')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'validation.json').unlink(missing_ok=True)
    paired_runs, checks = {}, []
    for key in completed:
        condition = Path(key[0])/key[1]/key[2]/f'seed{key[3]}'
        folders = [getattr(args, k + '_root')/condition for k in ('zero', 'one')]
        metadata = [json.loads((f/'lora_training.json').read_text()) for f in folders]
        normals = [json.loads((f/'normal_fit.json').read_text()) for f in folders]
        controls = check_pair_controls(*metadata, *[read(f/'lora_training.csv') for f in folders],
                                       *normals, key)
        reference = args.frozen/condition
        frozen_info = json.loads((reference/'metrics.json').read_text())
        proofs, runs = [], {}
        for kind, weight in [('zero', 0.), ('one', 1.)]:
            fresh, proof, target = audit_lora(
                getattr(args, kind + '_root')/condition,
                getattr(args, kind + '_training')/condition,
                getattr(args, kind + '_cache')/condition,
                getattr(args, kind + '_local')/condition,
                reference, args.frozen_local/condition, manifest, args.data_root,
                key, weight, args.out/('T0' if weight == 0 else 'T1'))
            for variant in VARIANTS:
                check_pair(fresh[variant], load_run(reference, variant, frozen_info))
            treatment_meta = metadata[0 if weight == 0 else 1]
            cache_checks = proof['tensor_provenance']['cache_checks']
            for split, field in [('fit', 'fit_clips'), ('calibration', 'calibration_clips')]:
                if sum(r['clips'] for r in cache_checks if r['split'] == split) != treatment_meta[field]:
                    raise ValueError('Teacher training clip counts differ from actual adapted targets')
            runs['T0' if weight == 0 else 'T1'] = fresh
            proofs.append({'teacher_weight': weight,
                           'condition_audit': str(target.relative_to(args.out)),
                           'condition_audit_sha256': digest(target)})
        for variant in VARIANTS:
            check_pair(runs['T0'][variant], runs['T1'][variant])
        paired_runs[key] = runs
        checks.append({'condition': list(key), 'matched_controls': controls,
                       'treatments': proofs})
        print('Actual teacher pair independently audited: ' + str(condition), flush=True)
    groups = [(m, mode, d) for m in MODELS for mode in MODES for d in DEVICES
              if ready_group(paired_runs, m, mode, d)]
    stored, points, draws, rows, deltas = {}, {}, {}, [], []
    for model, mode, device in groups:
        for treatment in ('T0', 'T1'):
            for variant in VARIANTS:
                runs = [paired_runs[model, mode, device, s][treatment][variant] for s in SEEDS]
                for run in runs[1:]:
                    check_pair(runs[0], run)
                values = bootstrap_draws(runs)
                keep = np.isfinite(values).all(axis=1)
                if keep.sum() < 950:
                    raise ValueError('Too many degenerate paired video resamples')
                point = np.mean([statistic(r, list(r)) for r in runs], axis=0)
                lo, hi = np.quantile(values[keep], [.025, .975], axis=0)
                key = (model, mode, device, treatment + '_' + variant)
                stored[key], points[key], draws[key] = (runs, values[keep]), point, values
                labels = np.concatenate([v[1] for v in runs[0].values()])
                rows.append({'backbone': model, 'mode': mode, 'device': device,
                             'variant': key[3], 'teacher_weight': 0. if treatment == 'T0' else 1.,
                             'score_variant': variant, 'seeds': 3, 'test_videos': len(runs[0]),
                             'frames': len(labels), 'anomaly_frames': int(labels.sum()),
                             'auroc_mean': float(point[0]), 'auroc_ci_low': float(lo[0]),
                             'auroc_ci_high': float(hi[0]), 'ap_mean': float(point[1]),
                             'ap_ci_low': float(lo[1]), 'ap_ci_high': float(hi[1]),
                             'bootstrap_draws': 1000, 'bootstrap_rejected': int((~keep).sum())})
        for variant in VARIANTS:
            zero, one = [(model, mode, device, t + '_' + variant) for t in ('T0', 'T1')]
            delta = draws[zero] - draws[one]
            keep = np.isfinite(delta).all(axis=1)
            if keep.sum() < 950:
                raise ValueError('Too many degenerate teacher difference draws')
            deltas.append(difference(model, mode, 'T0_minus_T1_' + variant,
                                     points[zero] - points[one], delta[keep], device))
    macro = macro_four_devices(stored)
    for model in MODELS:
        for mode in MODES:
            if not all((model, mode, d) in groups for d in DEVICES):
                continue
            for variant in VARIANTS:
                delta = np.mean([
                    bootstrap_draws(stored[model, mode, d, 'T0_' + variant][0],
                                    seed=np.random.SeedSequence([2026, int(d[1:])]))
                    - bootstrap_draws(stored[model, mode, d, 'T1_' + variant][0],
                                      seed=np.random.SeedSequence([2026, int(d[1:])]))
                    for d in DEVICES], axis=0)
                keep = np.isfinite(delta).all(axis=1)
                if keep.sum() < 950:
                    raise ValueError('Too many degenerate Macro4 teacher draws')
                point = np.mean([points[model, mode, d, 'T0_' + variant]
                                 - points[model, mode, d, 'T1_' + variant] for d in DEVICES], axis=0)
                macro['paired_deltas'].append(difference(model, mode, 'T0_minus_T1_' + variant,
                                                        point, delta[keep]))
    scope = ('Actual matched teacher 0/1 joint LoRA; per-seed metric mean over seeds0/1/2; '
             'paired whole-video bootstrap1000; equal-device Macro4 only with all4; no runtime claim')
    if not rows:
        (args.out/'device_summary.csv').unlink(missing_ok=True)
    if not macro['results']:
        (args.out/'macro_summary.csv').unlink(missing_ok=True)
    write(args.out/'device_summary', {'scope': scope, 'results': rows, 'paired_deltas': deltas})
    write(args.out/'macro_summary', macro)
    validation = {'status': 'passed', 'matrix_complete': len(completed) == 48,
                  'completed_seed_pairs': len(completed), 'complete_three_seed_groups': len(groups),
                  'treatment_conditions_checked': len(completed)*2,
                  'normal_thresholds_checked': len(completed)*8,
                  'pair_checks': checks, 'pending_pairs': [list(k) for k in required if k not in completed],
                  'device_summary_sha256': digest(args.out/'device_summary.json'),
                  'macro_summary_sha256': digest(args.out/'macro_summary.json'),
                  'verifier_sha256': digest(__file__),
                  'control_verifier_sha256': digest('src/ipad_jepa/teacher_weight_audit.py'),
                  'condition_verifier_sha256': digest(Path(__file__).with_name('summarize_lora_matrix.py')),
                  'manifest_sha256': digest(args.manifest), 'scope': scope}
    (args.out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k not in ['pair_checks', 'pending_pairs']}, indent=2))


if __name__ == '__main__':
    main()
