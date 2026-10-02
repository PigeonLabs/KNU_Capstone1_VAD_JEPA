"""Audit all primary LoRA or teacher pairs using current arrays or explicit archived retirement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_retained_lora import audit_retained
from ipad_jepa.teacher_weight_audit import check_pair_controls
from lora_reporting import summarize, MODELS, MODES, DEVICES, SEEDS, VARIANTS
from plot_lora_evaluation import calibration_check
from summarize_clip_ablation import audit_condition as audit_p3
from summarize_experiments import check_pair, load_run
from summarize_lora_matrix import audit_lora
from summarize_neighbour_ablation import digest, read, write


def treatment_paths(condition, weight):
    relative = Path(condition[0]) / condition[1] / condition[2] / f"seed{condition[3]}"
    zero = weight == 0.
    return {"public": (Path("results/stage05/ablations/teacher_weight/T0") if zero else Path("results/stage04")) / relative,
            "training": Path("artifacts/lora_teacher0" if zero else "artifacts/lora") / relative,
            "cache": Path("artifacts/features_lora_teacher0" if zero else "artifacts/features_lora") / relative,
            "local": Path("artifacts/runs_lora_teacher0" if zero else "artifacts/runs_lora") / relative,
            "inventory": Path("results/cache_retention") / ("T0" if zero else "T1") / relative / "payload_inventory.json"}


def route_condition(condition, weight, manifest_path, data_root, out):
    paths = treatment_paths(condition, weight)
    retirement = paths['inventory'].with_name('retirement.json')
    target = out / ("T0" if weight == 0 else "T1") / Path(condition[0]) / condition[1] / condition[2] / f"seed{condition[3]}"
    if retirement.exists():
        # An incomplete retirement cannot silently fall back to the current-array audit.
        release = json.loads(retirement.read_text())
        if release.get('status') != 'complete_verified_derived_payload_retirement':
            raise ValueError('Retirement exists but is incomplete; inspect its exact deletion journal')
        runs, proof = audit_retained(Path.cwd(), paths['inventory'], release['publication_revision'],
                                     manifest_path, data_root, retirement)
        path = target / 'retained_condition_audit.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(proof, indent=2) + '\n')
        records = json.loads(paths['inventory'].read_text())['records']
        cache_checks = [{'split': row['split'], 'clips': row['clips']} for row in records]
        evidence = 'archived_payloads_current_retained_replay'
    else:
        manifest = json.loads(manifest_path.read_text())['sequences']
        relative = Path(condition[0]) / condition[1] / condition[2] / f"seed{condition[3]}"
        runs, proof, path = audit_lora(paths['public'], paths['training'], paths['cache'], paths['local'],
                                      Path('results/stage02') / relative, Path('artifacts/runs') / relative,
                                      manifest, data_root, condition, weight, out / ("T0" if weight == 0 else "T1"))
        cache_checks = proof['tensor_provenance']['cache_checks']
        evidence = 'current_actual_adapted_arrays'
    return runs, {'teacher_weight': weight, 'evidence': evidence,
                  'condition_audit': str(path.relative_to(out)), 'condition_audit_sha256': digest(path)}, cache_checks


def completed(condition, weight):
    folder = treatment_paths(condition, weight)['public']
    for filename, status in [('metrics.json', 'complete_device_evaluation'), ('lora_training.json', 'complete_training')]:
        path = folder / filename
        if not path.is_file() or json.loads(path.read_text()).get('status') != status:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['lora', 'teacher'], default='lora')
    parser.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('../IPAD_dataset/IPAD_dataset'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    if args.out is None:
        args.out = Path('results/stage04/retained_matrix_audit' if args.kind == 'lora' else
                        'results/stage05/ablations/teacher_weight/retained_paired_audit')
    if args.out.resolve() in [Path('results/stage04/matrix_audit').resolve(),
                              Path('results/stage05/ablations/teacher_weight/paired_audit').resolve()]:
        raise ValueError('Use the distinct mixed current/retained evidence namespace')
    source_paths = [Path(__file__).resolve().relative_to(Path.cwd().resolve()),
                    Path('scripts/audit_retained_lora.py'), Path('scripts/lora_reporting.py'),
                    Path('scripts/summarize_lora_matrix.py'), Path('scripts/plot_lora_evaluation.py'),
                    Path('scripts/summarize_clip_ablation.py'), Path('scripts/summarize_experiments.py'),
                    Path('src/ipad_jepa/cache_retention.py'), Path('src/ipad_jepa/lora_audit.py'),
                    Path('src/ipad_jepa/teacher_weight_audit.py'), args.manifest]
    pins = {str(path): digest(path) for path in source_paths}
    required = [(model, mode, device, seed) for model in MODELS for mode in MODES for device in DEVICES for seed in SEEDS]
    weights = (1.,) if args.kind == 'lora' else (0., 1.)
    ready = [key for key in required if all(completed(key, weight) for weight in weights)]
    if not ready or args.require_full and len(ready) != 48:
        raise ValueError('Actual completed primary conditions or both completed teacher treatments required')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'validation.json').unlink(missing_ok=True)
    manifest = json.loads(args.manifest.read_text())['sequences']
    all_runs, checks = {}, []
    for condition in ready:
        relative = Path(condition[0]) / condition[1] / condition[2] / f"seed{condition[3]}"
        frozen = Path('results/stage02') / relative
        frozen_info = json.loads((frozen / 'metrics.json').read_text())
        calibration_check(frozen, frozen_info, manifest, args.data_root)
        audit_p3(frozen, manifest, args.data_root, *condition, 16)
        treatments, proofs, metadata, curves, normals = {}, [], [], [], []
        for weight in weights:
            runs, proof, cache_checks = route_condition(condition, weight, args.manifest, args.data_root, args.out)
            for variant in VARIANTS:
                check_pair(runs[variant], load_run(frozen, variant, frozen_info))
            paths = treatment_paths(condition, weight)
            meta = json.loads((paths['public'] / 'lora_training.json').read_text())
            for split, name in [('fit', 'fit_clips'), ('calibration', 'calibration_clips')]:
                if sum(row['clips'] for row in cache_checks if row['split'] == split) != meta[name]:
                    raise ValueError('Actual retained/current targets differ from training clip counts')
            treatments['L' if args.kind == 'lora' else 'T0' if weight == 0 else 'T1'] = runs
            proofs.append(proof)
            metadata.append(meta)
            curves.append(read(paths['public'] / 'lora_training.csv'))
            normals.append(json.loads((paths['public'] / 'normal_fit.json').read_text()))
        controls = None
        if args.kind == 'lora':
            treatments['F'] = {variant: load_run(frozen, variant, frozen_info) for variant in VARIANTS}
        else:
            controls = check_pair_controls(*metadata, *curves, *normals, condition)
            for variant in VARIANTS:
                check_pair(treatments['T0'][variant], treatments['T1'][variant])
        all_runs[condition] = treatments
        checks.append({'condition': list(condition), 'treatments': proofs, 'matched_controls': controls})
        print('Current/retained condition independently replayed: ' + str(relative), flush=True)
    device_summary, macro_summary, groups = summarize(all_runs, args.kind)
    for name, data in [('device_summary', device_summary), ('macro_summary', macro_summary)]:
        if not data['results']:
            (args.out / (name + '.csv')).unlink(missing_ok=True)
        write(args.out / name, data)
    if any(digest(path) != expected for path, expected in pins.items()):
        raise ValueError('Pinned mixed evidence verifier changed during replay')
    validation = {'status': 'passed_mixed_current_retained_evidence', 'kind': args.kind,
                  'matrix_complete': len(ready) == 48, 'completed_conditions_or_pairs': len(ready),
                  'complete_three_seed_groups': len(groups), 'condition_checks': checks,
                  'normal_thresholds_replayed': len(ready) * 8,
                  'retained_treatments_replayed': sum(proof['evidence'] == 'archived_payloads_current_retained_replay'
                                                     for row in checks for proof in row['treatments']),
                  'pending_conditions_or_pairs': [list(key) for key in required if key not in ready],
                  'source_sha256': pins, 'manifest_sha256': digest(args.manifest),
                  'device_summary_sha256': digest(args.out / 'device_summary.json'),
                  'macro_summary_sha256': digest(args.out / 'macro_summary.json'), 'scope': device_summary['scope']}
    (args.out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({k: v for k, v in validation.items() if k not in ['condition_checks', 'source_sha256', 'pending_conditions_or_pairs']}, indent=2))


if __name__ == '__main__':
    main()
