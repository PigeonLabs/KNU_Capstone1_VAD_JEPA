"""Verify completed diagnostic groups and cluster all interventions by original video."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ipad_jepa.audit import inspect_frames
from ipad_jepa.diagnostics import scenarios, CLASSES
from ipad_jepa.diagnostic_audit import audit_trace, class_f1, summary_row
from summarize_clip_ablation import audit_calibration

MODELS = ('dinov3-l', 'vjepa21-l')
MODES = ('offline', 'online')
DEVICES = ('R01', 'R02', 'R03', 'R04')
COUNTS = dict(zip(DEVICES, (6, 4, 3, 4)))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def source_inventory(record, manifest):
    names = ('diagnostic_evaluation', 'diagnostics', 'backbones', 'features', 'audit',
             'temporal', 'scoring', 'torch_memory', 'memory')
    required = {f'src/ipad_jepa/{name}.py' for name in names}
    required |= {str(manifest), 'configs/experiment_matrix.yaml'}
    normalized = {}
    for name, sha in record['source_sha256'].items():
        path = Path(name)
        if path.is_absolute() and path.parent.name == 'ipad_jepa' and path.stem in names:
            name = 'src/ipad_jepa/'+path.name
        if name not in required or name in normalized or digest(name) != sha:
            raise ValueError('Diagnostic pinned source inventory changed')
        normalized[name] = sha
    if set(normalized) != required:
        raise ValueError('Diagnostic required source missing')
    return normalized


def fixed_source(record, condition, primary, runs, manifest):
    proof = record['normal_only_fixed_source']
    public, local = primary/condition, runs/condition
    files = {'normal_fit': public/'normal_fit.json',
             'normal_calibration': public/'normal_calibration.csv',
             'phase_training': local/'phase_training.json',
             'phase_head': local/'phase_head.pt', 'memory': local/'memory.npz'}
    for name, path in files.items():
        if digest(path) != proof[name+'_sha256']:
            raise ValueError('Fixed normal source changed: '+str(path))
    fit = json.loads(files['normal_fit'].read_text())
    phase = json.loads(files['phase_training'].read_text())
    model, mode, device, seed_dir = condition.parts
    seed = int(seed_dir[4:])
    identity = (model, mode, device, seed)
    if (fit['status'] != 'normal_fit_and_calibration_complete' or phase['status'] != 'complete'
            or tuple(fit[k] for k in ('backbone','mode','device','seed')) != identity
            or tuple(phase[k] for k in ('backbone','mode','device','seed')) != identity
            or proof['seed'] != seed or phase['epochs'] != 20
            or phase['selected_epoch'] != fit['phase_selected_epoch']
            or fit['phase_checkpoint_sha256'] != proof['phase_head_sha256']
            or sorted(fit['normal_cache_fingerprints']) != sorted(phase['cache_fingerprints'])
            or (fit['bins'],fit['prototypes_per_bin'],fit['total_prototypes'],fit['pca_dimensions']) != (16,128,2048,256)
            or fit['cache_identity']['backbone'] != model or fit['cache_identity']['mode'] != mode
            or fit['cache_identity']['clip_frames'] != 16
            or fit['cache_identity']['adapter_sha256'] != digest('src/ipad_jepa/backbones.py')
            or fit['cache_identity']['reader_sha256'] != digest('src/ipad_jepa/features.py')):
        raise ValueError('Required unchanged normal head, memory or identity differs')
    curve = read(local/'phase_training.csv')
    losses = np.array([float(r['normal_calibration_ce']) for r in curve])
    if (len(curve) != 20 or [int(r['epoch']) for r in curve] != list(range(1,21))
            or not np.isfinite(losses).all() or phase['selected_epoch'] != int(losses.argmin())+1):
        raise ValueError('Head must minimize normal calibration CE across all 20 epochs')
    cycle = float(np.median([r['frames'] for r in manifest
                            if r['device'] == device and r.get('split') == 'fit']))
    if cycle != proof['cycle_length_normal_fit'] or cycle != fit['cycle_length_fit_median']:
        raise ValueError('Cycle length is not the original normal fit median')
    params = fit['calibration']['P3']
    if params != proof['P3_normal_calibration']:
        raise ValueError('Diagnostic thresholds differ from original normal calibration')
    samples = audit_calibration(public, manifest, device, mode, 16, cycle, params)
    with np.load(local/'memory.npz', allow_pickle=False) as bank:
        if (bank['mean'].shape != (1024,) or bank['components'].shape != (256,1024)
                or bank['prototypes'].shape != (16,128,256)
                or any(not np.isfinite(bank[k]).all() for k in ('mean','components','prototypes'))
                or float(bank['temperature']) != fit['temperature'] or float(bank['cycle_length']) != cycle):
            raise ValueError('Actual fixed normal bank differs')
    return params, cycle, samples, {str(path): digest(path) for path in files.values()}


def audit_condition(root, condition, rows, manifest, primary, runs, manifest_path):
    folder = root/condition
    metric_path = folder/'diagnostic_metrics.json'
    record = json.loads(metric_path.read_text())
    model, mode, device, seed_dir = condition.parts
    seed = int(seed_dir[4:])
    if (record['status'] != 'complete_diagnostic_condition'
            or tuple(record[k] for k in ('backbone','mode','device','seed')) != (model,mode,device,seed)
            or record['classes'] != list(CLASSES) or record['held_out_videos'] != COUNTS[device]
            or record['scenarios_per_video'] != 49 or record['valid_mask'] != 't=19..N-8 inclusive'):
        raise ValueError('Completed full diagnostic condition required')
    pins = source_inventory(record, manifest_path)
    params, cycle, normal_samples, normal_hashes = fixed_source(record, condition, primary, runs, manifest)
    recipe_proofs = {r['sequence']: r for r in record['recipes']}
    if len(recipe_proofs) != len(record['recipes']) or set(recipe_proofs) != {r['sequence'] for r in rows}:
        raise ValueError('Diagnostic recipe video inventory differs')
    expected, checksums, matrices = set(), {}, []
    for row in rows:
        recipe_path = folder.parent/'recipes'/f"{row['sequence']}.json"
        declared = recipe_proofs[row['sequence']]
        expected_suffix = str(Path(model)/mode/device/'recipes'/recipe_path.name)
        if (not declared['path'].endswith('/'+expected_suffix)
                or declared['sha256'] != digest(recipe_path)):
            raise ValueError('Recorded recipe path/hash differs')
        cases = scenarios(row)
        recipe_record = {'sequence':row['sequence'], 'frames':row['frames'],
                         'frames_content_sha256':row['frames_content_sha256'],
                         'frame_names_sha256':row['names_sha256'],
                         'recipes':[case.metadata() for case in cases]}
        # JSON serializes the dataclass box tuple as a list.
        recipe_record = json.loads(json.dumps(recipe_record))
        if json.loads(recipe_path.read_text()) != recipe_record or len(cases) != 49:
            raise ValueError('Recorded interventions differ from complete fixed protocol')
        checksums[str(recipe_path.relative_to(root))] = digest(recipe_path)
        matrix = np.zeros((4,4), np.int64)
        for case in cases:
            relative = f"traces/{row['sequence']}/{case.name}.csv"
            expected.add(relative)
            path = folder/relative
            actual = digest(path)
            if record['trace_sha256'].get(relative) != actual:
                raise ValueError('Actual diagnostic trace checksum differs')
            matrix += audit_trace(read(path), row['frames'], mode, case.metadata(), cycle, params)
            checksums[str(path.relative_to(root))] = actual
        matrices.append(matrix)
    if (set(record['trace_sha256']) != expected
            or {p.relative_to(folder).as_posix() for p in (folder/'traces').rglob('*') if p.is_file()} != expected):
        raise ValueError('Complete trace inventory required; missing/orphan outputs found')
    total = np.sum(matrices, axis=0)
    np.testing.assert_array_equal(total, record['confusion_matrix'])
    per_class = class_f1(total)
    np.testing.assert_allclose(per_class, record['per_class_f1'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(per_class.mean(), record['macro_f1'], rtol=0, atol=1e-12)
    checksums[str(metric_path.relative_to(root))] = digest(metric_path)
    proof = {'condition': str(condition), 'normal_calibration_frames_checked': normal_samples,
             'traces_checked': len(expected), 'source_sha256': pins,
             'fixed_normal_source_sha256': normal_hashes, 'output_sha256': checksums}
    return np.stack(matrices), proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('root','results/stage05/diagnostics'),('manifest','results/stage00/manifest.json'),
                          ('primary','results/stage02'),('runs','artifacts/runs'),
                          ('data-root','../IPAD_dataset/IPAD_dataset')]:
        parser.add_argument('--'+name, type=Path, default=Path(default))
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    required = [(model,mode,device) for model in MODELS for mode in MODES for device in DEVICES]
    complete = [key for key in required if all((args.root/Path(*key)/f'seed{s}'/'diagnostic_metrics.json').is_file()
                                              for s in (0,1,2))]
    if not complete or args.require_full and len(complete) != 16:
        raise ValueError('Completed three-seed full-scenario groups required')
    (args.root/'validation.json').unlink(missing_ok=True)
    manifest = json.loads(args.manifest.read_text())['sequences']
    checks, device_rows, stored, raw, source_checks = [], [], {}, {}, {}
    for model,mode,device in complete:
        rows = sorted([r for r in manifest if r['device']==device and r.get('split')=='diagnostic'],
                      key=lambda r:r['sequence'])
        if len(rows) != COUNTS[device] or any(r['partition']!='training' for r in rows):
            raise ValueError('Held-out normal diagnostic split differs')
        for row in rows:
            key = device+'/'+row['sequence']
            if key not in source_checks:
                actual = inspect_frames(args.data_root/row['relative_directory'])
                if any(actual[k] != row[k] for k in ('frames','names_sha256','frames_content_sha256')):
                    raise ValueError('Actual held-out input content differs from audited manifest')
                source_checks[key] = actual
        seed_matrices = []
        for seed in (0,1,2):
            condition = Path(model)/mode/device/f'seed{seed}'
            values, check = audit_condition(args.root, condition, rows, manifest,
                                           args.primary, args.runs, args.manifest)
            seed_matrices.append(values); checks.append(check)
        matrices = np.stack(seed_matrices)
        summary, boot, class_boot = summary_row(matrices, seed=np.random.SeedSequence([2026,int(device[1:])]))
        summary.update(backbone=model,mode=mode,device=device,
                       confusion_mean=matrices.sum(1).mean(0).tolist(),
                       valid_scenario_frames_per_seed=int(matrices[0].sum()),scenarios_per_video=49)
        device_rows.append(summary); stored[model,mode,device] = (summary,boot,class_boot)
        raw[f'{model}/{mode}/{device}'] = {r['sequence']:matrices[:,i].tolist() for i,r in enumerate(rows)}
    macro_rows, deltas = [], []
    macro = {}
    for model in MODELS:
        for mode in MODES:
            if all((model,mode,d) in stored for d in DEVICES):
                groups = [stored[model,mode,d] for d in DEVICES]
                point = float(np.mean([g[0]['macro_f1_mean'] for g in groups]))
                boot = np.mean([g[1] for g in groups], axis=0)
                bounds = np.quantile(boot,[.025,.975])
                macro_rows.append({'backbone':model,'mode':mode,'devices':4,'seeds':3,
                    'macro_f1_mean':point,'macro_f1_ci_low':float(bounds[0]),'macro_f1_ci_high':float(bounds[1]),
                    'per_class_f1_mean':np.mean([g[0]['per_class_f1_mean'] for g in groups],axis=0).tolist()})
                macro[model,mode] = (point,boot)
    pairs = [(f'{model}/online_minus_offline',(model,'online'),(model,'offline')) for model in MODELS]
    pairs += [(f'{mode}/vjepa21_minus_dinov3',('vjepa21-l',mode),('dinov3-l',mode)) for mode in MODES]
    for name,new,old in pairs:
        if new in macro and old in macro:
            boot = macro[new][1]-macro[old][1]
            bounds = np.quantile(boot,[.025,.975])
            deltas.append({'comparison':name,'macro_f1_difference':macro[new][0]-macro[old][0],
                           'ci_low':float(bounds[0]),'ci_high':float(bounds[1])})
    scope = ('All49 interventions clustered by original normal held-out video; fixed three-seed metric mean; '
             'device-stratified paired bootstrap1000; equal-device Macro4; fixed original normal component q99; '
             'intervention labels, not real fault types or runtime')
    write(args.root/'device_summary.json', {'scope':scope,'results':device_rows})
    write(args.root/'macro_summary.json', {'scope':scope,'results':macro_rows,'paired_deltas':deltas})
    write(args.root/'video_confusions.json', {'classes':list(CLASSES),'groups':raw})
    proof = {'status':'passed','matrix_complete':len(complete)==16,'completed_groups':len(complete),
             'seed_conditions_checked':len(checks),'normal_thresholds_checked':len(checks),
             'traces_checked':sum(c['traces_checked'] for c in checks),'actual_source_videos':source_checks,
             'checks':checks,'manifest_sha256':digest(args.manifest),'verifier_sha256':digest(__file__),
             'replay_code_sha256':digest('src/ipad_jepa/diagnostic_audit.py'),
             'shared_normal_calibration_auditor_sha256':digest(Path(__file__).with_name('summarize_clip_ablation.py')),
             'summary_sha256':{name:digest(args.root/name) for name in
                               ('device_summary.json','macro_summary.json','video_confusions.json')},
             'scope':scope,'audit_limits':'Replays recorded phase/time/normal normalization/types/truth/confusion. '
                'Checks actual input, fixed head/bank/normal source hashes and bank geometry. '
                'Does not repeat encoder extraction, GPU distance calculation or PCA fitting.'}
    write(args.root/'validation.json', proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('checks','actual_source_videos')},indent=2))


if __name__ == '__main__':
    main()
