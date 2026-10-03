"""Audit and export one shared dense-fit normal phase head, without tensors."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ipad_jepa.backbones import PhaseHead
from ipad_jepa.cache_data import sequences
from summarize_neighbour_ablation import digest
from summarize_representation_ablation import audit_dense


def check_curve(info, rows):
    if (info.get('status') != 'complete' or info.get('epochs') != 20
            or len(rows) != 20 or [int(r['epoch']) for r in rows] != list(range(1, 21))):
        raise ValueError('Complete twenty-epoch shared phase curve required')
    values = np.array([[float(r[k]) for k in ('train_ce', 'normal_calibration_ce',
                                              'normal_calibration_circular_mae')] for r in rows])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values[:, 2] > .5):
        raise ValueError('Finite nonnegative phase objectives and circular MAE required')
    selected = min(rows, key=lambda r: float(r['normal_calibration_ce']))
    if (int(selected['epoch']) != info.get('selected_epoch')
            or float(selected['normal_calibration_ce']) != info.get('best_normal_calibration_ce')):
        raise ValueError('Shared head must use first minimum normal CE, not MAE')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dense-fit', type=Path, required=True)
    p.add_argument('--original-cache', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('results/stage00/manifest.json'))
    p.add_argument('--producer-ledger', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Fresh shared phase export namespace required')
    info = json.loads((args.run / 'phase_training.json').read_text())
    with (args.run / 'phase_training.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    selected = check_curve(info, rows)
    if (info.get('fit_stride') != 1 or info.get('phase_batch_size') != 256
            or info.get('shared_between_representations') is not True
            or info.get('seed') not in (0, 1, 2)
            or info.get('selection') != 'Minimum normal calibration CE; first epoch wins ties; no anomaly labels'
            or info.get('code_sha256') != digest('src/ipad_jepa/representation_ablation.py')):
        raise ValueError('Original shared normal-only phase protocol required')
    manifest = json.loads(args.manifest.read_text())['sequences']
    producer = json.loads(args.producer_ledger.read_text())
    exporter = str(Path(__file__).resolve().relative_to(Path.cwd().resolve()))
    public_pins = {name: digest(name) for name in (str(args.manifest), exporter)}
    for name, expected in producer['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Original representation producer changed: ' + name)
        public_pins[str(Path(name).resolve().relative_to(Path.cwd().resolve()))] = expected
    dense, fit = audit_dense(args.dense_fit, manifest, info['backbone'], info['mode'], info['device'])
    normal_rows = [r for r in manifest if r['device'] == info['device']]
    calibration = sequences(args.original_cache, normal_rows, 'calibration')
    expected = dict(fit_input_sha256=digest(args.dense_fit / 'dense_fit.json'),
                    fit_clips=dense['capacity']['fit_clips'],
                    calibration_clips=sum(len(s.targets) for s in calibration),
                    normal_calibration_cache_fingerprints=[s.meta['fingerprint'] for s in calibration])
    if any(info.get(k) != value for k, value in expected.items()):
        raise ValueError('Shared phase inputs differ from actual dense normal/calibration caches')
    if any(s.identity != dense['original_cache_identity'] for s in calibration):
        raise ValueError('Original normal calibration encoder identity differs')
    if set(r['sequence'] for r in fit) & set(s.row['sequence'] for s in calibration):
        raise ValueError('Normal fit and calibration videos overlap')
    private_pins = {str(args.run / name): digest(args.run / name)
                    for name in ('phase_training.json', 'phase_training.csv', 'phase_head.pt')}
    private_pins[str(args.dense_fit / 'dense_fit.json')] = expected['fit_input_sha256']
    private_pins[str(args.producer_ledger)] = digest(args.producer_ledger)
    private_pins.update({str(args.dense_fit / name): value for name, value in dense['outputs_sha256'].items()})
    for s in calibration:
        for name in ('meta.json', 'targets.npy', 'global.npy'):
            private_pins[str(s.folder / name)] = digest(s.folder / name)
        if not np.isfinite(s.global_features).all():
            raise ValueError('Nonfinite original normal calibration phase input')
    head = PhaseHead()
    head.load_state_dict(torch.load(args.run / 'phase_head.pt', map_location='cpu', weights_only=True), strict=True)
    if any(not value.isfinite().all() for value in head.state_dict().values()):
        raise ValueError('Nonfinite selected shared head tensors')
    if sum(v.numel() for v in head.parameters()) != info['trainable_parameters']:
        raise ValueError('Selected head architecture differs')
    for name, expected_hash in {**public_pins, **private_pins}.items():
        if digest(name) != expected_hash:
            raise ValueError('Shared phase source changed during export: ' + name)
    args.out.mkdir(parents=True)
    for name in ('phase_training.json', 'phase_training.csv'):
        (args.out / name).write_bytes((args.run / name).read_bytes())
    (args.out / 'dense_fit_source.json').write_bytes((args.dense_fit / 'dense_fit.json').read_bytes())
    receipt = dict(status='verified_completed_shared_phase_artifacts_export',
                   condition=[info[k] for k in ('backbone', 'mode', 'device', 'seed')],
                   selected_epoch=info['selected_epoch'], epochs=20,
                   selected_normal_calibration_ce=float(selected['normal_calibration_ce']),
                   selected_normal_calibration_circular_mae=float(selected['normal_calibration_circular_mae']),
                   fit_videos=len(fit), calibration_videos=len(calibration),
                   fit_clips=info['fit_clips'], calibration_clips=info['calibration_clips'],
                   selected_head_sha256=private_pins[str(args.run / 'phase_head.pt')],
                   producer_status_at_export=producer['status'], original_training_exit_code=None,
                   source_sha256=public_pins, private_source_sha256=private_pins,
                   scope='Current actual twenty-epoch curve, minimum normal-CE selection, dense normal arrays/candidates, original normal calibration input and strict CPU selected-head loading verified. Original trainer exit code not retained. No retraining, GPU/encoder/CE replay, new PCA/memory/test evaluation, paired representation accuracy, CI, Macro4 or runtime claim. Checkpoint and arrays remain local.')
    (args.out / 'phase_export.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('source_sha256', 'private_source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
