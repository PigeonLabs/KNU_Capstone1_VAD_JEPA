"""Export verified completed normal-only LoRA curves without checkpoint tensors."""
import argparse
import json
from pathlib import Path

from ipad_jepa.adapted_features import selected_protocol
from ipad_jepa.cache_retention import digest
from ipad_jepa.lora_audit import check_training
from summarize_neighbour_ablation import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    pins = {str(args.run / name): digest(args.run / name)
            for name in ['lora_training.json', 'lora_training.csv', 'selected_adapter.pt']}
    meta, _ = selected_protocol(args.run)
    condition = tuple(meta[key] for key in ['backbone', 'mode', 'device', 'seed'])
    rows = read(args.run / 'lora_training.csv')
    check_training(meta, rows, condition, meta['teacher_weight'])
    selected = rows[meta['selected_epoch'] - 1]
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('Completed selected training changed during export')
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ['lora_training.json', 'lora_training.csv']:
        target = args.out / name
        data = (args.run / name).read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Another training identity exists at export destination')
        target.write_bytes(data)
    receipt = {'status': 'complete_training_export',
               **{key: meta[key] for key in ['backbone', 'mode', 'device', 'seed', 'teacher_weight']},
               'epochs': 20, 'training_metadata_sha256': digest(args.out / 'lora_training.json'),
               'training_curve_sha256': digest(args.out / 'lora_training.csv'),
               'selected_checkpoint_sha256': pins[str(args.run / 'selected_adapter.pt')],
               'selected_epoch': meta['selected_epoch'],
               'selected_normal_calibration_circular_mae': float(selected['normal_calibration_circular_mae']),
               'exporter_sha256': digest(__file__),
               'scope': 'Actual complete 20-epoch normal-only training, selected checkpoint protocol '
                        'and fixed normal-CE selection verified; no checkpoint export or accuracy/runtime claim.'}
    (args.out / 'training_export.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
