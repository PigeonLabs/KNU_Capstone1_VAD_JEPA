"""Export compact normal-only phase logs; leave all model tensors local."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('artifacts/runs'))
    parser.add_argument('--out',type=Path,default=Path('results/stage02'))
    args=parser.parse_args()
    summaries=[]
    for metadata in sorted(args.runs.glob('*/*/*/seed*/phase_training.json')):
        info=json.loads(metadata.read_text())
        if info['status']!='complete':
            continue
        with metadata.with_suffix('.csv').open() as stream:
            curve=list(csv.DictReader(stream))
        selected=next(r for r in curve if int(r['epoch'])==info['selected_epoch'])
        if len(curve)!=info['epochs']:
            raise ValueError('Incomplete training curve')
        target=args.out/metadata.parent.relative_to(args.runs)
        target.mkdir(parents=True,exist_ok=True)
        for name in ['phase_training.json','phase_training.csv']:
            shutil.copyfile(metadata.parent/name,target/name)
        summaries.append({k:info[k] for k in ['backbone','mode','device','seed','fit_clips','calibration_clips','selected_epoch']} | {
            'selected_normal_calibration_ce':float(selected['normal_calibration_ce']),
            'selected_normal_calibration_circular_mae':float(selected['normal_calibration_circular_mae']),
            'last_normal_calibration_circular_mae':float(curve[-1]['normal_calibration_circular_mae'])})
    if not summaries:
        raise ValueError('No completed phase-head training logs')
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'phase_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    print(json.dumps({'exported_runs':len(summaries),'scope':'Normal-only phase prediction; no anomaly AUROC/AP/FPS'},indent=2))


if __name__=='__main__':
    main()
