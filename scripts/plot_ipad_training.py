"""Plot measured native training curves; partial runs remain labelled partial."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    meta=json.loads((args.results/'training.json').read_text())
    with (args.results/'training.csv').open() as stream: rows=list(csv.DictReader(stream))
    epochs=np.array([int(r['epoch']) for r in rows])
    if not np.array_equal(epochs,np.arange(1,len(rows)+1)) or meta['max_steps'] is not None:
        raise ValueError('Only complete-epoch native curves may be plotted')
    values={k:np.array([float(r[k]) for r in rows]) for k in ['total','mse','phase_ce']}
    if not all(np.all(np.isfinite(v)) for v in values.values()): raise ValueError('Nonfinite native curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    axes[0].plot(epochs,values['total'],marker='o',color='#245B86',label='Total weighted training loss')
    axes[0].plot(epochs,values['mse'],marker='s',linestyle='--',color='#C17A1A',label='All-frame reconstruction MSE')
    axes[0].set(ylabel='Training loss',ylim=(0,max(values['total'])*1.2)); axes[0].legend(frameon=False,fontsize=8)
    axes[1].plot(epochs,values['phase_ce'],marker='o',color='#245B86')
    axes[1].set(ylabel='Unweighted phase CE (200 classes)',ylim=(0,max(values['phase_ce'])*1.2))
    for axis in axes:
        axis.set(xlabel='Completed epoch',xticks=epochs if len(epochs)<=10 else [1,10,20,30,40,50])
        axis.spines[['top','right']].set_visible(False)
    status='complete' if meta['status']=='complete_training' else 'partial'
    fig.suptitle(f"IPAD native repaired / {meta['device']} / {meta['scope']} / seed {meta['seed']} / {status} {len(rows)}/{meta['epochs_requested']} epochs",fontsize=12)
    fig.text(.02,-.04,'Source: training.csv and training.json. Normal training only; not an anomaly evaluation or runtime benchmark.',fontsize=8)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['.png','.svg']: fig.savefig(args.out.with_suffix(extension),dpi=180,bbox_inches='tight',facecolor='white')
    svg=args.out.with_suffix('.svg'); svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__': main()
