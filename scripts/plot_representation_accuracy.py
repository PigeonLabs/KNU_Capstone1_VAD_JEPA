"""Render verified shared-head patch/spatial-mean accuracy, including partial scope."""
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_neighbour_ablation import digest, save, NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/stage05/ablations/representation'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage05'))
    args = parser.parse_args()
    proof = json.loads((args.root/'validation.json').read_text())
    if (proof['status'] != 'passed' or proof['shared_phase_pairs_checked'] != proof['completed_groups']*3
            or proof['thresholds_checked'] != proof['completed_groups']*6
            or proof['verifier_sha256'] != digest(Path(__file__).with_name('summarize_representation_ablation.py'))):
        raise ValueError('Current verified shared-head three-seed representation comparisons required')
    for name in ('device_summary', 'macro_summary'):
        if digest(args.root/f'{name}.json') != proof[name+'_sha256']:
            raise ValueError('Audited representation summary changed')
    full = proof['matrix_complete']
    summary = json.loads((args.root/('macro_summary.json' if full else 'device_summary.json')).read_text())
    rows = summary['results']
    keys = sorted({(r['backbone'],r['mode'],None if full else r['device']) for r in rows})
    hashes = {}
    for model, mode, device in keys:
        selected = [r for r in rows if r['backbone']==model and r['mode']==mode
                    and (full or r['device']==device)]
        values = [next(r for r in selected if r['variant']==representation) for representation in ('patch','global_mean')]
        if len(selected) != 2 or any(r['seeds']!=3 for r in values):
            raise ValueError('Exactly two completed three-seed representations required')
        paired = [r for r in summary['paired_deltas'] if r['backbone']==model and r['mode']==mode
                  and r['comparison']=='global_mean_minus_patch' and (full or r['device']==device)]
        if len(paired) != 1:
            raise ValueError('Exact paired representation difference required')
        fig, axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for axis, metric in zip(axes[:2], ('auroc','ap')):
            for x, row, colour in zip((0,1),values,('#3979ae','#db8b37')):
                point, lo, hi = [row[metric+suffix]*100 for suffix in ('_mean','_ci_low','_ci_high')]
                axis.bar(x,point,color=colour,width=.5)
                axis.vlines(x,lo,hi,color='#202d39',linewidth=1.5)
                axis.hlines([lo,hi],x-.08,x+.08,color='#202d39')
                axis.text(x+.13,point+2,f'{point:.2f}',ha='left',fontsize=9)
            axis.set(xticks=[0,1],xticklabels=['Patch tokens','Spatial mean'],ylim=(0,100),ylabel=metric.upper()+' (%)')
        axis = axes[2]
        for y, metric, colour in zip((0,1),('auroc','ap'),('#3979ae','#db8b37')):
            point,lo,hi = [paired[0][metric+suffix]*100 for suffix in ('_delta','_delta_ci_low','_delta_ci_high')]
            axis.plot(point,y,'o',color=colour)
            axis.hlines(y,lo,hi,color=colour,linewidth=2)
            axis.vlines([lo,hi],y-.06,y+.06,color=colour)
        axis.axvline(0,color='#87939d',linestyle='--',linewidth=1)
        axis.set(yticks=[0,1],yticklabels=['AUROC','AP'],ylim=(-.5,1.5),xlabel='Spatial mean minus patch (pp)',title='Paired 95% CI')
        for axis in axes:
            axis.spines[['top','right']].set_visible(False)
        scope = 'equal-weight Macro4' if full else f'{device} only / partial matrix'
        fig.suptitle(f'{NAMES[model]} / {mode} / {scope}',fontsize=13)
        note = ('P3; both normal fits stride1, shared phase head; three seed metric mean; whole-video bootstrap CI (1000 draws).\n'
                'Shared targets t=19..N-8/GT; patch vs FP32 spatial mean of FP16 local patches; per-representation PCA/bank/q99.\n'
                'Accuracy comparison; no runtime claim. Source: representation/'+('macro_summary.json' if full else 'device_summary.json'))
        prefix = 'representation_'+model+'_'+mode+('_macro4' if full else '_'+device)
        hashes.update(save(fig,args.out/prefix,note))
    receipt = {'status':'rendered_from_verified_summaries','matrix_complete':full,
               'completed_groups':proof['completed_groups'],'validation_sha256':digest(args.root/'validation.json'),
               'device_summary_sha256':proof['device_summary_sha256'],'macro_summary_sha256':proof['macro_summary_sha256'],
               'plot_code_sha256':digest(__file__),'shared_plot_code_sha256':digest(Path(__file__).with_name('plot_neighbour_ablation.py')),
               'reporting_runtime':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
               'figures':hashes,'scope':'Completed three-seed shared-head representation groups only; final PNG review required'}
    (args.root/'figure_sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
