"""Render independently verified online B/L paired accuracy with explicit partial scope."""
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
    parser.add_argument('--root', type=Path, default=Path('results/stage05/ablations/backbone_size'))
    parser.add_argument('--out', type=Path, default=Path('docs/figures/stage05'))
    args = parser.parse_args()
    proof = json.loads((args.root/'validation.json').read_text())
    if (proof['status'] != 'passed' or proof['small_seed_conditions_checked'] != proof['completed_groups']*3
            or proof['thresholds_checked'] != proof['completed_groups']*6
            or proof['verifier_sha256'] != digest(Path(__file__).with_name('summarize_backbone_size.py'))):
        raise ValueError('Current independently verified three-seed B/L comparisons required')
    if proof['source_audit_sha256'] != digest('src/ipad_jepa/backbone_audit.py'):
        raise ValueError('B source audit changed')
    for name in ('device_summary', 'macro_summary'):
        if digest(args.root/f'{name}.json') != proof[name+'_sha256']:
            raise ValueError('Audited B/L summary changed')
    full = proof['matrix_complete']
    summary = json.loads((args.root/('macro_summary.json' if full else 'device_summary.json')).read_text())
    rows = summary['results']
    keys = sorted({(r['backbone'],r['mode'],None if full else r['device']) for r in rows})
    hashes = {}
    for model, mode, device in keys:
        selected = [r for r in rows if r['backbone']==model and r['mode']==mode
                    and (full or r['device']==device)]
        values = [next(r for r in selected if r['variant']==representation) for representation in ('L','B')]
        if len(selected) != 2 or any(r['seeds']!=3 for r in values):
            raise ValueError('Exactly two completed three-seed B/L sizes required')
        paired = [r for r in summary['paired_deltas'] if r['backbone']==model and r['mode']==mode
                  and r['comparison']=='B_minus_L' and (full or r['device']==device)]
        if len(paired) != 1:
            raise ValueError('Exact paired B/L difference required')
        fig, axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for axis, metric in zip(axes[:2], ('auroc','ap')):
            for x, row, colour in zip((0,1),values,('#3979ae','#db8b37')):
                point, lo, hi = [row[metric+suffix]*100 for suffix in ('_mean','_ci_low','_ci_high')]
                axis.bar(x,point,color=colour,width=.5)
                axis.vlines(x,lo,hi,color='#202d39',linewidth=1.5)
                axis.hlines([lo,hi],x-.08,x+.08,color='#202d39')
                axis.text(x+.13,point+2,f'{point:.2f}',ha='left',fontsize=9)
            axis.set(xticks=[0,1],xticklabels=['L (1024-dim)','B (768-dim)'],ylim=(0,100),ylabel=metric.upper()+' (%)')
        axis = axes[2]
        for y, metric, colour in zip((0,1),('auroc','ap'),('#3979ae','#db8b37')):
            point,lo,hi = [paired[0][metric+suffix]*100 for suffix in ('_delta','_delta_ci_low','_delta_ci_high')]
            axis.plot(point,y,'o',color=colour)
            axis.hlines(y,lo,hi,color=colour,linewidth=2)
            axis.vlines([lo,hi],y-.06,y+.06,color=colour)
        axis.axvline(0,color='#87939d',linestyle='--',linewidth=1)
        axis.set(yticks=[0,1],yticklabels=['AUROC','AP'],ylim=(-.5,1.5),xlabel='B minus L (pp)',title='Paired 95% CI')
        for axis in axes:
            axis.spines[['top','right']].set_visible(False)
        scope = 'equal-weight Macro4' if full else f'{device} only / partial matrix'
        family = 'DINOv3' if model == 'dinov3-l' else 'V-JEPA 2.1'
        fig.suptitle(f'{family} / B vs L / {mode} / {scope}',fontsize=13)
        note = ('Frozen online P3; separately refitted normal heads/PCA/banks/q99; three seed metric mean; whole-video CI (1000 draws).\n'
                'Shared targets t=19..N-8/GT; 16 frames, 576 local patches, PCA256, 2048 prototypes; different pretrained checkpoints.\n'
                'Accuracy comparison; no runtime claim. Source: backbone_size/'+('macro_summary.json' if full else 'device_summary.json'))
        prefix = 'backbone_size_'+model+'_'+mode+('_macro4' if full else '_'+device)
        hashes.update(save(fig,args.out/prefix,note))
    receipt = {'status':'rendered_from_verified_summaries','matrix_complete':full,
               'completed_groups':proof['completed_groups'],'validation_sha256':digest(args.root/'validation.json'),
               'device_summary_sha256':proof['device_summary_sha256'],'macro_summary_sha256':proof['macro_summary_sha256'],
               'plot_code_sha256':digest(__file__),'shared_plot_code_sha256':digest(Path(__file__).with_name('plot_neighbour_ablation.py')),
               'reporting_runtime':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
               'figures':hashes,'scope':'Completed three-seed paired B/L groups only; final PNG review required'}
    (args.root/'figure_sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
