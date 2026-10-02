"""Visualize independently verified diagnostic accuracy and evidence confusion."""
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ipad_jepa.diagnostic_audit import normalized_confusion, equal_device_confusion
from plot_neighbour_ablation import digest, save, NAMES

CLASSES = ['Normal','Appearance','Temporal','Joint']


def heatmap(axis, values, title):
    item = axis.imshow(values*100, vmin=0, vmax=100, cmap='Blues')
    for y in range(4):
        for x in range(4):
            axis.text(x,y,f'{values[y,x]*100:.1f}',ha='center',va='center',fontsize=9,
                      color='white' if values[y,x]>.6 else '#263445')
    axis.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,
             xlabel='Predicted component-q99 evidence',ylabel='Injected intervention',title=title)
    axis.tick_params(axis='x',labelrotation=25)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/stage05/diagnostics'))
    parser.add_argument('--out',type=Path,default=Path('docs/figures/stage05'))
    args = parser.parse_args()
    proof = json.loads((args.root/'validation.json').read_text())
    if (proof['status']!='passed' or proof['seed_conditions_checked']!=3*proof['completed_groups']
            or proof['verifier_sha256']!=digest(Path(__file__).with_name('summarize_diagnostics.py'))
            or proof['replay_code_sha256']!=digest('src/ipad_jepa/diagnostic_audit.py')):
        raise ValueError('Current independently verified diagnostic results required')
    for name,sha in proof['summary_sha256'].items():
        if digest(args.root/name)!=sha:raise ValueError('Audited diagnostic summary changed')
    device = json.loads((args.root/'device_summary.json').read_text())['results']
    macro = json.loads((args.root/'macro_summary.json').read_text())
    full = proof['matrix_complete']
    hashes = {}
    note = ('49 interventions clustered within each original held-out normal video; fixed normal component q99.\n'
            'Truth is injected intervention, not real fault cause. Accuracy only; no runtime claim.\n'
            'Source: diagnostics/device_summary.json, macro_summary.json, validation.json.')
    if full:
        if proof['completed_groups']!=16 or len(device)!=16 or len(macro['results'])!=4:
            raise ValueError('Complete four-device diagnostic matrix required')
        fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
        for axis,(model,mode) in zip(axes.flat,[(m,mode) for m in NAMES for mode in ('offline','online')]):
            rows=[r for r in device if r['backbone']==model and r['mode']==mode]
            if len(rows)!=4 or {r['device'] for r in rows}!={'R01','R02','R03','R04'}:
                raise ValueError('Four distinct device confusions required')
            values=equal_device_confusion([r['confusion_mean'] for r in rows])
            item=heatmap(axis,values,NAMES[model]+' / '+mode)
        fig.colorbar(item,ax=axes,label='Mean truth-row proportion across four devices (%)',shrink=.75)
        fig.suptitle('Diagnostic evidence confusion / equal-device display',fontsize=14)
        hashes.update(save(fig,args.out/'diagnostic_macro4_confusion',note+
            '\nEach device truth row normalized before equal-weight averaging; F1 computed separately from counts.'))
    else:
        for row in device:
            fig,axis=plt.subplots(figsize=(7,6),layout='constrained')
            item=heatmap(axis,normalized_confusion(row['confusion_mean']),
                         NAMES[row['backbone']]+' / '+row['mode']+' / '+row['device']+' only')
            fig.colorbar(item,ax=axis,label='Truth-row proportion (%)',shrink=.8)
            fig.suptitle('Diagnostic evidence confusion / partial matrix',fontsize=13)
            prefix='diagnostic_'+row['backbone']+'_'+row['mode']+'_'+row['device']
            hashes.update(save(fig,args.out/prefix,note+'\nThree-seed mean counts; no full Macro4.'))
    if full:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for x,row in enumerate(macro['results']):
            point,lo,hi=[row[k]*100 for k in ('macro_f1_mean','macro_f1_ci_low','macro_f1_ci_high')]
            axes[0].plot(x,point,'o',color='#245b86')
            axes[0].vlines(x,lo,hi,color='#245b86')
            axes[0].hlines([lo,hi],x-.07,x+.07,color='#245b86')
        labels=[NAMES[r['backbone']]+'\n'+r['mode'] for r in macro['results']]
        axes[0].set(xticks=range(4),xticklabels=labels,ylim=(0,100),ylabel='Mean Macro-F1 (%)',title='Equal-device Macro4 / 95% CI')
        deltas=macro['paired_deltas']
        if len(deltas)!=4:raise ValueError('All four paired diagnostic comparisons required')
        for y,row in enumerate(deltas):
            point,lo,hi=[row[k]*100 for k in ('macro_f1_difference','ci_low','ci_high')]
            axes[1].plot(point,y,'o',color='#b87922');axes[1].hlines(y,lo,hi,color='#b87922')
        labels=[r['comparison'].replace('_minus_',' − ').replace('vjepa21','V-JEPA').replace('dinov3','DINOv3').replace('/','\n') for r in deltas]
        axes[1].axvline(0,color='#88949c',linestyle='--',linewidth=1)
        axes[1].set(yticks=range(4),yticklabels=labels,xlabel='Macro-F1 difference (pp)',title='Paired 95% CI')
        for axis in axes:axis.spines[['top','right']].set_visible(False)
        hashes.update(save(fig,args.out/'diagnostic_macro4_accuracy',note+
            '\nMean of three fixed seed metrics then equal-device mean; original-video paired bootstrap1000.'))
    receipt={'status':'rendered_from_verified_summaries','matrix_complete':full,
             'completed_groups':proof['completed_groups'],'validation_sha256':digest(args.root/'validation.json'),
             'summary_sha256':proof['summary_sha256'],'plot_code_sha256':digest(__file__),
             'shared_plot_code_sha256':digest(Path(__file__).with_name('plot_neighbour_ablation.py')),
             'reporting_runtime':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
             'figures':hashes,'scope':'Actual independently verified outputs only; final PNG review still required'}
    (args.root/'figure_sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
