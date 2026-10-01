"""Source-backed device plots; seed means/paired video CI and labelled seed0 curves."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score

NAMES={'dinov3-l':'DINOv3-L','vjepa21-l':'V-JEPA 2.1-L'}
COLORS={'dinov3-l':'#245B86','vjepa21-l':'#C17A1A'}


def save(fig,out,source):
    fig.text(.02,-.04,source,fontsize=8)
    out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.svg']: fig.savefig(out.with_suffix(suffix),dpi=180,bbox_inches='tight',facecolor='white')
    p=out.with_suffix('.svg'); p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


def read_csv(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/stage02'))
    p.add_argument('--out',type=Path,default=Path('docs/figures/stage02'))
    p.add_argument('--device',default='R01'); p.add_argument('--mode',default='offline')
    args=p.parse_args()
    summary=json.loads((args.root/'device_summary.json').read_text())
    records=[r for r in summary['results'] if r['device']==args.device and r['mode']==args.mode]
    models=sorted({r['backbone'] for r in records}); variants=['P0','P1','P2','P3']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for axis,metric,title in zip(axes,['auroc','ap'],['Frame AUROC','Frame AP']):
        for m,model in enumerate(models):
            rs=[next(r for r in records if r['backbone']==model and r['variant']==v) for v in variants]
            values=np.array([r[metric+'_mean'] for r in rs])*100
            low=np.array([r[metric+'_ci_low'] for r in rs])*100; high=np.array([r[metric+'_ci_high'] for r in rs])*100
            xpos=np.arange(4)+(m-(len(models)-1)/2)*.32
            # CIs may exclude the point estimate in asymmetric percentile distributions.
            axis.bar(xpos,values,width=.28,color=COLORS[model],label=NAMES[model])
            axis.vlines(xpos,low,high,color='#202020',linewidth=1.2)
            axis.hlines(low,xpos-.04,xpos+.04,color='#202020'); axis.hlines(high,xpos-.04,xpos+.04,color='#202020')
        axis.set(xticks=np.arange(4),xticklabels=variants,ylim=(0,100),ylabel=f'{title} (%)',xlabel='Memory/score variant')
        axis.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle(f'{args.device} / {args.mode} / 3-seed mean; 95% video-bootstrap CI',fontsize=13)
    save(fig,args.out/f'{args.device}_{args.mode}_variants','Source: device_summary.json. P0 global hard; P1 phase hard; P2 phase soft; P3 + temporal. Device results, not macro4.')

    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for model in models:
        for variant in ['P0','P3']:
            rows=[]
            for path in sorted((args.root/model/args.mode/args.device/'seed0'/variant).glob('*.csv')):
                rows.extend(r for r in read_csv(path) if r['valid']=='1')
            y=np.array([int(float(r['label'])) for r in rows]); score=np.array([float(r['score']) for r in rows])
            fpr,tpr,_=roc_curve(y,score); precision,recall,_=precision_recall_curve(y,score)
            style='-' if variant=='P3' else '--'
            axes[0].plot(fpr,tpr,color=COLORS[model],linestyle=style,label=f'{NAMES[model]} {variant} ({roc_auc_score(y,score):.3f})')
            axes[1].plot(recall,precision,color=COLORS[model],linestyle=style,label=f'{NAMES[model]} {variant} ({average_precision_score(y,score):.3f})')
    axes[0].plot([0,1],[0,1],color='#888888',linestyle=':',linewidth=1)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC / seed 0',xlim=(0,1),ylim=(0,1))
    axes[1].axhline(y.mean(),color='#888888',linestyle=':',linewidth=1)
    axes[1].set(xlabel='Recall',ylabel='Precision',title='PR / seed 0',xlim=(0,1),ylim=(0,1))
    for a in axes: a.legend(frameon=False,fontsize=8); a.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{args.device} / {args.mode} / same valid frames; seed 0 only',fontsize=13)
    save(fig,args.out/f'{args.device}_{args.mode}_roc_pr','Source: seed0 P0/P3 score CSVs. These curves are single-seed examples; table and variant bars use all three seeds.')

    for model in models:
        folder=args.root/model/args.mode/args.device/'seed0'
        rows=read_csv(folder/'normal_calibration.csv')
        fig,ax=plt.subplots(figsize=(6,4.6),layout='constrained')
        for seq in sorted({r['sequence'] for r in rows}):
            r=[v for v in rows if v['sequence']==seq and v['valid']=='1']
            ax.scatter([float(v['relative_phase']) for v in r],[float(v['predicted_phase']) for v in r],s=7,alpha=.65,label=f'normal cycle {seq}')
        ax.plot([0,1],[0,1],color='#777777',linestyle=':',linewidth=1)
        ax.set(xlabel='Relative-position pseudo phase (t/N)',ylabel='Predicted circular phase',xlim=(0,1),ylim=(0,1),title=f'{NAMES[model]} / normal calibration / seed 0')
        ax.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(1,1)); ax.spines[['top','right']].set_visible(False)
        save(fig,args.out/f'{model}_{args.device}_{args.mode}_phase_alignment','Source: normal_calibration.csv. Selected by minimum normal CE. Relative positions are pseudo labels, not annotated process stages.')
        # Fixed sequence 03: selected by ID, not by observed metric or model success.
        rows=[r for r in read_csv(folder/'P3'/'03.csv') if r['valid']=='1']
        meta=json.loads((folder/'normal_fit.json').read_text()); c=meta['calibration']['P3']
        frame=np.array([int(r['frame']) for r in rows]); label=np.array([int(float(r['label'])) for r in rows])
        feature=(np.array([float(r['feature_raw']) for r in rows])-c['median'][0])/c['mad_scale'][0]
        temporal=(np.array([float(r['time_raw']) for r in rows])-c['median'][1])/c['mad_scale'][1]
        score=np.array([float(r['score']) for r in rows]); alarm=np.array([int(r['alarm']) for r in rows])
        fig,axes=plt.subplots(3,1,figsize=(10,5.8),sharex=True,layout='constrained')
        axes[0].plot(frame,feature,label='Feature robust z',color='#245B86'); axes[0].plot(frame,temporal,label='Temporal robust z',color='#C17A1A')
        axes[0].set_ylabel('Component z'); axes[0].legend(frameon=False,fontsize=8)
        axes[1].plot(frame,score,color=COLORS[model],label='P3 total'); axes[1].axhline(c['threshold'],color='#333333',linestyle='--',label='Normal q99 threshold')
        axes[1].set_ylabel('P3 score'); axes[1].legend(frameon=False,fontsize=8)
        axes[2].step(frame,label,where='mid',label='Binary ground truth',color='#B23B3B'); axes[2].step(frame,alarm*.8,where='mid',label='3-frame alarm',color='#245B86')
        axes[2].set(xlabel='Target frame index (not wall-clock detection time)',ylabel='GT / alarm',ylim=(-.1,1.2),yticks=[0,1]); axes[2].legend(frameon=False,fontsize=8)
        for a in axes: a.spines[['top','right']].set_visible(False)
        fig.suptitle(f'{NAMES[model]} / {args.device} test 03 / {args.mode} / seed 0',fontsize=13)
        save(fig,args.out/f'{model}_{args.device}_{args.mode}_sequence03','Source: P3/03.csv and normal_fit.json. Scores on common valid frames; wall-clock queue/lookahead delay has not been benchmarked.')


if __name__=='__main__': main()
