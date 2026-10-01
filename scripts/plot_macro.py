"""Plot device heterogeneity and equal-weight macro4 from verified three-seed scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/stage02'))
    parser.add_argument('--mode',choices=['offline','online'],default='offline')
    parser.add_argument('--out',type=Path,default=Path('docs/figures/stage02'))
    args=parser.parse_args()
    device_rows=json.loads((args.root/'device_summary.json').read_text())['results']
    macro_rows=json.loads((args.root/'macro_summary.json').read_text())['results']
    models=sorted({row['backbone'] for row in macro_rows if row['mode']==args.mode and row['variant']=='P3'})
    if not models: raise ValueError('No complete four-device P3 results for this mode')
    names={'dinov3-l':'DINOv3-L','vjepa21-l':'V-JEPA 2.1-L'}
    devices=['R01','R02','R03','R04']; x=np.arange(5); width=.32
    fig,axes=plt.subplots(len(models),2,figsize=(12,4.3*len(models)),squeeze=False,layout='constrained')
    for row_index,model in enumerate(models):
        for column,metric in enumerate(['auroc','ap']):
            axis=axes[row_index,column]
            for offset,variant,color,label in [(-.5,'P0','#718496','P0 global memory'),(.5,'P3','#B8791A','P3 phase + temporal')]:
                selected=[]
                for device in devices:
                    matches=[row for row in device_rows if (row['backbone'],row['mode'],row['device'],row['variant'])==(model,args.mode,device,variant)]
                    if len(matches)!=1: raise ValueError('Missing/duplicate device metric for a complete macro4 condition')
                    selected.append(matches[0])
                matches=[row for row in macro_rows if (row['backbone'],row['mode'],row['variant'])==(model,args.mode,variant)]
                if len(matches)!=1: raise ValueError('Missing/duplicate macro4 metric')
                selected.append(matches[0])
                means=np.array([row[metric+'_mean']*100 for row in selected])
                low=np.array([row[metric+'_ci_low']*100 for row in selected])
                high=np.array([row[metric+'_ci_high']*100 for row in selected])
                positions=x+offset*width
                axis.bar(positions,means,width,color=color,label=label)
                axis.vlines(positions,low,high,color='#252525',linewidth=1.2)
                axis.hlines(low,positions-.045,positions+.045,color='#252525',linewidth=1.2)
                axis.hlines(high,positions-.045,positions+.045,color='#252525',linewidth=1.2)
            axis.axvline(3.5,color='#A6ACB2',linewidth=1,linestyle=':')
            axis.set(xticks=x,xticklabels=devices+['Macro4'],ylim=(0,100),
                     ylabel='AUROC (%)' if metric=='auroc' else 'AP (%)',title=names.get(model,model))
            axis.spines[['top','right']].set_visible(False)
            if row_index==0 and column==0:
                axis.legend(frameon=True,facecolor='white',edgecolor='white',framealpha=1,
                            fontsize=9,loc='upper right')
    fig.suptitle(f'Frozen encoders / {args.mode} / mean of 3 seeds; 95% video-bootstrap CI',fontsize=13)
    fig.text(.015,-.045,'Macro4 gives each device equal weight. Videos resampled independently within devices; conditions paired.\nSource: device_summary.json and macro_summary.json. Scores/frame counts are never pooled across devices.',fontsize=9)
    args.out.mkdir(parents=True,exist_ok=True)
    destination=args.out/f'{args.mode}_macro4'
    for extension in ['.png','.svg']: fig.savefig(destination.with_suffix(extension),dpi=180,bbox_inches='tight',facecolor='white')
    svg=destination.with_suffix('.svg'); svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__': main()
