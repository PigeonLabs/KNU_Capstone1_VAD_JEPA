"""Plot actual strict-loaded encoder parameter counts, not anomaly/runtime performance."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('results/setup')
    rows=[]
    sources={}
    for model in ['dinov3-b','dinov3-l','vjepa21-b','vjepa21-l']:
        path=root/f'{model}_smoke.json'
        info=json.loads(path.read_text())
        if (info['model']!=model or info['status']!='strict_load_and_real_clip_passed'
                or info['trainable_parameters']!=0 or info['parameter_count']<=0):
            raise ValueError('Actual strict frozen encoder validation required')
        rows.append({'model':model,'parameter_count':info['parameter_count']})
        sources[str(path)]=digest(path)
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    blue,gold='#2563a6','#bb7411'
    x=np.array([0,1]);width=.32
    for offset,indices,name,color in [(-width/2,[0,1],'DINOv3',blue),(width/2,[2,3],'V-JEPA 2.1',gold)]:
        values=[rows[i]['parameter_count']/1e6 for i in indices]
        bars=ax.bar(x+offset,values,width,color=color,label=name,zorder=3)
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:.2f}M',(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),
                        textcoords='offset points',ha='center',fontsize=10,fontweight='bold')
    ax.set_xticks(x,['B encoder','L encoder'])
    ax.set_ylabel('Encoder parameters (millions)')
    ax.set_ylim(0,350)
    ax.set_title('Strict-loaded frozen encoder sizes',loc='left',fontsize=15,fontweight='bold',pad=16)
    ax.yaxis.grid(True,color='#dddddd',linewidth=.7,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='upper left')
    fig.text(.13,.035,'Actual loaded encoder counts; phase head excluded. Accuracy, FPS and latency not measured here.',
             ha='left',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.13,right=.98,bottom=.17,top=.84)
    output=Path('docs/figures/setup');output.mkdir(parents=True,exist_ok=True)
    figures={}
    for suffix in ['png','svg']:
        path=output/f'backbone_parameter_counts.{suffix}'
        fig.savefig(path,dpi=180,facecolor='white')
        figures[str(path)]=digest(path)
    plt.close(fig)
    report={'status':'rendered_from_actual_strict_load_reports','counts':rows,'source_sha256':sources,
            'plot_code_sha256':digest(Path(__file__)),'figures':figures,
            'scope':'Actual frozen encoder parameters only; excludes phase heads; no AUROC/AP/FPS/latency or memory-footprint claim'}
    (root/'small_backbone_figure_sources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
