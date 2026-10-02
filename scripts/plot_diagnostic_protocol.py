"""Visualize the fixed held-out intervention inventory and labels, never model results."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from ipad_jepa.diagnostics import scenarios,CLASSES


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=Path('results/stage00/manifest.json')
    rows=[r for r in json.loads(manifest.read_text())['sequences'] if r.get('split')=='diagnostic']
    assert len(rows)==17 and all(r['partition']=='training' for r in rows)
    groups={d:sum(r['device']==d for r in rows) for d in ['R01','R02','R03','R04']}
    assert groups=={'R01':6,'R02':4,'R03':3,'R04':4}
    row=next(r for r in rows if (r['device'],r['sequence'])==('R01','02'))
    cases=scenarios(row)
    counts=[sum(not c.temporal and not c.appearance for c in cases),sum(bool(c.temporal and not c.appearance) for c in cases),
            sum(bool(c.appearance and not c.temporal) for c in cases),sum(bool(c.temporal and c.appearance) for c in cases)]
    assert counts==[1,6,6,36]
    selected=[next(c for c in cases if c.name==name) for name in ['stall_16','local_colour_4','stall_8__occlusion_1']]
    colors=['#999999','#2563a6','#bb7411','#7656a4']
    fig,(left,right)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.6]})
    bars=left.bar(np.arange(4),counts,color=[colors[i] for i in [0,2,1,3]],zorder=3)
    for bar,count in zip(bars,counts):left.text(bar.get_x()+bar.get_width()/2,count+.5,str(count),ha='center',fontweight='bold')
    left.set(xticks=np.arange(4),xticklabels=['Normal','Time\nonly','Appearance\nonly','Mixed'],ylabel='Scenarios per held-out video',ylim=(0,42))
    left.set_title('49 fixed scenarios per video',loc='left',fontweight='bold',pad=12)
    left.yaxis.grid(True,color='#dddddd',zorder=0);left.spines[['top','right']].set_visible(False)
    right.imshow(np.stack([c.labels() for c in selected]),aspect='auto',interpolation='nearest',cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4.5),4),extent=(-.5,row['frames']-.5,2.5,-.5))
    right.set(yticks=np.arange(3),yticklabels=['Stall 16','Colour 4%','Stall 8 + occlusion 1%'],xlabel='Transformed timeline frame',xticks=[0,50,100,150,200])
    right.set_title('Known injection labels / R01 normal video 02',loc='left',fontweight='bold',pad=12)
    right.axvline(np.ceil(.25*row['frames'])-.5,color='#333333',ls='--',lw=.8)
    right.axvline(np.floor(.75*row['frames'])-.5,color='#333333',ls='--',lw=.8)
    right.legend(handles=[Patch(color=c,label=name) for c,name in zip(colors,CLASSES)],loc='upper center',bbox_to_anchor=(.5,-.2),ncols=4,frameon=False,fontsize=9)
    right.spines[['top','right']].set_visible(False)
    fig.suptitle('Controlled interventions on held-out normal videos',x=.06,ha='left',fontweight='bold',fontsize=15)
    fig.text(.06,.045,'17 held-out normal videos: R01=6, R02=4, R03=3, R04=4. Seed 0; injections stay within the middle 50%.\nMixed cases have frame-wise union labels. Existing normal component q99 stays fixed; real fault types and runtime are not assessed here.',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.29,wspace=.7)
    out=Path('docs/figures/stage05');out.mkdir(parents=True,exist_ok=True);figures={}
    for extension in ['png','svg']:
        path=out/f'diagnostic_protocol.{extension}';fig.savefig(path,dpi=180,facecolor='white');figures[str(path)]=digest(path)
    plt.close(fig)
    result={'status':'rendered_fixed_protocol_only','diagnostic_videos_by_device':groups,'source_videos':17,'scenarios_per_video':49,
            'scenario_families':dict(zip(['normal','temporal','appearance','mixed'],counts)),'planned_scenario_videos':17*49,
            'example_source':{'device':row['device'],'sequence':row['sequence'],'frames':row['frames']},'examples':[c.metadata() for c in selected],
            'manifest_sha256':digest(manifest),'intervention_code_sha256':digest('src/ipad_jepa/diagnostics.py'),'plot_code_sha256':digest(__file__),
            'figures':figures,'scope':'Actual held-out inventory and declared injection labels only; no encoder/evidence predictions, confusion, F1, accuracy or runtime completion claim'}
    Path('results/setup/diagnostic_protocol_figure_sources.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))


if __name__=='__main__':main()
