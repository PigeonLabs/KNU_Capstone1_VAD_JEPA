"""Three-seed device metrics and paired video bootstrap; no frame-independent CI."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


def load_run(path,variant,expected=None):
    videos={}
    for source in sorted((path/variant).glob('*.csv'),key=lambda p:int(p.stem)):
        with source.open() as stream: rows=[r for r in csv.DictReader(stream) if r['valid']=='1']
        frames=np.array([int(r['frame']) for r in rows])
        labels=np.array([float(r['label']) for r in rows])
        if not np.all(np.isin(labels,[0,1])): raise ValueError('Expected binary evaluated labels')
        labels=labels.astype(np.int8)
        scores=np.array([float(r['score']) for r in rows])
        if not np.all(np.isfinite(scores)): raise ValueError('Invalid scores')
        videos[source.stem]=(frames,labels,scores)
    if expected is not None:
        labels=np.concatenate([v[1] for v in videos.values()])
        m=expected['variants'][variant]
        if len(videos)!=expected['test_videos'] or len(labels)!=m['frames'] or int(labels.sum())!=m['anomaly_frames']:
            raise ValueError('Score CSVs do not match completed metric inventory')
        value=statistic(videos,list(videos))
        np.testing.assert_allclose(value,[m['frame_auroc'],m['frame_ap']],atol=1e-9,rtol=0)
    return videos


def check_pair(a,b):
    if list(a)!=list(b): raise ValueError('Paired video IDs differ')
    for key in a:
        if not np.array_equal(a[key][0],b[key][0]) or not np.array_equal(a[key][1],b[key][1]):
            raise ValueError('Paired valid frame/label mismatch')


def statistic(run,selected):
    labels=np.concatenate([run[k][1] for k in selected]); scores=np.concatenate([run[k][2] for k in selected])
    if len(np.unique(labels))<2: return np.array([np.nan,np.nan])
    return np.array([roc_auc_score(labels,scores),average_precision_score(labels,scores)])


def bootstrap(runs,count=1000):
    keys=list(runs[0]); rng=np.random.default_rng(2026); values=[]; rejected=0
    for _ in range(count):
        selected=rng.choice(keys,len(keys),replace=True)
        value=np.mean([statistic(run,selected) for run in runs],axis=0)
        if np.all(np.isfinite(value)): values.append(value)
        else: rejected+=1
    values=np.array(values)
    if len(values)<count*.95: raise ValueError('Too many degenerate video resamples')
    return values,rejected


def summarize(root,count=1000):
    groups={}
    for source in sorted(root.glob('*/*/*/seed*/metrics.json')):
        meta=json.loads(source.read_text())
        if meta['status']!='complete_device_evaluation': continue
        key=(meta['backbone'],meta['mode'],meta['device'])
        groups.setdefault(key,{})[meta['seed']]=(source.parent,meta)
    rows=[]; stored={}; deltas=[]
    for key,seeds in groups.items():
        if set(seeds)!={0,1,2}: continue
        variants=list(seeds[0][1]['variants'])
        if any(list(seeds[s][1]['variants'])!=variants for s in [1,2]):
            raise ValueError('Seed variant inventories differ')
        for variant in variants:
            runs=[load_run(seeds[s][0],variant,seeds[s][1]) for s in [0,1,2]]
            for r in runs[1:]: check_pair(runs[0],r)
            point=np.mean([statistic(r,list(r)) for r in runs],axis=0)
            values,rejected=bootstrap(runs,count)
            low,high=np.quantile(values,[.025,.975],axis=0)
            labels=np.concatenate([v[1] for v in runs[0].values()])
            row={'backbone':key[0],'mode':key[1],'device':key[2],'variant':variant,
                 'seeds':3,'test_videos':len(runs[0]),'frames':len(labels),'anomaly_frames':int(labels.sum()),
                 'auroc_mean':float(point[0]),'auroc_ci_low':float(low[0]),'auroc_ci_high':float(high[0]),
                 'ap_mean':float(point[1]),'ap_ci_low':float(low[1]),'ap_ci_high':float(high[1]),
                 'bootstrap_draws':count,'bootstrap_rejected':rejected}
            rows.append(row); stored[(*key,variant)]=(runs,values)
        pair=('P0','P3') if 'P0' in variants and 'P3' in variants else ('B0','B1') if 'B0' in variants and 'B1' in variants else None
        for a,b,label in ([] if pair is None else [(stored[(*key,pair[0])],stored[(*key,pair[1])],pair[1]+'_minus_'+pair[0])]):
            for ra,rb in zip(a[0],b[0]): check_pair(ra,rb)
            # Same RNG seed and video order make draws paired across variants.
            delta=b[1]-a[1]; low,high=np.quantile(delta,[.025,.975],axis=0)
            point=np.mean([statistic(rb,list(rb))-statistic(ra,list(ra)) for ra,rb in zip(a[0],b[0])],axis=0)
            deltas.append({'backbone':key[0],'mode':key[1],'device':key[2],'comparison':label,
                           'auroc_delta':float(point[0]),'auroc_delta_ci_low':float(low[0]),'auroc_delta_ci_high':float(high[0]),
                           'ap_delta':float(point[1]),'ap_delta_ci_low':float(low[1]),'ap_delta_ci_high':float(high[1])})
    # Backbone and online/offline differences use identical video draws and matched frames.
    for key,a in stored.items():
        model,mode,device,variant=key
        if model=='dinov3-l' and ('vjepa21-l',mode,device,variant) in stored:
            b=stored[('vjepa21-l',mode,device,variant)]
            for ra,rb in zip(a[0],b[0]): check_pair(ra,rb)
            delta=b[1]-a[1]; low,high=np.quantile(delta,[.025,.975],axis=0)
            point=np.mean([statistic(rb,list(rb))-statistic(ra,list(ra)) for ra,rb in zip(a[0],b[0])],axis=0)
            deltas.append({'backbone':'vjepa21-l_minus_dinov3-l','mode':mode,'device':device,'comparison':variant+'_backbone',
                           'auroc_delta':float(point[0]),'auroc_delta_ci_low':float(low[0]),'auroc_delta_ci_high':float(high[0]),
                           'ap_delta':float(point[1]),'ap_delta_ci_low':float(low[1]),'ap_delta_ci_high':float(high[1])})
        if mode=='offline' and (model,'online',device,variant) in stored:
            b=stored[(model,'online',device,variant)]
            for ra,rb in zip(a[0],b[0]): check_pair(ra,rb)
            delta=b[1]-a[1]; low,high=np.quantile(delta,[.025,.975],axis=0)
            point=np.mean([statistic(rb,list(rb))-statistic(ra,list(ra)) for ra,rb in zip(a[0],b[0])],axis=0)
            deltas.append({'backbone':model,'mode':'online_minus_offline','device':device,'comparison':variant+'_mode',
                           'auroc_delta':float(point[0]),'auroc_delta_ci_low':float(low[0]),'auroc_delta_ci_high':float(high[0]),
                           'ap_delta':float(point[1]),'ap_delta_ci_low':float(low[1]),'ap_delta_ci_high':float(high[1])})
    if not rows: raise ValueError('No complete three-seed device experiments')
    output={'scope':'Device results; macro4 requires all four devices. Mean of three seed metrics.',
            'bootstrap':'1000 paired whole-video resamples, seed=2026; percentile 95% CI. Frames within a video stay together.',
            'results':rows,'paired_deltas':deltas}
    (root/'device_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    with (root/'device_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/stage02'))
    args=parser.parse_args()
    output=summarize(args.root)
    print(json.dumps({'device_variant_groups':len(output['results']),'scope':output['scope']},indent=2))


if __name__=='__main__': main()
