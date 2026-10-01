"""Three-seed device metrics and paired video bootstrap; no frame-independent CI."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from ipad_jepa.bootstrap_metrics import VideoBootstrapMetric


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


def bootstrap_draws(runs,count=1000,seed=2026):
    """Keep draw indices, including degenerate draws, for paired macro comparisons."""
    keys=list(runs[0]); rng=np.random.default_rng(seed); values=[]
    prepared=[VideoBootstrapMetric(run) for run in runs]
    for _ in range(count):
        selected=rng.choice(keys,len(keys),replace=True)
        value=np.mean([metric(selected) for metric in prepared],axis=0)
        values.append(value)
    return np.array(values)


def bootstrap(runs,count=1000):
    values=bootstrap_draws(runs,count)
    valid=np.all(np.isfinite(values),axis=1); rejected=int((~valid).sum())
    values=values[valid]
    if len(values)<count*.95: raise ValueError('Too many degenerate video resamples')
    return values,rejected


def macro_four_devices(stored,count=1000):
    """Equal device weights; independent video draws per device, paired across conditions."""
    devices=['R01','R02','R03','R04']; conditions={}; references={}
    for (model,mode,device,variant),(runs,_) in stored.items():
        if device not in devices: raise ValueError('Unexpected device in macro4 input')
        if len(runs)!=3: raise ValueError('Macro4 requires three seeds')
        for run in runs:
            if device in references: check_pair(references[device],run)
            else: references[device]=run
        conditions.setdefault((model,mode,variant),{})[device]=runs
    complete={key:value for key,value in conditions.items() if set(value)==set(devices)}
    excluded=[{'backbone':key[0],'mode':key[1],'variant':key[2],
               'missing_devices':[device for device in devices if device not in value]}
              for key,value in sorted(conditions.items()) if key not in complete]
    points={}; draws={}; rows=[]; deltas=[]
    for key,by_device in sorted(complete.items()):
        points[key]=np.mean([np.mean([statistic(run,list(run)) for run in by_device[device]],axis=0)
                             for device in devices],axis=0)
        # The same device-specific stream is reused across backbones, modes and variants.
        # Different devices have independent streams; filtering waits until all strata align.
        draws[key]=np.mean([bootstrap_draws(by_device[device],count,
                            np.random.SeedSequence([2026,int(device[1:])]))
                            for device in devices],axis=0)
    valid=np.ones(count,dtype=bool)
    for values in draws.values(): valid &= np.all(np.isfinite(values),axis=1)
    rejected=int((~valid).sum())
    if draws and int(valid.sum())<count*.95: raise ValueError('Too many degenerate macro4 video resamples')
    for key,by_device in sorted(complete.items()):
        draws[key]=draws[key][valid]
        low,high=np.quantile(draws[key],[.025,.975],axis=0)
        counts={device:{'test_videos':len(by_device[device][0]),
                        'frames':sum(len(value[1]) for value in by_device[device][0].values()),
                        'anomaly_frames':int(sum(value[1].sum() for value in by_device[device][0].values()))}
                for device in devices}
        rows.append({'backbone':key[0],'mode':key[1],'variant':key[2],'devices':4,'seeds':3,
                     'auroc_mean':float(points[key][0]),'auroc_ci_low':float(low[0]),'auroc_ci_high':float(high[0]),
                     'ap_mean':float(points[key][1]),'ap_ci_low':float(low[1]),'ap_ci_high':float(high[1]),
                     'bootstrap_draws':count,'bootstrap_rejected':rejected,'device_counts':counts})
    for key in sorted(complete):
        model,mode,variant=key; pairs=[]
        if variant=='P0': pairs.append(((model,mode,'P3'),model,mode,'P3_minus_P0'))
        if variant=='B0': pairs.append(((model,mode,'B1'),model,mode,'B1_minus_B0'))
        if model=='dinov3-l': pairs.append((('vjepa21-l',mode,variant),'vjepa21-l_minus_dinov3-l',mode,variant+'_backbone'))
        if mode=='offline': pairs.append(((model,'online',variant),model,'online_minus_offline',variant+'_mode'))
        for target,backbone,result_mode,comparison in pairs:
            if target not in complete: continue
            low,high=np.quantile(draws[target]-draws[key],[.025,.975],axis=0)
            point=points[target]-points[key]
            deltas.append({'backbone':backbone,'mode':result_mode,'comparison':comparison,
                           'auroc_delta':float(point[0]),'auroc_delta_ci_low':float(low[0]),'auroc_delta_ci_high':float(high[0]),
                           'ap_delta':float(point[1]),'ap_delta_ci_low':float(low[1]),'ap_delta_ci_high':float(high[1])})
    return {'scope':'Equal-weight mean of R01/R02/R03/R04 device metrics, each a mean of three seed metrics. No pooling of frames or seed scores.',
            'bootstrap':f'{count} video resamples independently within each device; device-specific SeedSequence([2026, device_number]); shared draws across seeds and paired conditions; percentile 95% CI.',
            'results':rows,'paired_deltas':deltas,'incomplete_conditions':excluded}


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
    macro=macro_four_devices(stored,count)
    (root/'macro_summary.json').write_text(json.dumps(macro,indent=2)+'\n')
    if macro['results']:
        fields=[key for key in macro['results'][0] if key!='device_counts']
        with (root/'macro_summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
            writer.writeheader(); writer.writerows(macro['results'])
    elif (root/'macro_summary.csv').exists():
        (root/'macro_summary.csv').unlink()
    output={'scope':'Device results; macro4 requires all four devices. Mean of three seed metrics.',
            'bootstrap':f'{count} paired whole-video resamples, seed=2026; percentile 95% CI. Frames within a video stay together.',
            'results':rows,'paired_deltas':deltas,'macro_summary':'macro_summary.json'}
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
