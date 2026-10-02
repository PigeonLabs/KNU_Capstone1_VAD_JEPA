"""Frozen, decoder-free P0–P3 experiments with normal-only fitting/calibration."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import torch
from ipad_jepa.backbones import PhaseHead
from ipad_jepa.alignment import align,POLICY
from ipad_jepa.small_data import sequences,normal_candidates
from ipad_jepa.memory import PrototypeMemory,balanced_counts
from ipad_jepa.scoring import NormalCalibration
from ipad_jepa.temporal import circular_phase,temporal_score,common_mask
from ipad_jepa.torch_memory import TorchMemory

VARIANTS={'P0':{'k':1,'global_search':True},'P1':{'k':1,'global_search':False},
          'P2':{'k':5,'global_search':False},'P3':{'k':5,'global_search':False}}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def predict(sequence,head,batch=256):
    values=[]
    with torch.inference_mode():
        for start in range(0,len(sequence.targets),batch):
            x=torch.from_numpy(np.array(sequence.global_features[start:start+batch],dtype=np.float32)).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=head(x)
            values.append(logits.float().softmax(1).cpu().numpy())
    phase=circular_phase(np.concatenate(values))
    dense=np.full(sequence.row['frames'],np.nan)
    dense[sequence.targets]=phase
    return phase,dense


def temperature_sample(calibration,phases,scorer,seed,limit=50000):
    rng=np.random.default_rng(seed)
    capacities=np.array([s.patches.shape[0]*s.patches.shape[1] for s in calibration])
    counts=balanced_counts(capacities,limit,rng)
    distances=[]
    with torch.inference_mode():
        for s,phi,count,capacity in zip(calibration,phases,counts,capacities):
            ids=rng.choice(int(capacity),int(count),replace=False)
            ci,pi=ids//s.patches.shape[1],ids%s.patches.shape[1]
            # Group by predicted query bin to preserve each sampled patch's phase conditioning.
            for b in range(scorer.bins):
                selected=np.flatnonzero(np.floor(phi[ci]*scorer.bins).astype(int)==b)
                if not len(selected): continue
                x=torch.from_numpy(s.patches[ci[selected],pi[selected]].astype(np.float32))[None].cuda()
                representative=torch.tensor([(b+.5)/scorer.bins],device='cuda')
                distances.append(scorer.nearest(x,representative,k=5)[2][0,:,-1].cpu().numpy())
    return max(float(np.median(np.concatenate(distances))),1e-6),int(counts.sum())


def score_sequence(sequence,phase,dense,scorer,cycle,batch=16):
    raw={v:[] for v in ('P0','P1','P2')}
    with torch.inference_mode():
        for start in range(0,len(phase),batch):
            x=torch.from_numpy(np.array(sequence.patches[start:start+batch],dtype=np.float32)).cuda()
            phi=torch.from_numpy(phase[start:start+batch].astype(np.float32)).cuda()
            for v in raw:
                raw[v].append(scorer.frame_scores(x,phi,**VARIANTS[v]).cpu().numpy())
    time_score=temporal_score(dense,cycle)[sequence.targets]
    scores={v:np.stack([np.concatenate(values),time_score],1) for v,values in raw.items()}
    scores['P3']=scores['P2'].copy()
    return scores


def combined(cal,pairs,variant):
    return cal.combine(pairs) if variant=='P3' else (pairs[:,0]-cal.median[0])/cal.scale[0]


def alarms(scores,threshold,valid,consecutive=3):
    result=np.zeros(len(scores),dtype=bool); streak=0
    for i,(score,keep) in enumerate(zip(scores,valid)):
        streak=streak+1 if keep and score>threshold else 0
        result[i]=streak>=consecutive
    return result


def metrics(labels,scores):
    if len(np.unique(labels))!=2 or not np.all(np.isfinite(scores)):
        raise ValueError('Both labels and finite scores required for AUROC/AP')
    return {'frame_auroc':float(roc_auc_score(labels,scores)),
            'frame_ap':float(average_precision_score(labels,scores)),
            'frames':len(labels),'anomaly_frames':int(labels.sum())}


def run(cache,rows,phase_run,out,local,seed,data_root):
    if out.exists() or local.exists() and (local/'memory.npz').exists():
        raise ValueError('Fresh B evaluation namespace required')
    started=time.perf_counter()
    out.mkdir(parents=True,exist_ok=True)
    # A failed rerun must not leave an old completion marker beside new partial outputs.
    (out/'metrics.json').unlink(missing_ok=True)
    fit=sequences(cache,rows,'fit'); calibration=sequences(cache,rows,'calibration')
    if fit[0].identity!=calibration[0].identity:
        raise ValueError('Fit/calibration cache identity differs')
    head_meta=json.loads((phase_run/'phase_training.json').read_text())
    actual=[s.meta['fingerprint'] for s in fit+calibration]
    if head_meta.get('feature_dimension')!=768 or head_meta['epochs']!=20 or head_meta['status']!='complete' or head_meta['seed']!=seed or sorted(actual)!=sorted(head_meta['cache_fingerprints']):
        raise ValueError('Phase checkpoint does not match normal caches/seed')
    head=PhaseHead(768).cuda().eval()
    head.load_state_dict(torch.load(phase_run/'phase_head.pt',map_location='cpu',weights_only=True),strict=True)
    x,phi,groups,sampling=normal_candidates(fit,seed=seed)
    memory=PrototypeMemory(seed=seed).fit(x,phi,groups)
    del x,phi,groups
    scorer=TorchMemory(memory).cuda().eval()
    cal_phase=[predict(s,head) for s in calibration]
    memory.temperature,n_temperature=temperature_sample(calibration,[p[0] for p in cal_phase],scorer,seed)
    scorer.temperature=memory.temperature
    cycle=float(np.median([s.row['frames'] for s in fit]))
    calibrators={}; cal_pairs={v:[] for v in VARIANTS}; normal_outputs=[]
    for s,(phase,dense) in zip(calibration,cal_phase):
        raw=score_sequence(s,phase,dense,scorer,cycle)
        valid=common_mask(s.row['frames'])[s.targets]
        normal_outputs.append((s,phase,raw,valid))
        for v in VARIANTS:
            pairs=raw[v][valid]
            if not np.all(np.isfinite(pairs)): raise ValueError('Invalid calibration scores')
            cal_pairs[v].append(pairs)
    for v in VARIANTS:
        pairs=np.concatenate(cal_pairs[v])
        calibrators[v]=NormalCalibration().fit(pairs)
        calibrators[v].threshold=float(np.quantile(combined(calibrators[v],pairs,v),.99))
    local.mkdir(parents=True,exist_ok=True)
    np.savez(local/'memory.npz',mean=memory.pca.mean_,components=memory.pca.components_,
             prototypes=memory.prototypes,temperature=memory.temperature,cycle_length=cycle)
    metadata={'status':'normal_fit_and_calibration_complete','seed':seed,'device':fit[0].row['device'],
              'backbone':fit[0].identity['backbone'],'mode':fit[0].identity['mode'],
              'cache_identity':fit[0].identity,'normal_cache_fingerprints':actual,
              'phase_checkpoint_sha256':digest(phase_run/'phase_head.pt'),
              'phase_selected_epoch':head_meta['selected_epoch'],'cycle_length_fit_median':cycle,
              'bins':16,'prototypes_per_bin':128,'total_prototypes':2048,'pca_dimensions':256,
              'feature_dimension':768,'base_experiment_sha256':digest(Path(__file__).with_name('experiment.py')),'pca_samples':50000,'pca_explained_variance_ratio':float(memory.pca.explained_variance_ratio_.sum()),
              'candidate_sampling':sampling,'temperature':memory.temperature,'temperature_samples':n_temperature,
              'calibration_valid_frames':sum(len(x) for x in cal_pairs['P3']),
              'calibration':{v:{'median':c.median.tolist(),'mad_scale':c.scale.tolist(),'threshold':c.threshold,
                                 'component_thresholds':c.component_thresholds.tolist()} for v,c in calibrators.items()},
              'code_sha256':digest(__file__),'memory_code_sha256':digest(Path(__file__).with_name('memory.py')),
              'scope':'Normal-only fit/calibration; held-out test labels used only below for evaluation'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'normal_fit.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (out/'normal_calibration.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['sequence','frame','valid','relative_phase','predicted_phase','feature_raw','time_raw',*VARIANTS])
        for s,phase,raw,valid in normal_outputs:
            combined_scores=[combined(calibrators[v],raw[v],v) for v in VARIANTS]
            writer.writerows(zip([s.row['sequence']]*len(phase),s.targets,valid.astype(int),s.targets/s.row['frames'],phase,
                                 raw['P2'][:,0],raw['P2'][:,1],*combined_scores))
    print(f"normal memory ready: {metadata['backbone']}/{metadata['mode']}/{metadata['device']}/seed{seed}",flush=True)
    # Tests are opened only after every model and calibration parameter is fixed.
    test=sequences(cache,rows,'test')
    if test[0].identity!=fit[0].identity: raise ValueError('Test cache identity differs')
    results={v:{'labels':[],'scores':[]} for v in VARIANTS}
    sensitivity={v:{o:{'labels':[],'scores':[]} for o in [-1,0,1]} for v in VARIANTS}
    alignment_notes=[]
    for s in test:
        row=s.row
        label_path=data_root/row['label_file']
        if digest(label_path)!=row['label_sha256']: raise ValueError('Test labels changed')
        label,known,candidates=align(np.load(label_path,allow_pickle=False),row['frames'])
        phase,dense=predict(s,head)
        raw=score_sequence(s,phase,dense,scorer,cycle)
        inference_valid=common_mask(row['frames'])[s.targets]
        valid=inference_valid&known[s.targets]
        if row['alignment_status']!='matched':
            alignment_notes.append({'sequence':row['sequence'],'unknown_common_frames':int((inference_valid&~known[s.targets]).sum())})
        for v,pairs in raw.items():
            c=calibrators[v]; scores=combined(c,pairs,v)
            if not np.all(np.isfinite(scores[valid])): raise ValueError('Invalid test scores')
            # Ground-truth uncertainty changes metric masks, never inference or alarm state.
            alarm=alarms(scores,c.threshold,inference_valid)
            types=np.zeros(len(scores),dtype=int)
            finite=np.all(np.isfinite(pairs),axis=1); types[finite]=c.types(pairs[finite])
            folder=out/v; folder.mkdir(exist_ok=True)
            with (folder/f"{row['sequence']}.csv").open('w',newline='') as stream:
                fields=['frame','label','valid','inference_valid','phase','feature_raw','time_raw','score','alarm','evidence_type']
                writer=csv.writer(stream,lineterminator='\n'); writer.writerow(fields)
                writer.writerows(zip(s.targets,label[s.targets],valid.astype(int),inference_valid.astype(int),phase,pairs[:,0],pairs[:,1],scores,alarm.astype(int),types))
            results[v]['labels'].append(label[s.targets][valid]); results[v]['scores'].append(scores[valid])
            for offset in [-1,0,1]:
                gt=candidates.get(offset,candidates[0])[s.targets]
                keep=inference_valid&(gt>=0)
                sensitivity[v][offset]['labels'].append(gt[keep]); sensitivity[v][offset]['scores'].append(scores[keep])
        print(f"scored: {row['device']}/{row['sequence']}",flush=True)
    summary={'status':'complete_device_evaluation','device':metadata['device'],'backbone':metadata['backbone'],
             'mode':metadata['mode'],'seed':seed,'test_videos':len(test),'valid_mask':'t=19..N-8 inclusive',
             'variants':{v:metrics(np.concatenate(r['labels']),np.concatenate(r['scores'])) for v,r in results.items()},
             'alignment_policy':POLICY,'unresolved_alignment_sequences':alignment_notes,
             'alignment_sensitivity':{v:{str(o):metrics(np.concatenate(r['labels']),np.concatenate(r['scores'])) for o,r in values.items()} for v,values in sensitivity.items()} if alignment_notes else {},
             'seconds':time.perf_counter()-started,'note':'Device frame metrics only; not macro4 or throughput benchmark'}
    temporary=out/'metrics.json.tmp'
    temporary.write_text(json.dumps(summary,indent=2)+'\n')
    temporary.replace(out/'metrics.json')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--phase-run',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    parser.add_argument('--device',choices=['R01','R02','R03','R04'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--local',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[r for r in json.loads(args.manifest.read_text())['sequences'] if r['device']==args.device]
    run(args.cache,rows,args.phase_run,args.out,args.local,args.seed,args.data_root)


if __name__=='__main__': main()
