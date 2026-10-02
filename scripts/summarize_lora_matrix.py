"""Audit completed LoRA seeds and report only exact three-seed frozen/LoRA groups."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from ipad_jepa.lora_audit import check_training,check_bank,ready_group
from summarize_clip_ablation import audit_condition as audit_p3
from summarize_experiments import load_run,check_pair,statistic,bootstrap_draws,macro_four_devices
from summarize_neighbour_ablation import digest,read,write,difference

MODELS=('dinov3-l','vjepa21-l')
MODES=('offline','online')
DEVICES=('R01','R02','R03','R04')
SEEDS=(0,1,2)
VARIANTS=('P0','P1','P2','P3')


def audit_lora(folder, training, cache, local, frozen, frozen_local, manifest, data_root, condition, weight, out):
    # Existing tensor-level provenance verification runs on CPU and loads the
    # exact selected adapter/joint head. It is reused without changing active code.
    from plot_lora_evaluation import calibration_check,provenance
    info=json.loads((folder/'metrics.json').read_text());meta=json.loads((training/'lora_training.json').read_text())
    check_training(meta,read(training/'lora_training.csv'),condition,weight)
    if tuple(info[k] for k in ['backbone','mode','device','seed'])!=condition or set(info['variants'])!=set(VARIANTS):
        raise ValueError('Complete LoRA metric condition/variant inventory differs')
    normal,sources=calibration_check(folder,info,manifest,data_root)
    args=SimpleNamespace(training=training,cache=cache,local=local,frozen_results=frozen,
                         frozen_local=frozen_local,results=folder)
    tensor_proof=provenance(args,info,normal,manifest)
    previous=folder/'provenance_check.json'
    if previous.is_file():
        prior=json.loads(previous.read_text())
        for name in ['selected_adapter_sha256','joint_head_sha256','rebuilt_memory_sha256','frozen_memory_sha256','phase_metadata_sha256']:
            if prior[name]!=tensor_proof[name]:raise ValueError('Previously published selected LoRA source changed')
    with np.load(local/'memory.npz',allow_pickle=False) as bank:check_bank(bank,normal)
    model,mode,device,seed=condition
    _,p3_check=audit_p3(folder,manifest,data_root,model,mode,device,seed,16)
    runs={v:load_run(folder,v,info) for v in VARIANTS}
    for run in runs.values():check_pair(runs['P3'],run)
    for name in ['lora_training.json','lora_training.csv']:sources[name]=digest(folder/name)
    proof={'status':'passed_completed_lora_condition','backbone':model,'mode':mode,'device':device,'seed':seed,
           'teacher_weight':weight,'normal_thresholds_checked':4,'tensor_provenance':tensor_proof,'P3_replay':p3_check,
           'public_sources_sha256':sources,'training_sha256':digest(training/'lora_training.json'),
           'training_curve_sha256':digest(training/'lora_training.csv'),'selected_adapter_sha256':tensor_proof['selected_adapter_sha256'],
           'bank_sha256':digest(local/'memory.npz'),'phase_metadata_sha256':digest(local/'phase_training.json'),
           'auditor_sha256':digest(__file__),'protocol_audit_sha256':digest('src/ipad_jepa/lora_audit.py'),
           'selected_tensor_verifier_sha256':digest(Path(__file__).with_name('plot_lora_evaluation.py')),
           'limits':'Exact selected adapter/joint head equality, actual adapted cache metadata/targets/shapes, new PCA/bank geometry, P2/P3 normal MAD/component q99, all variant q99/test GT/time/score/alarm/metrics, P3 finite phase/raw/calibration replay. Does not re-encode pixels, retrain PCA/k-center, or replay GPU feature distances/temperature samples. P0/P1 calibration CSV stores normalized values, so their raw median/MAD are not independently rebuilt.'}
    target=out/model/mode/device/f'seed{seed}'/'condition_audit.json';target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(proof,indent=2)+'\n');temp.replace(target)
    return runs,proof,target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('root','results/stage04'),('training','artifacts/lora'),('cache','artifacts/features_lora'),
                         ('local','artifacts/runs_lora'),('source','results/stage02'),('source-local','artifacts/runs'),
                         ('out','results/stage04/matrix_audit'),('manifest','results/stage00/manifest.json'),
                         ('data-root','../IPAD_dataset/IPAD_dataset')]:
        p.add_argument('--'+name,type=Path,default=Path(default))
    p.add_argument('--teacher-weight',type=float,choices=[0.,1.],default=1.)
    p.add_argument('--require-full',action='store_true')
    args=p.parse_args();manifest=json.loads(args.manifest.read_text())['sequences']
    required=[(m,mode,d,s) for m in MODELS for mode in MODES for d in DEVICES for s in SEEDS]
    completed=[]
    for key in required:
        model,mode,device,seed=key;folder=args.root/model/mode/device/f'seed{seed}'
        if not (folder/'metrics.json').is_file() or not (folder/'lora_training.json').is_file():continue
        info=json.loads((folder/'metrics.json').read_text());meta=json.loads((folder/'lora_training.json').read_text())
        if info.get('status')=='complete_device_evaluation' and meta.get('status')=='complete_training':completed.append(key)
    if not completed or args.require_full and len(completed)!=48:raise ValueError('Required completed LoRA seed conditions missing')
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'validation.json').unlink(missing_ok=True)
    runs_by_condition={};checks=[]
    for key in completed:
        model,mode,device,seed=key;condition=Path(model)/mode/device/f'seed{seed}'
        paired=args.source/condition
        # The all-variant frozen audit plus explicit P3 replay also checks the
        # actual reference GT and normal q99 independently of the LoRA treatment.
        from plot_lora_evaluation import calibration_check
        frozen_info=json.loads((paired/'metrics.json').read_text())
        calibration_check(paired,frozen_info,manifest,args.data_root)
        audit_p3(paired,manifest,args.data_root,model,mode,device,seed,16)
        fresh,proof,target=audit_lora(args.root/condition,args.training/condition,args.cache/condition,args.local/condition,
                                      paired,args.source_local/condition,manifest,args.data_root,key,args.teacher_weight,args.out)
        frozen={v:load_run(paired,v,frozen_info) for v in VARIANTS}
        for variant in VARIANTS:check_pair(frozen[variant],fresh[variant])
        runs_by_condition[key]={'F':frozen,'L':fresh}
        checks.append({'condition':list(key),'teacher_weight':args.teacher_weight,'condition_audit':str(target.relative_to(args.out)),
                       'condition_audit_sha256':digest(target),'normal_thresholds_checked':8,'selected_adapter_sha256':proof['selected_adapter_sha256']})
        print(f'LoRA seed independently audited: {model}/{mode}/{device}/seed{seed} teacher={args.teacher_weight:g}',flush=True)
    groups=[(m,mode,d) for m in MODELS for mode in MODES for d in DEVICES if ready_group(runs_by_condition,m,mode,d)]
    stored,points,draws,rows,deltas={},{},{},[],[]
    for model,mode,device in groups:
        for kind in ('F','L'):
            for variant in VARIANTS:
                runs=[runs_by_condition[model,mode,device,s][kind][variant] for s in SEEDS]
                for run in runs[1:]:check_pair(runs[0],run)
                values=bootstrap_draws(runs);valid=np.isfinite(values).all(axis=1)
                if valid.sum()<950:raise ValueError('Too many degenerate video resamples')
                point=np.mean([statistic(run,list(run)) for run in runs],axis=0);lo,hi=np.quantile(values[valid],[.025,.975],axis=0)
                name=kind+'_'+variant;key=(model,mode,device,name)
                stored[key]=(runs,values[valid]);points[key]=point;draws[key]=values
                labels=np.concatenate([v[1] for v in runs[0].values()])
                rows.append({'backbone':model,'mode':mode,'device':device,'variant':name,'adaptation':'frozen' if kind=='F' else 'lora',
                             'teacher_weight':None if kind=='F' else args.teacher_weight,'score_variant':variant,'seeds':3,
                             'test_videos':len(runs[0]),'frames':len(labels),'anomaly_frames':int(labels.sum()),
                             'auroc_mean':float(point[0]),'auroc_ci_low':float(lo[0]),'auroc_ci_high':float(hi[0]),
                             'ap_mean':float(point[1]),'ap_ci_low':float(lo[1]),'ap_ci_high':float(hi[1]),
                             'bootstrap_draws':1000,'bootstrap_rejected':int((~valid).sum())})
        for variant in VARIANTS:
            a,b=(model,mode,device,'F_'+variant),(model,mode,device,'L_'+variant)
            delta=draws[b]-draws[a];keep=np.isfinite(delta).all(axis=1)
            if keep.sum()<950:raise ValueError('Too many degenerate paired draws')
            deltas.append(difference(model,mode,'LoRA_minus_frozen_'+variant,points[b]-points[a],delta[keep],device))
        a,b=(model,mode,device,'L_P0'),(model,mode,device,'L_P3')
        delta=draws[b]-draws[a];keep=np.isfinite(delta).all(axis=1)
        deltas.append(difference(model,mode,'LoRA_P3_minus_P0',points[b]-points[a],delta[keep],device))
    macro=macro_four_devices(stored)
    for model in MODELS:
        for mode in MODES:
            if not all((model,mode,d) in groups for d in DEVICES):continue
            comparisons=[('F_'+v,'L_'+v,'LoRA_minus_frozen_'+v) for v in VARIANTS]
            comparisons.append(('L_P0','L_P3','LoRA_P3_minus_P0'))
            for old_name,new_name,comparison in comparisons:
                pair=[(model,mode,d,old_name) for d in DEVICES];new=[(model,mode,d,new_name) for d in DEVICES]
                delta=np.mean([bootstrap_draws(stored[b][0],seed=np.random.SeedSequence([2026,int(b[2][1:])]))-
                               bootstrap_draws(stored[a][0],seed=np.random.SeedSequence([2026,int(a[2][1:])])) for a,b in zip(pair,new)],axis=0)
                keep=np.isfinite(delta).all(axis=1)
                if keep.sum()<950:raise ValueError('Too many degenerate Macro4 paired draws')
                point=np.mean([points[b]-points[a] for a,b in zip(pair,new)],axis=0)
                macro['paired_deltas'].append(difference(model,mode,comparison,point,delta[keep]))
    for key in stored:
        model,mode,device,variant=key
        targets=[]
        if model=='dinov3-l':targets.append((('vjepa21-l',mode,device,variant),'vjepa21-l_minus_dinov3-l',mode,variant+'_backbone'))
        if mode=='offline':targets.append(((model,'online',device,variant),model,'online_minus_offline',variant+'_mode'))
        for target,label,result_mode,comparison in targets:
            if target not in stored:continue
            for a,b in zip(stored[key][0],stored[target][0]):check_pair(a,b)
            delta=draws[target]-draws[key];keep=np.isfinite(delta).all(axis=1)
            if keep.sum()<950:raise ValueError('Too many degenerate paired backbone/mode draws')
            deltas.append(difference(label,result_mode,comparison,points[target]-points[key],delta[keep],device))
    scope='Completed LoRA seeds independently audited; accuracy rows require exactly seeds0/1/2; matched frozen targets/GT; per-seed metrics averaged; original-video paired bootstrap1000; Macro4 only with all four equally weighted devices; no runtime claim'
    if not rows:(args.out/'device_summary.csv').unlink(missing_ok=True)
    if not macro['results']:(args.out/'macro_summary.csv').unlink(missing_ok=True)
    write(args.out/'device_summary',{'scope':scope,'results':rows,'paired_deltas':deltas});write(args.out/'macro_summary',macro)
    validation={'status':'passed','matrix_complete':len(completed)==48,'completed_seed_conditions':len(completed),'complete_three_seed_groups':len(groups),
                'teacher_weight':args.teacher_weight,'normal_thresholds_checked':len(completed)*8,'condition_checks':checks,
                'pending_conditions':[list(k) for k in required if k not in completed],
                'device_summary_sha256':digest(args.out/'device_summary.json'),'macro_summary_sha256':digest(args.out/'macro_summary.json'),
                'verifier_sha256':digest(__file__),'protocol_audit_sha256':digest('src/ipad_jepa/lora_audit.py'),
                'selected_tensor_verifier_sha256':digest(Path(__file__).with_name('plot_lora_evaluation.py')),
                'shared_trace_audit_sha256':digest(Path(__file__).with_name('summarize_clip_ablation.py')),
                'shared_bootstrap_sha256':digest(Path(__file__).with_name('summarize_experiments.py')),
                'manifest_sha256':digest(args.manifest),'scope':scope}
    (args.out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k not in ['condition_checks','pending_conditions']},indent=2),flush=True)

if __name__=='__main__':main()
