"""Paced, no-drop FIFO JPEG replay with measured GPU service and arrival-to-alarm time."""
import argparse
from collections import deque
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import torch
from ipad_jepa.alignment import align
from ipad_jepa.audit import inspect_frames
from ipad_jepa.backbones import Backbone,PhaseHead
from ipad_jepa.adaptation import install_adapters,load_adapter_state,adapter_training_mode
from ipad_jepa.features import ClipDataset,file_hash
from ipad_jepa.experiment import metrics
from ipad_jepa.runtime_state import StreamingScore,event_metrics
from ipad_jepa.runtime_adaptation import selected_runtime_adapter
from ipad_jepa.streaming import DinoFrameRing
from ipad_jepa.temporal import circular_phase,common_mask
from ipad_jepa.torch_memory import TorchMemory


def require_isolated_gpu():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    others={int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}-{os.getpid()}
    if others: raise RuntimeError(f'Runtime measurement requires GPU isolation; live compute PIDs: {sorted(others)}')


def precision_context(precision):
    return torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()


def fixed_components(args):
    meta=json.loads((args.results/'normal_fit.json').read_text())
    phase_meta=json.loads((args.run/'phase_training.json').read_text())
    if (meta['status']!='normal_fit_and_calibration_complete' or
        (meta['backbone'],meta['mode'],meta['device'],meta['seed'])!=(args.model,args.mode,args.device,args.seed) or
        phase_meta['status']!='complete' or phase_meta['epochs']!=20 or phase_meta['seed']!=args.seed or
        tuple(phase_meta.get(key) for key in ['backbone','mode','device'])!=(args.model,args.mode,args.device) or
        meta.get('phase_selected_epoch')!=phase_meta.get('selected_epoch') or
        sorted(meta['normal_cache_fingerprints'])!=sorted(phase_meta['cache_fingerprints']) or
        file_hash(args.run/'phase_head.pt')!=meta['phase_checkpoint_sha256']):
        raise ValueError('Fixed normal fit/head does not match completed primary condition')
    identity=meta['cache_identity']; source=Path(__file__).resolve().parents[1]/'src/ipad_jepa'
    commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    expected={'backbone':args.model,'mode':args.mode,'weights_sha256':file_hash(args.weights),
              'upstream_commit':commit,'adapter_sha256':file_hash(source/'backbones.py'),
              'reader_sha256':file_hash(source/'features.py'),'image_size':384,'clip_frames':16,'fit_stride':4,
              'preprocessing':'RGB full-frame PIL bilinear resize; ImageNet mean/std',
              'feature_dtype':'float16 from BF16 inference'}
    phase_state=torch.load(args.run/'phase_head.pt',map_location='cpu',weights_only=True)
    adapter,adaptation=selected_runtime_adapter(args,identity,phase_meta,phase_state,expected)
    with np.load(args.run/'memory.npz',allow_pickle=False) as data:
        # Keep the fitted PCA strides; changing layout can change FP32 neighbour tie rounding.
        mean,components,prototypes=(data[k].copy(order='K') for k in ['mean','components','prototypes'])
        temperature,cycle=float(data['temperature']),float(data['cycle_length'])
    if (mean.shape!=(1024,) or components.shape!=(256,1024) or prototypes.shape!=(16,128,256) or
        not all(np.isfinite(value).all() for value in [mean,components,prototypes]) or
        temperature!=meta['temperature'] or cycle!=meta['cycle_length_fit_median']):
        raise ValueError('Invalid fixed PCA/prototype bank')
    memory=SimpleNamespace(bins=16,dimensions=256,temperature=temperature,
        pca=SimpleNamespace(mean_=mean,components_=components),prototypes=prototypes)
    scorer=TorchMemory(memory).cuda().eval()
    head=PhaseHead().cuda().eval()
    head.load_state_dict(phase_state,strict=True)
    model=Backbone(args.model,args.upstream,args.weights,args.mode).cuda().eval()
    if adapter is not None:
        install_adapters(model)
        load_adapter_state(model,adapter)
        adapter_training_mode(model,False)
    return model,head,scorer,meta,cycle,adaptation


def score_features(local,global_features,head,scorer,precision):
    # BF16 primary accuracy caches round to FP16 before FP32 head/search input.
    # FP32 full/reuse comparisons retain FP32 and are an explicitly separate condition.
    start=time.perf_counter()
    if precision=='bf16':
        local,global_features=local.to(torch.float16).float(),global_features.to(torch.float16).float()
    torch.cuda.synchronize(); cast_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    with precision_context(precision): logits=head(global_features.float())
    phase=float(circular_phase(logits.float().softmax(1).cpu().numpy())[0])
    torch.cuda.synchronize(); head_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    feature=float(scorer.frame_scores(local.float(),torch.tensor([phase],device='cuda'),k=5).item())
    torch.cuda.synchronize(); search_ms=(time.perf_counter()-start)*1000
    return phase,feature,cast_ms,head_ms,search_ms


def replay(args,row,model,head,scorer,calibration,cycle):
    folder=args.data_root/row['relative_directory']
    if inspect_frames(folder)['frames_content_sha256']!=row['frames_content_sha256']:
        raise ValueError('Runtime input content changed')
    reader=ClipDataset(folder,np.array([],dtype=np.int64),args.mode,cache_frames=16)
    if len(reader.paths)!=row['frames']: raise ValueError('Runtime frame inventory changed')
    state=StreamingScore(calibration,cycle,first_target=15 if args.mode=='online' else 8)
    images=deque(maxlen=16)
    gpu_images=deque(maxlen=16) if args.implementation=='buffer' else None
    ring=DinoFrameRing(model.encoder) if args.implementation=='reuse' else None
    trace=[]; origin=time.perf_counter()
    with torch.inference_mode():
        for index,_ in enumerate(reader.paths):
            arrival=index/args.arrival_fps
            wait=origin+arrival-time.perf_counter()
            if wait>0: time.sleep(wait)
            service_start=time.perf_counter()-origin
            start=time.perf_counter(); frame=reader.frame(index)
            decode_ms=(time.perf_counter()-start)*1000
            images.append(frame); start=time.perf_counter()
            local=global_features=None
            with precision_context(args.precision):
                if ring is not None:
                    ring.push(frame.cuda(),index)
                    if index>=15: local,global_features=ring.features()
                elif gpu_images is not None:
                    gpu_images.append(frame.cuda())
                    if index>=15:
                        local,global_features=model(torch.stack(list(gpu_images),dim=1)[None])
                elif index>=15:
                    clip=torch.stack(list(images),dim=1)[None].cuda()
                    local,global_features=model(clip)
            torch.cuda.synchronize(); encoder_ms=(time.perf_counter()-start)*1000
            target=None; values={}; cast_ms=head_ms=search_ms=score_ms=0.
            if local is not None:
                target=index if args.mode=='online' else index-7
                phase,feature,cast_ms,head_ms,search_ms=score_features(local,global_features,head,scorer,args.precision)
                start=time.perf_counter(); values=state.push(target,phase,feature)
                score_ms=(time.perf_counter()-start)*1000
            emitted=time.perf_counter()-origin
            trace.append({'arrival_frame':index,'arrival_seconds':arrival,'target_frame':target,
                'service_start_seconds':service_start,'queue_wait_ms':max(0,service_start-arrival)*1000,
                'lookahead_wait_ms':7000/args.arrival_fps if target is not None and args.mode=='offline' else 0,
                'decode_resize_ms':decode_ms,'encoder_ms':encoder_ms,'phase_head_ms':head_ms,
                'feature_cast_ms':cast_ms,
                'projection_search_ms':search_ms,'score_alarm_ms':score_ms,
                'service_ms':(emitted-service_start)*1000,'emitted_seconds':emitted,
                'target_latency_ms':(emitted-target/args.arrival_fps)*1000 if target is not None else None,
                **{key:values.get(key) for key in ['phase','feature_raw','time_raw','score','inference_valid','alarm']}})
    elapsed=time.perf_counter()-origin
    # GT/EOF affects post-run reporting, never the streaming score or alarm state.
    label_path=args.data_root/row['label_file']
    if file_hash(label_path)!=row['label_sha256']: raise ValueError('Runtime labels changed')
    labels,known,_=align(np.load(label_path,allow_pickle=False),row['frames'])
    eligible=[item for item in trace if item['inference_valid']]
    frames=np.array([item['target_frame'] for item in eligible],dtype=int)
    alarms=np.array([item['alarm'] for item in eligible],dtype=bool)
    emissions=np.array([item['emitted_seconds'] for item in eligible])
    mask=common_mask(row['frames'])[frames]; metric_mask=mask&known[frames]
    scores=np.array([item['score'] for item in eligible])
    frame_metrics=metrics(labels[frames][metric_mask],scores[metric_mask]) if len(np.unique(labels[frames][metric_mask]))==2 else None
    for item in trace:
        target=item['target_frame']
        item['label']=int(labels[target]) if target is not None else None
        item['shared_metric_valid']=int(target is not None and common_mask(row['frames'])[target] and known[target])
        item['inference_valid']=int(bool(item['inference_valid']))
        item['alarm']=int(bool(item['alarm']))
    report={'sequence':row['sequence'],'input_frames':row['frames'],'elapsed_seconds':elapsed,
            'frames_content_sha256':row['frames_content_sha256'],'label_sha256':row['label_sha256'],
            'input_fps':(row['frames']-1)/elapsed,'eligible_targets':len(eligible),
            'queue_wait_final_ms':trace[-1]['queue_wait_ms'],
            'queue_growth_ms':trace[-1]['queue_wait_ms']-trace[0]['queue_wait_ms'],
            'frame_metrics_shared_mask':frame_metrics,
            'operational_events':event_metrics(labels,frames,alarms,emissions,args.arrival_fps),
            'shared_target_events':event_metrics(labels,frames[mask],alarms[mask],emissions[mask],args.arrival_fps)}
    return trace,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['dinov3-l','vjepa21-l'],required=True)
    parser.add_argument('--mode',choices=['online','offline'],required=True)
    parser.add_argument('--device',choices=['R01','R02','R03','R04'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--adaptation',choices=['frozen','lora'],default='frozen')
    parser.add_argument('--lora-run',type=Path,help='Completed joint training directory; required for LoRA')
    parser.add_argument('--implementation',choices=['full','buffer','reuse'],default='full')
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    parser.add_argument('--arrival-fps',type=float,default=30)
    parser.add_argument('--sequences',nargs='+',help='Omit for every real test video of this device')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    args=parser.parse_args()
    if not np.isfinite(args.arrival_fps) or args.arrival_fps<=0: raise ValueError('Positive arrival FPS required')
    if args.implementation=='reuse' and (args.model,args.mode,args.precision)!=('dinov3-l','online','fp32'):
        raise ValueError('Only separately reported DINO online FP32 reuse is eligible; BF16 parity failed')
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Use a fresh, empty runtime output directory')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    require_isolated_gpu(); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads(args.manifest.read_text())['sequences']
    rows=[row for row in manifest if row['device']==args.device and row['partition']=='testing' and
          (args.sequences is None or row['sequence'] in args.sequences)]
    if not rows or (args.sequences and set(args.sequences)!={row['sequence'] for row in rows}):
        raise ValueError('Invalid test-video selection')
    model,head,scorer,meta,cycle,adaptation=fixed_components(args)
    normal=next(row for row in manifest if row['device']==args.device and row.get('split')=='fit')
    warm=ClipDataset(args.data_root/normal['relative_directory'],np.array([20]),args.mode)[0][0][None].cuda()
    warm_ring=DinoFrameRing(model.encoder) if args.implementation=='reuse' else None
    with torch.inference_mode():
        for index in range(20):
            with precision_context(args.precision):
                if warm_ring is not None:
                    warm_ring.push(warm[0,:,index%16],index)
                    local,pooled=warm_ring.features(padding=True)
                else:
                    local,pooled=model(warm)
            score_features(local,pooled,head,scorer,args.precision)
    del warm,local,pooled,warm_ring
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); args.out.mkdir(parents=True,exist_ok=True)
    reports=[]; all_eligible=[]; all_trace=[]
    for row in rows:
        require_isolated_gpu()
        trace,report=replay(args,row,model,head,scorer,meta['calibration']['P3'],cycle)
        require_isolated_gpu()
        with (args.out/f"{row['sequence']}.csv").open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(trace[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(trace)
        reports.append(report); all_trace.extend(trace)
        all_eligible.extend(item for item in trace if item['inference_valid'])
        print(json.dumps({'sequence':row['sequence'],'input_fps':report['input_fps'],'eligible_targets':report['eligible_targets']}),flush=True)
    latency=np.array([item['target_latency_ms'] for item in all_eligible])
    queue=np.array([item['queue_wait_ms'] for item in all_trace])
    shared=[item for item in all_eligible if item['shared_metric_valid']]
    shared_labels=np.array([item['label'] for item in shared])
    shared_scores=np.array([item['score'] for item in shared])
    event_totals={}
    for scope in ['operational_events','shared_target_events']:
        delays=[event['delay_seconds'] for item in reports for event in item[scope]['events']
                if event['delay_seconds'] is not None]
        event_totals[scope]={key:sum(item[scope][key] for item in reports) for key in
            ['covered_events','events_outside_coverage','detected_events','missed_events',
             'observed_end_evaluable_events','events_detected_before_observed_end',
             'late_detected_events','events_without_alarm_before_observed_end',
             'false_alarm_episode_starts','unknown_alarm_episode_starts','false_alarm_frames','evaluated_normal_frames']}
        event_totals[scope].update(onset_delay_p50_seconds=float(np.quantile(delays,.5)) if delays else None,
                                  onset_delay_p95_seconds=float(np.quantile(delays,.95)) if delays else None)
    report={'status':'complete_measured_replay','scope':'Paced wall-clock FIFO JPEG replay; no camera capture, no drops. Raw-source audit reads JPEGs before timing (warm file cache).',
        'backbone':args.model,'mode':args.mode,'device':args.device,'seed':args.seed,'variant':'P3',
        'implementation':args.implementation,'precision':args.precision,'arrival_fps':args.arrival_fps,
        'all_test_videos':args.sequences is None,'warmup_forward_calls':20,'alarm_warmup_target':19,
        'lookahead_frames':7 if args.mode=='offline' else 0,'test_bank_updates':False,
        'sustained_input_fps':sum(item['input_frames']-1 for item in reports)/sum(item['elapsed_seconds'] for item in reports),
        'fps_definition':'Completed interarrival intervals divided by first-arrival to last-completion wall time, including queue drain. At fixed 30 FPS arrival this is not a maximum-throughput benchmark.',
        'target_latency_p50_ms':float(np.quantile(latency,.5)),'target_latency_p95_ms':float(np.quantile(latency,.95)),
        'queue_wait_p50_ms':float(np.quantile(queue,.5)),'queue_wait_p95_ms':float(np.quantile(queue,.95)),
        'queue_wait_max_ms':float(queue.max()),
        'queue_growth_max_ms':max(item['queue_growth_ms'] for item in reports),
        'frame_metrics_shared_mask':metrics(shared_labels,shared_scores) if len(np.unique(shared_labels))==2 else None,
        'event_totals':event_totals,
        'stage_mean_ms':{key:float(np.mean([item[key] for item in all_eligible])) for key in
                         ['decode_resize_ms','encoder_ms','feature_cast_ms','phase_head_ms','projection_search_ms','score_alarm_ms','service_ms']},
        'peak_allocated_vram_mib':torch.cuda.max_memory_allocated()/1024**2,
        'peak_reserved_vram_mib':torch.cuda.max_memory_reserved()/1024**2,
        'normal_fit_sha256':file_hash(args.results/'normal_fit.json'),'phase_head_sha256':file_hash(args.run/'phase_head.pt'),
        'memory_sha256':file_hash(args.run/'memory.npz'),'weights_sha256':meta['cache_identity']['weights_sha256'],
        'manifest_sha256':file_hash(args.manifest),
        **adaptation,
        'runtime_adaptation_sha256':file_hash(Path(__file__).resolve().parents[1]/'src/ipad_jepa/runtime_adaptation.py'),
        'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
        'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'runtime_code_sha256':file_hash(Path(__file__)),
        'score_state_sha256':file_hash(Path(__file__).resolve().parents[1]/'src/ipad_jepa/runtime_state.py'),
        'cache_conversion':'BF16 output rounded to FP16 then FP32' if args.precision=='bf16' else 'FP32 output retained; secondary precision condition',
        'encoder_stage_includes':'Input clip stacking, host/device transfer, encoder and CUDA synchronization',
        'gpu_isolation':'No other compute PID at setup and before/after each video; transient jobs between checks are not independently monitored',
        'sequences':reports,
        'note':'Actual batch1 scores may differ from primary batch4 BF16 cache. FP32 reuse is not declared equivalent until paired score/alarm results are checked.'}
    temp=args.out/'runtime.json.tmp'; temp.write_text(json.dumps(report,indent=2)+'\n'); temp.replace(args.out/'runtime.json')
    print(json.dumps({key:report[key] for key in ['status','sustained_input_fps','target_latency_p95_ms','peak_allocated_vram_mib']}),flush=True)


if __name__=='__main__': main()
