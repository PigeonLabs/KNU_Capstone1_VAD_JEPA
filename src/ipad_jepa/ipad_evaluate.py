"""Calibrate and evaluate paired native IPAD pixel/latent scores from one checkpoint."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ipad_jepa.alignment import align,POLICY
from ipad_jepa.audit import inspect_frames
from ipad_jepa.ipad_baseline import load_model,NativeClips,paired_scores
import ipad_jepa.ipad_baseline as baseline
from ipad_jepa.temporal import common_mask
from ipad_jepa.experiment import metrics,alarms,digest


def collect(model,root,rows,batch=8):
    dataset=NativeClips(root,rows,preload=True)
    loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=2,pin_memory=True)
    output={r['sequence']:{'frame':[],'B0':[],'B1':[]} for r in rows}
    with torch.inference_mode():
        for x,_,video,targets in loader:
            pixel,feature,_=paired_scores(model,x.cuda(non_blocking=True))
            pixel=pixel.cpu().numpy(); feature=feature.cpu().numpy()
            for i,(v,t) in enumerate(zip(video.tolist(),targets.tolist())):
                item=output[rows[v]['sequence']]
                item['frame'].append(t); item['B0'].append(float(pixel[i])); item['B1'].append(float(feature[i]))
    return {key:{k:np.asarray(v) for k,v in values.items()} for key,values in output.items()}


def evaluate(root,rows,upstream,training,out):
    info=json.loads((training/'training.json').read_text())
    if info['status']!='complete_training' or info['epochs_completed']!=50 or info['max_steps'] is not None:
        raise ValueError('Native pilot/partial checkpoint is not a completed 50-epoch baseline')
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD']).decode().strip()
    if commit!=info['upstream_commit'] or digest(baseline.__file__)!=info['code_sha256']:
        raise ValueError('Native source differs from the training protocol')
    if {r['device'] for r in rows}!={info['device']}:
        raise ValueError('Native evaluation device differs from training')
    selected=[r for r in rows if r['partition']=='training' and (info['scope']=='reference' or r.get('split')=='fit')]
    if info['normal_source_hashes']!=[r['frames_content_sha256'] for r in selected] or info['normal_videos']!=[r['sequence'] for r in selected]:
        raise ValueError('Native checkpoint normal split differs from manifest')
    out.mkdir(parents=True,exist_ok=True); (out/'metrics.json').unlink(missing_ok=True)
    model=load_model(upstream,info['addressing']).cuda().eval()
    checkpoint=torch.load(training/'checkpoint.pt',map_location='cpu',weights_only=True)
    if checkpoint['epoch']!=50 or checkpoint['seed']!=info['seed']: raise ValueError('Wrong native checkpoint epoch/seed')
    model.load_state_dict(checkpoint['model'],strict=True); del checkpoint
    calibration_rows=[r for r in rows if r['partition']=='training' and r.get('split')=='calibration']
    for row in calibration_rows:
        if inspect_frames(root/row['relative_directory'])['frames_content_sha256']!=row['frames_content_sha256']:
            raise ValueError('Native calibration source changed')
    cal=collect(model,root,calibration_rows)
    parameters={}
    for variant in ['B0','B1']:
        values=np.concatenate([cal[r['sequence']][variant][common_mask(r['frames'])[cal[r['sequence']]['frame']]] for r in calibration_rows])
        if not np.all(np.isfinite(values)): raise ValueError('Invalid native normal calibration')
        median=float(np.median(values)); scale=max(float(1.4826*np.median(np.abs(values-median))),1e-6)
        threshold=float(np.quantile((values-median)/scale,.99))
        parameters[variant]={'median':median,'mad_scale':scale,'threshold':threshold}
    normal_info={'status':'normal_calibration_complete','checkpoint_sha256':digest(training/'checkpoint.pt'),
                 'training_scope':info['scope'],'calibration_overlap_with_training':info['scope']=='reference',
                 'selection':'Fixed final epoch50; normal calibration only','variants':parameters,
                 'B0':'Anomaly-high -PSNR at input frame8','B1':'Central latent temporal cell2 MSE over all 768 channels and 8x8 cells; no decoder in latent_score()'}
    (out/'normal_fit.json').write_text(json.dumps(normal_info,indent=2)+'\n')
    with (out/'normal_calibration.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n'); writer.writerow(['sequence','frame','valid','B0_raw','B1_raw'])
        for row in calibration_rows:
            item=cal[row['sequence']]; keep=common_mask(row['frames'])[item['frame']]
            writer.writerows(zip([row['sequence']]*len(keep),item['frame'],keep.astype(int),item['B0'],item['B1']))
    all_labels={v:[] for v in parameters}; all_scores={v:[] for v in parameters}
    sensitivity={v:{o:([],[]) for o in [-1,0,1]} for v in parameters}; unknown=[]
    for row in [r for r in rows if r['partition']=='testing']:
        if inspect_frames(root/row['relative_directory'])['frames_content_sha256']!=row['frames_content_sha256']:
            raise ValueError('Native test source changed')
        label_path=root/row['label_file']
        if digest(label_path)!=row['label_sha256']: raise ValueError('Native labels changed')
        labels,known,candidates=align(np.load(label_path,allow_pickle=False),row['frames'])
        values=collect(model,root,[row])[row['sequence']]; frames=values['frame']
        native_valid=common_mask(row['frames'])[frames]; valid=native_valid&known[frames]
        if not np.all(known[frames][native_valid]): unknown.append({'sequence':row['sequence'],'unknown_common_frames':int((native_valid&~known[frames]).sum())})
        for variant,c in parameters.items():
            score=(values[variant]-c['median'])/c['mad_scale']; alarm=alarms(score,c['threshold'],native_valid)
            if not np.all(np.isfinite(score)): raise ValueError('Invalid native test score')
            folder=out/variant; folder.mkdir(exist_ok=True)
            with (folder/f"{row['sequence']}.csv").open('w',newline='') as stream:
                writer=csv.writer(stream,lineterminator='\n'); writer.writerow(['frame','label','valid','inference_valid','raw','score','alarm'])
                writer.writerows(zip(frames,labels[frames],valid.astype(int),native_valid.astype(int),values[variant],score,alarm.astype(int)))
            all_labels[variant].append(labels[frames][valid]); all_scores[variant].append(score[valid])
            for o in [-1,0,1]:
                gt=candidates.get(o,candidates[0])[frames]; eligible=native_valid&(gt>=0)
                sensitivity[variant][o][0].append(gt[eligible]); sensitivity[variant][o][1].append(score[eligible])
        print(f"native scored: {row['device']}/{row['sequence']}",flush=True)
    result={'status':'complete_device_evaluation','device':info['device'],'backbone':'IPAD-native-repaired',
            'mode':'offline','scope':info['scope'],'seed':info['seed'],'test_videos':len([r for r in rows if r['partition']=='testing']),
            'variants':{v:metrics(np.concatenate(all_labels[v]),np.concatenate(all_scores[v])) for v in parameters},
            'alignment_policy':POLICY,'unresolved_alignment_sequences':unknown,
            'alignment_sensitivity':{v:{str(o):metrics(np.concatenate(y),np.concatenate(s)) for o,(y,s) in vals.items()} for v,vals in sensitivity.items()} if unknown else {},
            'addressing':info['addressing'],'decoder_training':'Used for both B0 and B1; B1 excludes decoder during its independent inference path.',
            'not_paper_exact':True,'code_sha256':digest(__file__)}
    temp=out/'metrics.json.tmp'; temp.write_text(json.dumps(result,indent=2)+'\n'); temp.replace(out/'metrics.json')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True); p.add_argument('--upstream',type=Path,default=Path('third_party/IPAD'))
    p.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json')); p.add_argument('--device',required=True)
    p.add_argument('--training',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    rows=[r for r in json.loads(args.manifest.read_text())['sequences'] if r['device']==args.device]
    print(json.dumps(evaluate(args.data_root,rows,args.upstream,args.training,args.out),indent=2))


if __name__=='__main__': main()
