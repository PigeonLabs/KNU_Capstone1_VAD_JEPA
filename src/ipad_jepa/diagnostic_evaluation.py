"""Evaluate fixed frozen P3 evidence on all held-out diagnostic interventions, in RAM."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from ipad_jepa.audit import inspect_frames
from ipad_jepa.backbones import Backbone, PhaseHead
from ipad_jepa.diagnostics import scenarios, confusion, macro_f1, CLASSES
from ipad_jepa.features import anchors, file_hash
from ipad_jepa.scoring import NormalCalibration
from ipad_jepa.temporal import clip_indices, common_mask, circular_phase, temporal_score
from ipad_jepa.torch_memory import TorchMemory

MODELS=('dinov3-l','vjepa21-l')
MODES=('offline','online')
DEVICES=('R01','R02','R03','R04')
SEEDS=(0,1,2)


def check_source_pins(pinned):
    """JSON source inventories store path keys as strings; hash actual Path objects."""
    if any(file_hash(Path(path))!=sha for path,sha in pinned.items()):
        raise ValueError('Active diagnostic protocol sources changed')


def save_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


class DiagnosticClips(Dataset):
    """Each target reads transformed timeline clips; original-source remapping is an injection."""
    def __init__(self, rgb, intervention, mode):
        if rgb.shape!=(intervention.frames,384,384,3) or rgb.dtype!=np.uint8:
            raise ValueError('Expected complete resized diagnostic uint8 RGB video')
        self.rgb,self.intervention,self.mode=rgb,intervention,mode
        self.sources=intervention.source_indices()
        self.targets=anchors(len(rgb),mode,'diagnostic',frames=16)
        self.cache=OrderedDict()
        self.mean=torch.tensor([.485,.456,.406])[:,None,None]
        self.std=torch.tensor([.229,.224,.225])[:,None,None]

    def __len__(self):return len(self.targets)

    def frame(self,index):
        if index not in self.cache:
            pixels=self.intervention.appearance_frame(self.rgb[self.sources[index]],index)
            # Same full-frame PIL resize/uint8 -> FP32 ImageNet transform as the primary reader.
            tensor=torch.from_numpy(np.array(pixels)).permute(2,0,1).float()/255
            self.cache[index]=(tensor-self.mean)/self.std
            if len(self.cache)>64:self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self,index):
        target=int(self.targets[index])
        ids=clip_indices(target,len(self.rgb),self.mode,frames=16,padding=False)
        return torch.stack([self.frame(int(i)) for i in ids],dim=1),target


def load_rgb(root,row):
    if row.get('partition')!='training' or row.get('split')!='diagnostic':
        raise ValueError('Only held-out diagnostic normal videos are eligible')
    folder=root/row['relative_directory']
    actual=inspect_frames(folder)
    if any(actual[k]!=row[k] for k in ['frames','names_sha256','frames_content_sha256']):
        raise ValueError('Audited held-out diagnostic source changed')
    paths=sorted(folder.glob('*.jpg'),key=lambda p:int(p.stem))
    frames=[]
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.array(image.convert('RGB').resize((384,384),Image.Resampling.BILINEAR)))
    return np.stack(frames)


def fixed_heads(root,public,model,mode,device,identity,manifest):
    """Bind three selected heads, actual normal banks and original unchanged normal q99."""
    result=[]
    for seed in SEEDS:
        condition=Path(model)/mode/device/f'seed{seed}'
        local,folder=root/condition,public/condition
        fit=json.loads((folder/'normal_fit.json').read_text())
        head_meta=json.loads((local/'phase_training.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())
        if (fit['status']!='normal_fit_and_calibration_complete' or metrics['status']!='complete_device_evaluation'
                or tuple(fit[k] for k in ['backbone','mode','device','seed'])!=(model,mode,device,seed)
                or tuple(metrics[k] for k in ['backbone','mode','device','seed'])!=(model,mode,device,seed)
                or head_meta['status']!='complete' or head_meta['epochs']!=20 or head_meta['seed']!=seed
                or tuple(head_meta[k] for k in ['backbone','mode','device'])!=(model,mode,device)
                or fit['phase_selected_epoch']!=head_meta['selected_epoch']
                or fit['cache_identity']!=identity
                or sorted(fit['normal_cache_fingerprints'])!=sorted(head_meta['cache_fingerprints'])
                or file_hash(local/'phase_head.pt')!=fit['phase_checkpoint_sha256']
                or (fit['bins'],fit['prototypes_per_bin'],fit['total_prototypes'],fit['pca_dimensions'])!=(16,128,2048,256)):
            raise ValueError('Fixed frozen normal primary condition or head differs')
        cycle=float(np.median([r['frames'] for r in manifest if r['device']==device and r.get('split')=='fit']))
        params=fit['calibration']['P3']
        with (folder/'normal_calibration.csv').open() as stream:
            cal_rows=[r for r in csv.DictReader(stream) if r['valid']=='1']
        pairs=np.array([[float(r['feature_raw']),float(r['time_raw'])] for r in cal_rows])
        cal=NormalCalibration().fit(pairs)
        for computed,declared in [(cal.median,params['median']),(cal.scale,params['mad_scale']),
                                  (cal.component_thresholds,params['component_thresholds']),(cal.threshold,params['threshold'])]:
            np.testing.assert_allclose(computed,declared,rtol=0,atol=1e-9)
        # Recalculation above verifies the existing normal fit; use its recorded values unchanged.
        cal.median=np.array(params['median']);cal.scale=np.array(params['mad_scale'])
        cal.component_thresholds=np.array(params['component_thresholds']);cal.threshold=params['threshold']
        with np.load(local/'memory.npz',allow_pickle=False) as bank:
            mean,components,prototypes=(bank[k].copy(order='K') for k in ['mean','components','prototypes'])
            temperature=float(bank['temperature'])
            if (mean.shape!=(1024,) or components.shape!=(256,1024) or prototypes.shape!=(16,128,256)
                    or not all(np.isfinite(a).all() for a in [mean,components,prototypes])
                    or temperature!=fit['temperature'] or float(bank['cycle_length'])!=cycle
                    or fit['cycle_length_fit_median']!=cycle):
                raise ValueError('Actual normal bank geometry/temperature/cycle differs')
        memory=SimpleNamespace(bins=16,dimensions=256,temperature=temperature,
                               pca=SimpleNamespace(mean_=mean,components_=components),prototypes=prototypes)
        head=PhaseHead().cuda().eval()
        head.load_state_dict(torch.load(local/'phase_head.pt',weights_only=True,map_location='cpu'),strict=True)
        proof={'seed':seed,'normal_fit_sha256':file_hash(folder/'normal_fit.json'),
               'normal_calibration_sha256':file_hash(folder/'normal_calibration.csv'),
               'phase_training_sha256':file_hash(local/'phase_training.json'),
               'phase_head_sha256':file_hash(local/'phase_head.pt'),'memory_sha256':file_hash(local/'memory.npz'),
               'cycle_length_normal_fit':cycle,'P3_normal_calibration':params}
        result.append((head,TorchMemory(memory).cuda().eval(),cal,cycle,proof))
    return result


def encode(model,dataset,batch):
    patches=np.empty((len(dataset),576,1024),np.float16)
    global_features=np.empty((len(dataset),1024),np.float16)
    cursor=0
    with torch.inference_mode():
        for video,ids in DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0):
            if not np.array_equal(ids.numpy(),dataset.targets[cursor:cursor+len(ids)]):
                raise ValueError('Diagnostic target order changed')
            with torch.autocast('cuda',dtype=torch.bfloat16):local,pooled=model(video.cuda())
            if local.shape!=(len(ids),576,1024) or pooled.shape!=(len(ids),1024) or not local.isfinite().all() or not pooled.isfinite().all():
                raise ValueError('Invalid real diagnostic encoder output')
            patches[cursor:cursor+len(ids)]=local.float().cpu().numpy().astype(np.float16)
            global_features[cursor:cursor+len(ids)]=pooled.float().cpu().numpy().astype(np.float16)
            cursor+=len(ids)
    if cursor!=len(dataset) or not np.isfinite(patches).all() or not np.isfinite(global_features).all():
        raise ValueError('Incomplete or nonfinite FP16 diagnostic representation')
    return patches,global_features


def evaluate(patches,global_features,targets,length,fixed):
    head,scorer,cal,cycle,_=fixed
    probabilities=[]
    raw_features=[]
    with torch.inference_mode():
        for start in range(0,len(targets),256):
            value=torch.from_numpy(global_features[start:start+256].astype(np.float32)).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=head(value)
            probabilities.append(logits.float().softmax(1).cpu().numpy())
        phase=circular_phase(np.concatenate(probabilities))
        for start in range(0,len(targets),16):
            value=torch.from_numpy(patches[start:start+16].astype(np.float32)).cuda()
            phi=torch.from_numpy(phase[start:start+16].astype(np.float32)).cuda()
            raw_features.append(scorer.frame_scores(value,phi,k=5).cpu().numpy())
    dense=np.full(length,np.nan);dense[targets]=phase
    raw=np.stack([np.concatenate(raw_features),temporal_score(dense,cycle,5)[targets]],axis=1)
    scores=cal.combine(raw)
    predicted=np.zeros(len(targets),np.int64)
    finite=np.isfinite(raw).all(1);predicted[finite]=cal.types(raw[finite])
    valid=common_mask(length)[targets]
    if not np.isfinite(raw[valid]).all() or not np.isfinite(scores[valid]).all():
        raise ValueError('Nonfinite eligible diagnostic scores')
    return phase,raw,scores,predicted,valid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    parser.add_argument('--models',choices=MODELS,nargs='+',default=list(MODELS))
    parser.add_argument('--modes',choices=MODES,nargs='+',default=list(MODES))
    parser.add_argument('--devices',choices=DEVICES,nargs='+',default=list(DEVICES))
    parser.add_argument('--runs',type=Path,default=Path('artifacts/runs'))
    parser.add_argument('--primary',type=Path,default=Path('results/stage02'))
    parser.add_argument('--out',type=Path,default=Path('results/stage05/diagnostics'))
    parser.add_argument('--ledger',type=Path,default=Path('artifacts/tmp/diagnostic_pipeline.json'))
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    if args.batch_size!=4:raise ValueError('Primary BF16 encoder batch4 protocol required')
    for values in [args.models,args.modes,args.devices]:
        if len(set(values))!=len(values):raise ValueError('Duplicate diagnostic group')
    if args.out.exists() or args.ledger.exists():raise ValueError('Fresh diagnostic namespace required; inspect actual owners before recovery')
    manifest=json.loads(args.manifest.read_text())['sequences']
    source=Path(__file__).parent
    sources=[Path(__file__),source/'diagnostics.py',args.manifest,Path('configs/experiment_matrix.yaml')]
    sources += [source/f'{name}.py' for name in ['backbones','features','audit','temporal','scoring','torch_memory','memory']]
    pinned={str(p):file_hash(p) for p in sources}
    ledger={'status':'running','models':args.models,'modes':args.modes,'devices':args.devices,'seeds':list(SEEDS),
            'scenarios_per_video':49,'source_sha256':pinned,'completed_groups':[]}
    save_json(args.ledger,ledger)
    def check_sources():
        check_source_pins(pinned)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    try:
        for name in args.models:
            weights=Path('artifacts/weights')/{'dinov3-l':'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                                             'vjepa21-l':'vjepa2_1_vitl_dist_vitG_384.pt'}[name]
            upstream=Path('third_party')/('dinov3' if name=='dinov3-l' else 'vjepa2')
            identity={'backbone':name,'weights_sha256':file_hash(weights),'upstream_commit':subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
                      'adapter_sha256':file_hash(source/'backbones.py'),'reader_sha256':file_hash(source/'features.py'),'image_size':384,'clip_frames':16,
                      'fit_stride':4,'preprocessing':'RGB full-frame PIL bilinear resize; ImageNet mean/std','feature_dtype':'float16 from BF16 inference'}
            for mode in args.modes:
                model=Backbone(name,upstream,weights,mode).cuda().eval()
                for device in args.devices:
                    check_sources()
                    rows=[r for r in manifest if r['device']==device and r.get('split')=='diagnostic']
                    expected={'R01':6,'R02':4,'R03':3,'R04':4}[device]
                    if len(rows)!=expected or any(r['partition']!='training' for r in rows):raise ValueError('Held-out diagnostic inventory differs')
                    fixed=fixed_heads(args.runs,args.primary,name,mode,device,dict(identity,mode=mode),manifest)
                    matrices=[np.zeros((4,4),np.int64) for _ in SEEDS]
                    outputs=[{} for _ in SEEDS]
                    recipe_proofs=[]
                    for row in rows:
                        rgb=load_rgb(args.data_root,row)
                        cases=scenarios(row)
                        if len(cases)!=49:raise ValueError('Full diagnostic scenario inventory required')
                        recipes={'sequence':row['sequence'],'frames':row['frames'],'frames_content_sha256':row['frames_content_sha256'],
                                 'frame_names_sha256':row['names_sha256'],'recipes':[c.metadata() for c in cases]}
                        recipe_path=args.out/name/mode/device/'recipes'/f"{row['sequence']}.json"
                        save_json(recipe_path,recipes);recipe_proofs.append({'sequence':row['sequence'],'path':str(recipe_path),'sha256':file_hash(recipe_path)})
                        for intervention in cases:
                            check_sources()
                            ledger['current']={'backbone':name,'mode':mode,'device':device,'sequence':row['sequence'],'scenario':intervention.name}
                            save_json(args.ledger,ledger)
                            dataset=DiagnosticClips(rgb,intervention,mode)
                            patches,pooled=encode(model,dataset,args.batch_size)
                            truth=intervention.labels()[dataset.targets]
                            for seed,components in zip(SEEDS,fixed):
                                phase,raw,scores,predicted,valid=evaluate(patches,pooled,dataset.targets,len(rgb),components)
                                matrices[seed]+=confusion(truth[valid],predicted[valid])
                                path=args.out/name/mode/device/f'seed{seed}'/'traces'/row['sequence']/f'{intervention.name}.csv'
                                path.parent.mkdir(parents=True,exist_ok=True)
                                with path.open('w',newline='') as stream:
                                    writer=csv.writer(stream,lineterminator='\n')
                                    writer.writerow(['frame','valid','phase','feature_raw','time_raw','score','evidence_type','intervention_type'])
                                    writer.writerows(zip(dataset.targets,valid.astype(int),phase,raw[:,0],raw[:,1],scores,predicted,truth))
                                outputs[seed][str(path.relative_to(args.out/name/mode/device/f'seed{seed}'))]=file_hash(path)
                            del dataset,patches,pooled
                            print(f'diagnostic scenario complete: {name}/{mode}/{device}/{row["sequence"]}/{intervention.name}',flush=True)
                        del rgb
                    check_sources()
                    for seed,components,matrix in zip(SEEDS,fixed,matrices):
                        f1,per_class=macro_f1(matrix)
                        record={'status':'complete_diagnostic_condition','backbone':name,'mode':mode,'device':device,'seed':seed,
                                'classes':list(CLASSES),'normal_only_fixed_source':components[-1],'held_out_videos':len(rows),'scenarios_per_video':49,
                                'confusion_matrix':matrix.tolist(),'macro_f1':f1,'per_class_f1':per_class,'valid_mask':'t=19..N-8 inclusive',
                                'recipes':recipe_proofs,'trace_sha256':outputs[seed],'source_sha256':pinned,'gpu':torch.cuda.get_device_name(0),
                                'precision':'BF16 encoder batch4 -> FP16 local/context features -> FP32 head/search inputs; BF16 head batch256',
                                'scope':'Intervention timeline truth and component-q99 evidence classification; fixed normal-only P3 fit; no synthetic training, causal fault labels or runtime claim'}
                        save_json(args.out/name/mode/device/f'seed{seed}'/'diagnostic_metrics.json',record)
                    ledger['completed_groups'].append(f'{name}/{mode}/{device}')
                    save_json(args.ledger,ledger)
                    del fixed
                del model;torch.cuda.empty_cache()
        check_sources();ledger['status']='complete_diagnostic_matrix';ledger.pop('current',None);save_json(args.ledger,ledger)
    except BaseException as error:
        ledger['status']='failed';ledger['error']=repr(error);save_json(args.ledger,ledger);raise


if __name__=='__main__':main()
