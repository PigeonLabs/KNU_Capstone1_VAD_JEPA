"""Pinned IPAD VST baseline, with explicit executable repairs and paired B0/B1 scores."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import types
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


def repaired_memory_forward(self,features,period_score):
    if features.ndim!=2 or period_score.ndim!=2 or len(features)%len(period_score):
        raise ValueError('Flattened memory tokens must split into whole videos')
    batch=len(period_score); tokens=len(features)//batch
    confidence,classes=period_score.max(1)
    denominator=period_score.shape[1] if self.addressing=='paper200' else 126
    centres=torch.floor(classes.float()/denominator*self.mem_dim).long()
    logits=(features@self.weight.T).reshape(batch,tokens,self.mem_dim)
    columns=torch.arange(self.mem_dim,device=features.device)
    if self.addressing=='paper200':
        mask=(columns[None,:]>=centres[:,None]-7)&(columns[None,:]<centres[:,None]+8)
    else:
        # Preserve literal Python slicing (including empty out-of-range windows).
        mask=torch.zeros((batch,self.mem_dim),dtype=torch.bool,device=features.device)
        for b,centre in enumerate(centres.detach().cpu().tolist()):
            start,stop,_=slice(centre-7,centre+8).indices(self.mem_dim)
            mask[b,start:stop]=True
    logits=logits*(1+mask[:,None,:]*confidence[:,None,None])
    weights=logits.flatten(0,1).softmax(1)
    if self.shrink_thres>0:
        delta=weights-self.shrink_thres
        weights=(torch.relu(delta)*weights)/(delta.abs()+1e-12)
        weights=torch.nn.functional.normalize(weights,p=1,dim=1)
    return {'output':weights@self.weight,'att':weights}


def load_model(upstream:Path,addressing='paper200'):
    if addressing not in {'paper200','public126'}: raise ValueError('Unknown addressing policy')
    sys.path.insert(0,str(upstream.resolve()))
    model=importlib.import_module('model.video_swin_transformer').VST()
    memory=model.mem_rep.memory
    memory.addressing=addressing
    memory.forward=types.MethodType(repaired_memory_forward,memory)
    return model


class NativeClips(Dataset):
    def __init__(self,root:Path,rows:list[dict],preload=True,stride=1):
        import cv2
        cv2.setNumThreads(1)
        self.cv2=cv2; self.rows=rows; self.paths=[]; self.preloaded=[]; self.samples=[]
        if stride<1: raise ValueError('Positive stride required')
        for video,row in enumerate(rows):
            paths=sorted((root/row['relative_directory']).glob('*.jpg'),key=lambda p:int(p.stem))
            if len(paths)!=row['frames'] or [int(p.stem) for p in paths]!=list(range(row['frames'])):
                raise ValueError('Raw frame inventory differs from manifest')
            self.paths.append(paths)
            if preload:
                images=[]
                for path in paths: images.append(self.read(path))
                self.preloaded.append(np.stack(images))
            else: self.preloaded.append(None)
            self.samples.extend((video,start) for start in range(0,len(paths)-15,stride))
        if not self.samples: raise ValueError('No complete native clips')

    def read(self,path):
        pixels=self.cv2.imread(str(path))
        if pixels is None: raise ValueError(f'Cannot decode {path}')
        return self.cv2.resize(pixels,(256,256))

    def __len__(self): return len(self.samples)

    def __getitem__(self,index):
        video,start=self.samples[index]
        raw=self.preloaded[video]
        values=np.stack([self.read(p) for p in self.paths[video][start:start+16]]) if raw is None else raw[start:start+16]
        x=values.astype(np.float32)/127.5-1
        tensor=torch.from_numpy(np.ascontiguousarray(x.transpose(3,0,1,2)))
        phase=start*200//self.rows[video]['frames']
        return tensor,phase,video,start+8


def losses(output,input,phase):
    reconstruction=(output['output']-input).square().mean()
    attention=output['att'].movedim(1,-1).reshape(-1,output['att'].shape[1])
    entropy=-(attention*(attention+1e-12).log()).sum(1).mean()
    period=torch.nn.functional.cross_entropy(output['recon_index'],phase)
    return reconstruction+.0002*entropy+.02*period,(reconstruction,entropy,period)


def latent(model,input):
    before=model.transformer_encoder(input)
    phase=model.period(before)
    result=model.mem_rep(before,phase)
    return before,result['output'],result['att'],phase


def latent_score(model,input):
    before,after,_,_=latent(model,input)
    # Native encoder temporal cell 2 corresponds to the centre target at frame 8.
    return (before[:,:,2]-after[:,:,2]).square().mean((1,2,3))


def paired_scores(model,input):
    before,after,_,phase=latent(model,input)
    reconstruction=model.transformer_decoder(after.clone())
    mse=(reconstruction[:,:,8]-input[:,:,8]).square().mean((1,2,3))
    pixel=10*torch.log10(mse.clamp_min(1e-12)) # -PSNR: larger means more anomalous.
    feature=(before[:,:,2]-after[:,:,2]).square().mean((1,2,3))
    return pixel,feature,phase


def train(root,rows,upstream,out,seed,scope='controlled',epochs=50,preload=True,max_steps=None,addressing='paper200',resume=False):
    import csv
    from ipad_jepa.audit import inspect_frames
    from ipad_jepa.experiment import digest
    if scope not in {'reference','controlled'} or epochs<1 or (max_steps is not None and max_steps<1):
        raise ValueError('Invalid native training scope')
    selected=[r for r in rows if r['partition']=='training' and (scope=='reference' or r.get('split')=='fit')]
    previous=None
    if (out/'training.json').exists():
        if not resume: raise FileExistsError('Native run exists; use --resume for a full-epoch checkpoint')
        previous=json.loads((out/'training.json').read_text())
        expected={'scope':scope,'seed':seed,'addressing':addressing,'epochs_requested':epochs,'normal_videos':[r['sequence'] for r in selected],
                  'normal_source_hashes':[r['frames_content_sha256'] for r in selected],'code_sha256':digest(__file__)}
        if previous.get('max_steps') is not None or max_steps is not None or any(previous.get(k)!=v for k,v in expected.items()):
            raise ValueError('Cannot resume a pilot or a different native protocol/source')
    elif resume: raise FileNotFoundError('No native run to resume')
    if not selected or len({r['device'] for r in rows})!=1:
        raise ValueError('One device and nonempty normal training split required')
    upstream_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD']).decode().strip()
    if previous is not None and previous['upstream_commit']!=upstream_commit:
        raise ValueError('Native upstream changed since checkpoint')
    for row in selected:
        if inspect_frames(root/row['relative_directory'])['frames_content_sha256']!=row['frames_content_sha256']:
            raise ValueError('Normal training source contents changed')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dataset=NativeClips(root,selected,preload=preload)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,drop_last=True,generator=generator)
    model=load_model(upstream,addressing).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); curves=[]; step=0; start_epoch=0; prior_seconds=0
    if previous is not None:
        state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch']; curves=state['curves']; step=curves[-1]['steps_total']
        if len(curves)!=start_epoch or curves[-1]['epoch']!=start_epoch or state['seed']!=seed:
            raise ValueError('Inconsistent native checkpoint history/seed')
        prior_seconds=state['training_seconds']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['loader_rng']); del state
        if start_epoch>=epochs:
            if previous['status']!='complete_training' or start_epoch!=epochs: raise ValueError('Inconsistent completed native epoch')
            return previous
    metadata={'status':'running','scope':scope,'device':rows[0]['device'],'seed':seed,'epochs_requested':epochs,
              'upstream_commit':upstream_commit,
              'normal_videos':[r['sequence'] for r in selected],'normal_source_hashes':[r['frames_content_sha256'] for r in selected],
              'clips_per_epoch':len(dataset),'batch_size':8,'drop_last':True,'precision':'FP32, public-code default',
              'optimizer':'Adam lr=1e-4','addressing':addressing,
              'repair':'Undefined i repaired per video. paper200 also maps 200 phase classes into memory and clips boundary windows; public126 preserves original indexing.',
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'loss':'all-frame MSE + 0.0002 * memory entropy + 0.02 * start-position phase CE',
              'preprocessing':'OpenCV BGR full resize 256, [-1,1]','synthetic_pretraining':False,
              'checkpoint_selection':'Final epoch, public training policy. Not a paper-exact reproduction.',
              'code_sha256':digest(__file__),'torch':torch.__version__,
              'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
              'resume_from_epoch':start_epoch,
              'max_steps':max_steps}
    (out/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for epoch in range(start_epoch,epochs):
        model.train(); sums=np.zeros(4); n=0
        for x,phase,_,_ in loader:
            x=x.cuda(non_blocking=True); phase=phase.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss,parts=losses(model(x),x,phase)
            if not loss.isfinite(): raise ValueError('Nonfinite native training loss')
            loss.backward(); optimizer.step()
            sums+=np.array([float(loss.detach()),*[float(p.detach()) for p in parts]])*len(x); n+=len(x); step+=1
            if max_steps and step>=max_steps: break
        record={'epoch':epoch+1,'total':float(sums[0]/n),'mse':float(sums[1]/n),'entropy':float(sums[2]/n),'phase_ce':float(sums[3]/n),'clips':n,'steps_total':step}
        curves.append(record); print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==epochs or (max_steps and step>=max_steps):
            temp=out/'checkpoint.tmp.pt'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'seed':seed,
                        'curves':curves,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                        'loader_rng':generator.get_state(),'training_seconds':prior_seconds+time.perf_counter()-start},temp)
            temp.replace(out/'checkpoint.pt')
        with (out/'training.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(record),lineterminator='\n'); writer.writeheader(); writer.writerows(curves)
        if max_steps and step>=max_steps: break
    metadata.update(status='pilot_complete' if max_steps else 'complete_training',epochs_completed=len(curves),steps=step,seconds=prior_seconds+time.perf_counter()-start)
    (out/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True); p.add_argument('--upstream',type=Path,default=Path('third_party/IPAD'))
    p.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    p.add_argument('--device',choices=['R01','R02','R03','R04'],required=True); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--scope',choices=['reference','controlled'],default='controlled'); p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--max-steps',type=int,help='GPU pilot only; not a full baseline result')
    p.add_argument('--addressing',choices=['paper200','public126'],default='paper200')
    p.add_argument('--out',type=Path,required=True); p.add_argument('--no-preload',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume only from a saved complete epoch with identical source/protocol')
    args=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    rows=[r for r in json.loads(args.manifest.read_text())['sequences'] if r['device']==args.device]
    print(json.dumps(train(args.data_root,rows,args.upstream,args.out,args.seed,args.scope,args.epochs,not args.no_preload,args.max_steps,args.addressing,args.resume),indent=2))


if __name__=='__main__': main()
