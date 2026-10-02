"""Run fresh lambda_teacher=0 LoRA conditions against the existing lambda=1 matrix."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path

from ipad_jepa.teacher_ablation import check_zero_training, expected_cache_bytes

MODELS=("dinov3-l","vjepa21-l")
MODES=("offline","online")
DEVICES=("R01","R02","R03","R04")
SEEDS=(0,1,2)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--devices',nargs='+',choices=DEVICES,default=list(DEVICES))
    p.add_argument('--training',type=Path,default=Path('artifacts/lora_teacher0'))
    p.add_argument('--cache',type=Path,default=Path('artifacts/features_lora_teacher0'))
    p.add_argument('--runs',type=Path,default=Path('artifacts/runs_lora_teacher0'))
    p.add_argument('--out',type=Path,default=Path('results/stage05/ablations/teacher_weight/T0'))
    p.add_argument('--ledger',type=Path,default=Path('artifacts/tmp/teacher0_pipeline.json'))
    p.add_argument('--disk-reserve-gib',type=float,default=64.)
    args=p.parse_args()
    if not 0<=args.disk_reserve_gib<1024 or any(len(x)!=len(set(x)) for x in [args.models,args.modes,args.devices]):
        raise ValueError('Invalid disk reserve or duplicate conditions')
    for path in [args.training,args.cache,args.runs,args.out,args.ledger]:
        if path.exists():raise ValueError(f'Fresh treatment namespace required: {path}')
    files=[Path(__file__),args.manifest,Path('configs/experiment_matrix.yaml')]
    files.extend(Path('src/ipad_jepa')/f'{name}.py' for name in ['teacher_ablation','train_lora','adaptation','adapted_features','backbones','features','temporal','audit','cache_data','train_phase','experiment','memory','torch_memory','scoring','alignment'])
    pins={str(path):digest(path) for path in files}
    manifest=json.loads(args.manifest.read_text())['sequences']
    state={'status':'running','teacher_weight':0.,'paired_teacher_weight':1.,'normal_only':True,
           'planned_seed_conditions':len(args.models)*len(args.modes)*len(args.devices)*3,'seeds':list(SEEDS),
           'source_sha256':pins,'completed':[],
           'scope':'Phase CE plus zero times fixed-teacher normalized dense L2; identical trainer/data/seed/optimizer/20-epoch CE selection. Teacher caches remain read for identical input validation and dense-loss monitoring, not a weighted objective. New adapted features/head/PCA/bank/normal calibration. No accuracy/runtime claim from matrix start.'}
    write(args.ledger,state)
    def check_sources():
        if any(digest(path)!=value for path,value in pins.items()):raise ValueError('Active zero-weight source changed')
    def run(command):
        check_sources();subprocess.run([sys.executable,*command],check=True)
    try:
        for model in args.models:
            weights=Path('artifacts/weights')/{'dinov3-l':'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth','vjepa21-l':'vjepa2_1_vitl_dist_vitG_384.pt'}[model]
            upstream=Path('third_party')/('dinov3' if model=='dinov3-l' else 'vjepa2')
            for mode in args.modes:
                for device in args.devices:
                    rows=[r for r in manifest if r['device']==device]
                    needed=expected_cache_bytes(rows,mode)+int(args.disk_reserve_gib*1024**3)
                    for seed in SEEDS:
                        condition=Path(model)/mode/device/f'seed{seed}'
                        training,cache,local,public=[root/condition for root in [args.training,args.cache,args.runs,args.out]]
                        state['current']={'backbone':model,'mode':mode,'device':device,'seed':seed,'step':'checking_new_condition_disk_capacity'};write(args.ledger,state)
                        while shutil.disk_usage(Path.cwd()).free<needed:
                            check_sources();state.update(status='waiting_for_derived_cache_capacity',required_free_bytes=needed,free_bytes=shutil.disk_usage(Path.cwd()).free);write(args.ledger,state);time.sleep(20)
                        state['status']='running';state['current']['step']='full20_normal_only_zero_teacher_weight_training';write(args.ledger,state)
                        run(['-m','ipad_jepa.train_lora','--data-root',str(args.data_root),'--manifest',str(args.manifest),
                             '--teacher-cache',f'artifacts/features/{model}/{mode}','--model',model,'--mode',mode,
                             '--device',device,'--seed',str(seed),'--weights',str(weights),'--upstream',str(upstream),
                             '--epochs','20','--accumulation','8','--teacher-weight','0.0','--out',str(training)])
                        meta=json.loads((training/'lora_training.json').read_text())
                        with (training/'lora_training.csv').open() as stream:curves=list(csv.DictReader(stream))
                        check_zero_training(meta,curves,model,mode,device,seed)
                        state['current']['step']='reencode_selected_zero_weight_normal_and_test_features';write(args.ledger,state)
                        run(['-m','ipad_jepa.adapted_features','--run',str(training),'--weights',str(weights),'--upstream',str(upstream),
                             '--data-root',str(args.data_root),'--manifest',str(args.manifest),'--cache',str(cache),'--phase-out',str(local)])
                        public.mkdir(parents=True,exist_ok=True)
                        for name in ['lora_training.json','lora_training.csv']:shutil.copyfile(training/name,public/name)
                        shutil.copyfile(local/'phase_training.json',public/'phase_training.json')
                        state['current']['step']='refit_PCA_memory_temperature_normal_q99_and_evaluate';write(args.ledger,state)
                        run(['-m','ipad_jepa.experiment','--cache',str(cache),'--manifest',str(args.manifest),'--phase-run',str(local),
                             '--data-root',str(args.data_root),'--device',device,'--seed',str(seed),'--out',str(public),'--local',str(local)])
                        metrics=json.loads((public/'metrics.json').read_text());normal=json.loads((public/'normal_fit.json').read_text());phase=json.loads((local/'phase_training.json').read_text())
                        if (metrics['status']!='complete_device_evaluation' or normal['status']!='normal_fit_and_calibration_complete'
                                or tuple(metrics[k] for k in ['backbone','mode','device','seed'])!=(model,mode,device,seed)
                                or normal['phase_checkpoint_sha256']!=digest(local/'phase_head.pt')
                                or phase['selected_adapter_sha256']!=digest(training/'selected_adapter.pt')):
                            raise ValueError('Incomplete or mismatched zero-weight adapted evaluation')
                        caches=[]
                        for row in rows:
                            if row.get('split','test') not in {'fit','calibration','test'}:continue
                            base=cache/device/row['partition']/row['sequence'];info=json.loads((base/'meta.json').read_text())
                            if info['status']!='complete' or info['spec']['selected_adapter_sha256']!=phase['selected_adapter_sha256']:raise ValueError('Zero-weight selected cache differs')
                            caches.append({'partition':row['partition'],'sequence':row['sequence'],'split':row.get('split','test'),
                                           'fingerprint':info['fingerprint'],'spec':info['spec'],'metadata_sha256':digest(base/'meta.json'),'targets_sha256':digest(base/'targets.npy')})
                        check_sources()
                        proof={'status':'completed_zero_teacher_weight_condition','teacher_weight':0.,'backbone':model,'mode':mode,'device':device,'seed':seed,
                               'source_sha256':pins,'cache_metadata':caches,'selected_adapter_sha256':phase['selected_adapter_sha256'],
                               'joint_head_sha256':digest(local/'phase_head.pt'),'memory_sha256':digest(local/'memory.npz'),
                               'outputs_sha256':{str(path.relative_to(public)):digest(path) for path in sorted(public.rglob('*')) if path.is_file()},
                               'paired_teacher1_reference':str(Path('results/stage04')/condition),'scope':state['scope']}
                        write(public/'teacher0_source_proof.json',proof);state['completed'].append(str(condition));write(args.ledger,state)
                        print(f'zero teacher-weight condition complete: {condition}',flush=True)
        state['status']='complete_matrix';state.pop('current',None);write(args.ledger,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error));write(args.ledger,state);raise

if __name__=='__main__':main()
