"""Read/hash a completed derived cache; this command never deletes any files."""
import argparse
import json
from pathlib import Path
from ipad_jepa.cache_retention import digest, prepare_inventory
from summarize_lora_matrix import audit_lora


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['dinov3-l','vjepa21-l'],required=True)
    p.add_argument('--mode',choices=['offline','online'],required=True)
    p.add_argument('--device',choices=['R01','R02','R03','R04'],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--teacher-weight',type=float,choices=[0.,1.],default=1.)
    p.add_argument('--data-root',type=Path,default=Path('../IPAD_dataset/IPAD_dataset'))
    p.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    p.add_argument('--out',type=Path,default=Path('results/cache_retention'))
    args=p.parse_args();key=(args.model,args.mode,args.device,args.seed)
    relative=Path(args.model)/args.mode/args.device/f'seed{args.seed}'
    tag='T0' if args.teacher_weight==0 else 'T1'
    target=args.out/tag/relative
    if target.exists():raise ValueError('Existing inventory owner/output; inspect before retrying')
    zero=args.teacher_weight==0
    public=(Path('results/stage05/ablations/teacher_weight/T0') if zero else Path('results/stage04'))/relative
    training=Path('artifacts/lora_teacher0' if zero else 'artifacts/lora')/relative
    cache=Path('artifacts/features_lora_teacher0' if zero else 'artifacts/features_lora')/relative
    local=Path('artifacts/runs_lora_teacher0' if zero else 'artifacts/runs_lora')/relative
    names=['adaptation','adapted_features','features','backbones','cache_data','train_lora','memory','experiment','alignment']
    sources={name:digest(Path('src/ipad_jepa')/(name+'.py')) for name in names}
    source_pins={str(Path('src/ipad_jepa')/(name+'.py')):sha for name,sha in sources.items()}
    prepare_path=Path(__file__).resolve().relative_to(Path.cwd().resolve())
    for path in [prepare_path,args.manifest,Path('scripts/summarize_lora_matrix.py'),
                 Path('scripts/plot_lora_evaluation.py'),Path('src/ipad_jepa/lora_audit.py'),
                 Path('src/ipad_jepa/cache_retention.py')]:
        source_pins[str(path)]=digest(path)
    manifest=json.loads(args.manifest.read_text())['sequences']
    # A fresh independent audit must pass immediately before payload inventory.
    _,witness,witness_path=audit_lora(public,training,cache,local,Path('results/stage02')/relative,
                                     Path('artifacts/runs')/relative,manifest,args.data_root,
                                     key,args.teacher_weight,target/'original_condition_audit')
    normal=json.loads((public/'normal_fit.json').read_text())
    receipt=prepare_inventory(Path.cwd(),cache,manifest,witness,public,normal,key,args.teacher_weight,sources)
    if any(digest(path)!=sha for path,sha in source_pins.items()):
        raise ValueError('Pinned audit/inventory source changed during actual payload verification')
    receipt.update(manifest_sha256=digest(args.manifest),original_witness=str(witness_path.relative_to(target)),
                   original_witness_sha256=digest(witness_path),public_root=str(public),
                   training_root=str(training),local_root=str(local),
                   prepared_by_sha256=source_pins[str(prepare_path)],prepared_source_sha256=source_pins)
    tmp=target/'payload_inventory.json.tmp';tmp.write_text(json.dumps(receipt,indent=2)+'\n')
    tmp.replace(target/'payload_inventory.json')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['records','source_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
