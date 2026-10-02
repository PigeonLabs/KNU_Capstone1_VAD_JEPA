"""Strict inventories for completed derived LoRA caches; never selects raw/teacher files."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np

MODELS = {'dinov3-l', 'vjepa21-l'}
DEVICES = {'R01', 'R02', 'R03', 'R04'}
ARRAY_NAMES = ('patch.npy', 'global.npy')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def reject_symlinks(path, boundary):
    path, boundary = Path(path).absolute(), Path(boundary).absolute()
    try: parts = path.relative_to(boundary).parts
    except ValueError: raise ValueError('Cache member escapes repository') from None
    if boundary.is_symlink(): raise ValueError('Symlink repository boundary')
    current = boundary
    for part in parts:
        current /= part
        if current.is_symlink(): raise ValueError('Symlink in cache path')


def condition_cache(repo, cache, condition, teacher_weight):
    model, mode, device, seed = condition
    if (model not in MODELS or mode not in {'offline','online'} or device not in DEVICES
            or seed not in (0,1,2) or teacher_weight not in (0.,1.)):
        raise ValueError('Accepted completed LoRA condition required')
    repo = Path(repo).absolute()
    namespace = 'features_lora_teacher0' if teacher_weight == 0 else 'features_lora'
    expected = repo/'artifacts'/namespace/model/mode/device/f'seed{seed}'
    actual = Path(cache) if Path(cache).is_absolute() else repo/Path(cache)
    reject_symlinks(actual, repo)
    if actual.resolve() != expected.resolve():
        raise ValueError('Only own per-condition adapted cache is eligible')
    return expected


def member(root, relative, allowed_names):
    path = Path(relative)
    if (path.is_absolute() or '..' in path.parts or str(path) != str(relative)
            or len(path.parts) != 4 or path.parts[0] not in DEVICES
            or path.parts[1] not in {'training','testing'}
            or not path.parts[2].isdigit() or path.name not in allowed_names):
        raise ValueError('Invalid derived cache member')
    target = Path(root)/path
    reject_symlinks(target, root)
    return target


def expected_targets(row, spec):
    if spec['clip_frames'] != 16 or spec['fit_stride'] != 4:
        raise ValueError('Primary LoRA input/fit density differs')
    start = (0 if row.get('split') == 'fit' else 15) if spec['mode']=='online' else 8
    stop = row['frames'] if spec['mode']=='online' else row['frames']-7
    return np.arange(start, max(start, stop), 4 if row.get('split')=='fit' else 1, dtype=np.int64)


def metadata_record(root, row, old_check, proof, normal, source_hashes):
    relative = Path(row['device'])/row['partition']/row['sequence']
    meta_path = member(root, str(relative/'meta.json'), ('meta.json',))
    target_path = member(root, str(relative/'targets.npy'), ('targets.npy',))
    meta = json.loads(meta_path.read_text()); spec = meta['spec']
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    expected = {'device':row['device'],'partition':row['partition'],'sequence':row['sequence'],
                'frames':row['frames'],'split':row.get('split','test'),
                'frame_content_sha256':row['frames_content_sha256'],
                'frame_names_sha256':row['names_sha256'], 'adaptation':'qv_lora_last4',
                'adaptation_seed':proof['seed'], 'selected_epoch':proof['tensor_provenance']['selected_epoch'],
                'selected_adapter_sha256':proof['selected_adapter_sha256'],
                'adaptation_source_sha256':source_hashes['adaptation'],
                'adapted_reader_sha256':source_hashes['adapted_features']}
    if meta.get('status')!='complete' or meta.get('fingerprint')!=fingerprint:
        raise ValueError('Retained cache fingerprint/status differs')
    if any(spec.get(k)!=v for k,v in expected.items()):
        raise ValueError('Retained cache differs from manifest/selected adapter')
    identity = {k:spec[k] for k in normal['cache_identity']}
    if identity != normal['cache_identity']:
        raise ValueError('Retained cache encoder/input identity differs')
    targets = np.load(target_path, allow_pickle=False)
    np.testing.assert_array_equal(targets, expected_targets(row,spec))
    if targets.dtype!=np.int64 or targets.ndim!=1 or old_check['clips']!=len(targets):
        raise ValueError('Retained target dtype/count differs')
    if old_check['metadata_sha256']!=digest(meta_path) or old_check['fingerprint']!=fingerprint:
        raise ValueError('Retained metadata differs from actual original cache audit')
    return {'relative_directory':str(relative),'metadata_sha256':digest(meta_path),
            'targets_sha256':digest(target_path),'fingerprint':fingerprint,'clips':len(targets),
            'split':row.get('split','test')}


def inspect_array(path, shape, chunk_rows=16):
    """Hash the full actual NPY file and check every payload value in bounded RAM."""
    before = Path(path).stat()
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if value.shape!=shape or value.dtype!=np.float16 or not value.flags.c_contiguous:
        raise ValueError('Derived array shape/dtype/layout differs')
    for first in range(0,len(value),chunk_rows):
        if not np.isfinite(value[first:first+chunk_rows]).all():
            raise ValueError('Nonfinite derived encoder payload')
    header_bytes=int(value.offset)
    with Path(path).open('rb') as stream:
        header_sha256=hashlib.sha256(stream.read(header_bytes)).hexdigest()
    del value
    sha = digest(path); after = Path(path).stat()
    identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if identity(before)!=identity(after): raise ValueError('Derived array changed during inventory')
    return {'sha256':sha,'bytes':after.st_size,'shape':list(shape),'dtype':'float16',
            'npy_header_bytes':header_bytes,'npy_header_sha256':header_sha256,
            'finite_payload_checked':True,'file_identity':list(identity(after))}


def prepare_inventory(repo, cache, manifest, witness, public, normal, condition, teacher_weight, source_hashes):
    root = condition_cache(repo, cache, condition, teacher_weight)
    if witness['status']!='passed_completed_lora_condition' or witness['teacher_weight']!=teacher_weight:
        raise ValueError('Actual original independent condition audit required')
    if tuple(witness[k] for k in ('backbone','mode','device','seed'))!=tuple(condition):
        raise ValueError('Original condition audit differs')
    for name, expected in witness['public_sources_sha256'].items():
        if digest(Path(public)/name)!=expected: raise ValueError('Original audited public source changed')
    rows = [r for split in ('fit','calibration','test') for r in manifest
            if r['device']==condition[2] and r.get('split','test')==split
            and r['partition']==('testing' if split=='test' else 'training')]
    checks = witness['tensor_provenance']['cache_checks']
    keys = lambda values:[(r['partition'],r['sequence']) for r in values]
    if not rows or keys(rows)!=keys(checks): raise ValueError('Historical cache inventory differs from manifest')
    records=[]
    for row, old in zip(rows,checks):
        item=metadata_record(root,row,old,witness,normal,source_hashes)
        n=item['clips']; directory=item['relative_directory']
        item['arrays']={name:inspect_array(member(root,str(Path(directory)/name),ARRAY_NAMES),shape)
                        for name,shape in [('patch.npy',(n,576,1024)),('global.npy',(n,1024))]}
        records.append(item)
    return {'status':'prepared_complete_derived_payload_inventory','condition':list(condition),
            'teacher_weight':teacher_weight,'cache_root':str(root.relative_to(Path(repo).absolute())),
            'records':records,'array_files_checked':len(records)*2,
            'derived_bytes_checked':sum(a['bytes'] for r in records for a in r['arrays'].values()),
            'source_sha256':source_hashes,'inventory_code_sha256':digest(__file__),
            'scope':'Actual full FP16 payload hashes, shapes/layout and all finite values checked before any removal. Metadata/targets/model/head/bank/public traces are retained. No encoder or GPU distance replay.'}


def check_saved_inventory(repo, cache, receipt, witness, manifest, normal, source_hashes):
    condition=tuple(receipt['condition']);weight=receipt['teacher_weight']
    root=condition_cache(repo,cache,condition,weight)
    if (witness['status']!='passed_completed_lora_condition' or witness['teacher_weight']!=weight
            or tuple(witness[k] for k in ('backbone','mode','device','seed'))!=condition):
        raise ValueError('Historical actual condition witness differs')
    if (receipt['status']!='prepared_complete_derived_payload_inventory'
            or receipt['cache_root']!=str(root.relative_to(Path(repo).absolute()))
            or receipt['inventory_code_sha256']!=digest(__file__)
            or receipt['source_sha256']!=source_hashes):
        raise ValueError('Historical payload inventory/source differs')
    rows=[r for split in ('fit','calibration','test') for r in manifest
          if r['device']==condition[2] and r.get('split','test')==split
          and r['partition']==('testing' if split=='test' else 'training')]
    old=witness['tensor_provenance']['cache_checks']; records=receipt['records']
    if len(rows)!=len(old) or len(rows)!=len(records) or receipt['array_files_checked']!=len(rows)*2:
        raise ValueError('Historical payload inventory count differs')
    if [(r['partition'],r['sequence']) for r in rows]!=[(r['partition'],r['sequence']) for r in old]:
        raise ValueError('Historical manifest/cache inventory differs')
    total=0;plan=[]
    for row,check,record in zip(rows,old,records):
        actual=metadata_record(root,row,check,witness,normal,source_hashes)
        if any(record.get(k)!=v for k,v in actual.items()) or set(record['arrays'])!=set(ARRAY_NAMES):
            raise ValueError('Retained metadata/targets or array inventory changed')
        n=actual['clips']
        for name,shape in [('patch.npy',(n,576,1024)),('global.npy',(n,1024))]:
            value=record['arrays'][name]
            if (value['shape']!=list(shape) or value['dtype']!='float16'
                    or value['finite_payload_checked'] is not True
                    or not isinstance(value.get('npy_header_bytes'),int) or value['npy_header_bytes']<10
                    or value['bytes']!=int(np.prod(shape))*2+value['npy_header_bytes']
                    or re.fullmatch('[0-9a-f]{64}',value['sha256']) is None
                    or re.fullmatch('[0-9a-f]{64}',value.get('npy_header_sha256','')) is None
                    or len(value.get('file_identity',[]))!=4
                    or not all(isinstance(item,int) and item>=0 for item in value['file_identity'])
                    or value['file_identity'][2]!=value['bytes']):
                raise ValueError('Historical derived payload evidence is incomplete')
            target=member(root,str(Path(record['relative_directory'])/name),ARRAY_NAMES)
            plan.append((target,value));total+=value['bytes']
    if total!=receipt['derived_bytes_checked']:raise ValueError('Historical byte count differs')
    return root,plan
