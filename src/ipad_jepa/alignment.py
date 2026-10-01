"""Explicit bounded-offset sensitivity for the three unresolved R02 annotations."""
import argparse
import json
from pathlib import Path
import numpy as np
from ipad_jepa.temporal import common_mask
from ipad_jepa.audit import digest


POLICY='Matched lengths: index alignment. Unresolved +/-1 length: consensus over label indices t-1,t,t+1; disagreements/out-of-bounds unknown. Assumes index offset stays within +/-1, not proven exact alignment.'


def align(labels,frames):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.all(np.isin(labels,[0,1])):
        raise ValueError('Expected binary labels')
    if len(labels)==frames:
        return labels.astype(np.int8),np.ones(frames,dtype=bool),{0:labels.astype(np.int8)}
    if abs(len(labels)-frames)!=1:
        raise ValueError('Offset policy only supports audited one-frame length mismatches')
    candidates={}
    for offset in [-1,0,1]:
        ids=np.arange(frames)+offset
        available=(ids>=0)&(ids<len(labels))
        values=np.full(frames,-1,dtype=np.int8)
        values[available]=labels[ids[available]]
        candidates[offset]=values
    matrix=np.stack(list(candidates.values()))
    known=(matrix>=0).all(0)&(matrix==matrix[0]).all(0)
    consensus=np.where(known,matrix[0],-1).astype(np.int8)
    return consensus,known,candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('results/stage00/manifest.json'))
    parser.add_argument('--out',type=Path,default=Path('results/stage00/alignment_policy.json'))
    args=parser.parse_args(); rows=json.loads(args.manifest.read_text())['sequences']
    result=[]
    for row in rows:
        if row['partition']!='testing' or row['alignment_status']=='matched': continue
        path=args.data_root/row['label_file']
        if digest(path)!=row['label_sha256']: raise ValueError('Audited label file changed')
        values=np.load(path,allow_pickle=False); labels,known,candidates=align(values,row['frames'])
        common=common_mask(row['frames'])
        result.append({'device':row['device'],'sequence':row['sequence'],'frames':row['frames'],'label_count':len(values),
                       'label_sha256':row['label_sha256'],'unknown_indices':np.flatnonzero(~known).tolist(),
                       'unknown_common_indices':np.flatnonzero(common&~known).tolist(),
                       'common_frames':int(common.sum()),'retained_common_frames':int((common&known).sum()),
                       'candidate_offsets':list(candidates),'status':'bounded_offset_policy_not_exact_alignment'})
    output={'exact_alignment_confirmed':False,'policy':POLICY,'source':'Audited raw labels; original files untouched',
            'mismatches':result,'unknown_common_total':sum(len(r['unknown_common_indices']) for r in result)}
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
