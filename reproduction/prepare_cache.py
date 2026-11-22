"""Create independent, aligned tensor caches; existing datasets remain read-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import os
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
FEATURE_ROOT=Path(os.environ.get('TRAFFIC_CBM_FEATURE_ROOT', ROOT/'data'))
FLOW_ROOT=Path(os.environ.get('TRAFFIC_CBM_FLOW_ROOT', ROOT/'dataset'))
sys.path.insert(0,str(ROOT))
from rebuttal_controls import INPUTS,fit_array_scaler
from datasets.traffic_dataset import TrafficDataset,SEQ_CONT_IDX,collate_fn
from model.encoders.stat_groups import build_cic_groups
from packet_sequence import extract_correct


def corrected_job(item):
    fid,path=item
    try:
        a=extract_correct(path,100)
        if not len(a) or not np.isfinite(a).all():
            raise ValueError('Empty or nonfinite decoded sequence')
        return fid,a
    except Exception as error:
        raise RuntimeError(f'{fid}: {error}') from error


def prepare(dataset,output,budget,corrected,workers,aligned_ids=None):
    source=FEATURE_ROOT/dataset
    output.mkdir(parents=True,exist_ok=True)
    tag=dataset+('_corrected' if corrected else '_legacy')+('_full' if budget==0 else f'_n{budget}')
    destination=output/(tag+'.pt')
    if destination.exists():
        raise FileExistsError(destination)
    ds=TrafficDataset(str(source/'test_ids.txt'),str(source),bytes_per_packet=True)
    available=set(ds.id2stat)&set(ds.id2seq)&set(ds.id2bytes)
    splits = {}
    for split in ('train', 'val', 'test'):
        ids = [line.strip() for line in ((aligned_ids or source) / f'{split}_ids.txt').read_text().splitlines() if line.strip()]
        if aligned_ids is not None and not set(ids).issubset(available):
            raise ValueError(f'{split}: registered flows missing from extracted features')
        splits[split] = [flow_id for flow_id in ids if flow_id in available]
        if not splits[split]:
            raise ValueError(f'{split}: no flows with all three feature sources')
    original_sizes={s:len(ids) for s,ids in splits.items()}
    if budget and len(splits['train'])>budget:
        from sklearn.model_selection import StratifiedShuffleSplit
        ids=splits['train'];y=[ds.label2id[f.split('/',1)[0]] for f in ids]
        selected,_=next(StratifiedShuffleSplit(n_splits=1,train_size=budget,random_state=20261026).split(y,y))
        splits['train']=[ids[i] for i in sorted(selected)]
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not (set(splits[a])&set(splits[b]))
    if corrected:
        jobs=[(fid,str(FLOW_ROOT/dataset/'flows_filtered'/(fid+'.pcap'))) for ids in splits.values() for fid in ids]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n,(fid,sequence) in enumerate(pool.map(corrected_job,jobs,chunksize=32)):
                ds.id2seq[fid]=sequence
                if n%5000==0:
                    print(f'[corrected] {dataset} {n}/{len(jobs)}',flush=True)
    stats=np.stack([np.nan_to_num(ds.id2stat[f].copy(),nan=0,posinf=0,neginf=0) for f in splits['train']])
    seq=np.concatenate([ds.id2seq[f][:100,SEQ_CONT_IDX] for f in splits['train']])
    if not np.isfinite(seq).all():
        raise ValueError('Nonfinite sequence features')
    ds.stat_scaler,ds.seq_scaler=fit_array_scaler(stats),fit_array_scaler(seq)
    packed={}
    for split,ids in splits.items():
        ds.flow_ids=ids
        pieces={k:[] for k in INPUTS+['label']}
        for start in range(0,len(ids),256):
            batch=collate_fn([ds[i] for i in range(start, min(start + 256, len(ids)))])
            for k in pieces:
                pieces[k].append(batch[k])
        packed[split]={k:torch.cat(v) for k,v in pieces.items()}
        packed[split]['flow_ids']=ids
        print(f'[cache] {tag} {split} {len(ids)}',flush=True)
    meta=dict(dataset=dataset,preprocessing='linktype_corrected_sequence' if corrected else 'historical_features',
        train_budget=budget or None,original_aligned_sizes=original_sizes,subset_seed=20261026,
        sizes={s:len(v) for s,v in splits.items()},stat_dim=ds.stat_dim,stat_columns=ds.stat_feat_cols,
        concept_groups=build_cic_groups(ds.stat_feat_cols,list(range(len(ds.stat_feat_cols)))),class_mapping=ds.label2id,
        normalization='selected training flows only',alignment='registered flow IDs' if aligned_ids else 'intersection of the three Traffic-CBM feature sources',
        split_hashes={s:hashlib.sha256('\n'.join(ids).encode()).hexdigest() for s,ids in splits.items()})
    torch.save(dict(meta=meta,splits=packed,stat_scaler=ds.stat_scaler,seq_scaler=ds.seq_scaler),destination)
    with destination.with_suffix('.json').open('x') as f:
        json.dump(meta,f,indent=2)
    print(destination,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--budget',type=int,default=0)
    ap.add_argument('--corrected',action='store_true');ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--aligned-ids',type=Path,help='Directory containing ordered train_ids.txt, val_ids.txt, and test_ids.txt')
    args=ap.parse_args();prepare(args.dataset,args.output,args.budget,args.corrected,args.workers,args.aligned_ids)
