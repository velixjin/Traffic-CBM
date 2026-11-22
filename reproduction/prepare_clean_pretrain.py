"""Globally exclude held-out packet duplicates, then refit pretraining scalers."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rebuttal_controls import fit_array_scaler
from datasets.traffic_dataset import SEQ_CONT_IDX

OUT=ROOT/'outputs/reproduction'
DATASETS=['CipherSpectrum-AES128','CSTNET-TLS13','ISCX-VPN-App','ISCX-VPN-Service','CipherSpectrum-mix','USTC-TFC2016']


def main():
    dest=OUT/'clean_pretrain_data.pt'
    if dest.exists():raise FileExistsError(dest)
    required=[OUT/'cache'/(d+'_corrected_full.pt') for d in DATASETS]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing aligned caches: {missing}')
    duplicates=json.loads((OUT/'source_audit/results.json').read_text())['confirmed_duplicates']
    banned=defaultdict(set);deduplicated=defaultdict(set)
    for group in duplicates:
        flows=group['flows'];splits={r['split'] for r in flows}
        for split in ['train','val']:
            members=[(r['dataset'],r['flow_id']) for r in flows if r['split']==split]
            if ('test' in splits) or (split=='train' and 'val' in splits):
                banned[split].update(members)
            elif len(members)>1:
                deduplicated[split].update(sorted(members)[1:])
    pieces={s:defaultdict(list) for s in ['train','val']};manifest=[]
    for dataset,path in zip(DATASETS,required):
        cache=torch.load(path,map_location='cpu');selected={}
        for split in ['train','val']:
            ids=cache['splits'][split]['flow_ids']
            selected[split]=[i for i,f in enumerate(ids) if (dataset,f) not in banned[split]|deduplicated[split]]
        train=cache['splits']['train'];pick=selected['train']
        raw=train['x_seq'][pick].double().numpy()
        raw[:,:,SEQ_CONT_IDX]=raw[:,:,SEQ_CONT_IDX]*cache['seq_scaler'].std+cache['seq_scaler'].mean
        scaler=fit_array_scaler(raw[:,:,SEQ_CONT_IDX][train['attn_mask'][pick].numpy()>0])
        entry=dict(dataset=dataset,cache=str(path),splits={})
        for split in ['train','val']:
            values=cache['splits'][split];pick=selected[split]
            seq=values['x_seq'][pick].double().numpy()
            seq[:,:,SEQ_CONT_IDX]=seq[:,:,SEQ_CONT_IDX]*cache['seq_scaler'].std+cache['seq_scaler'].mean
            seq[:,:,SEQ_CONT_IDX]=scaler.transform(seq[:,:,SEQ_CONT_IDX])
            seq[values['attn_mask'][pick].numpy()==0]=0
            pieces[split]['x_seq'].append(torch.from_numpy(seq.astype(np.float32)))
            for k in ['attn_mask','x_bytes','bytes_mask']:
                pieces[split][k].append(values[k][pick])
            ids=[values['flow_ids'][i] for i in pick]
            entry['splits'][split]=dict(original=len(values['flow_ids']),selected=len(pick),
                excluded_heldout_duplicates=sum((dataset,f) in banned[split] for f in values['flow_ids']),
                removed_repeated_training_views=sum((dataset,f) in deduplicated[split] for f in values['flow_ids']),
                sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),flow_ids=ids)
        entry['sequence_scaler']=dict(mean=scaler.mean.tolist(),std=scaler.std.tolist(),fitted_rows=scaler.n)
        manifest.append(entry);print('[clean data]',dataset,{s:entry['splits'][s]['selected'] for s in ['train','val']},flush=True)
    packed={s:{k:torch.cat(v) for k,v in p.items()} for s,p in pieces.items()}
    protocol=dict(selection='Exclude all train flows packet-identical to ANY dataset validation/test; exclude validation flows identical to ANY test; deduplicate remaining cross-dataset views within each pretraining split.',
        sequence='Link-type-aware corrected features; normalizers refit only on the remaining pretraining train rows per dataset.',
        duplicate_audit='SHA-256 verification over all captured packet bytes for candidates matched on the first 32 packets.',
        datasets=manifest,
        sizes={s:len(v['x_seq']) for s,v in packed.items()})
    torch.save(dict(splits=packed,protocol=protocol),dest)
    with (OUT/'clean_pretrain_data.json').open('x') as f:json.dump(protocol,f,indent=2)
    print('[ready]',dest,protocol['sizes'],flush=True)


if __name__=='__main__':main()
