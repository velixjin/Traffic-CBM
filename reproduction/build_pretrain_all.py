"""Assemble a three-branch pretraining corpus: x_stat + x_seq + x_bytes."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / 'outputs/reproduction/clean_pretrain_data.pt'
VIS = ROOT / 'outputs/reproduction/pretrain_visquic.pt'
OUT = ROOT / 'outputs/reproduction/pretrain_all.pt'


def main():
    corpus = torch.load(CORPUS, map_location='cpu')
    protocol = corpus['protocol']
    per_split = {'train': [], 'val': []}
    for entry in protocol['datasets']:
        dataset = entry['dataset']
        cache = torch.load(entry['cache'], map_location='cpu', mmap=True)
        for split in ('train', 'val'):
            ids = entry['splits'][split]['flow_ids']
            index = {fid: i for i, fid in enumerate(cache['splits'][split]['flow_ids'])}
            rows = np.stack([cache['splits'][split]['x_stat'][index[fid]].numpy() for fid in ids])
            per_split[split].append(rows)
            print(f'  {dataset:24s} {split:5s} {rows.shape[0]:7d}', flush=True)
        del cache
    stat_train = np.concatenate(per_split['train'], 0)
    stat_val = np.concatenate(per_split['val'], 0)
    assert stat_train.shape[0] == corpus['splits']['train']['x_seq'].shape[0]
    assert stat_val.shape[0] == corpus['splits']['val']['x_seq'].shape[0]

    vis = torch.load(VIS, map_location='cpu')
    # Winsorise standardised statistics when assembling the pretraining corpus.
    stat_train = np.clip(stat_train, -10.0, 10.0)
    stat_val = np.clip(stat_val, -10.0, 10.0)
    vis_stat_train = np.clip(vis['splits']['train']['x_stat'].numpy(), -10.0, 10.0)
    vis_stat_val = np.clip(vis['splits']['val']['x_stat'].numpy(), -10.0, 10.0)
    splits = {}
    for split in ('train', 'val'):
        splits[split] = dict(
            x_stat=torch.cat([torch.from_numpy(stat_train if split == 'train' else stat_val),
                              torch.from_numpy(vis_stat_train if split == 'train' else vis_stat_val)], 0),
            x_seq=torch.cat([corpus['splits'][split]['x_seq'], vis['splits'][split]['x_seq']], 0),
            attn_mask=torch.cat([corpus['splits'][split]['attn_mask'], vis['splits'][split]['attn_mask']], 0),
            x_bytes=torch.cat([corpus['splits'][split]['x_bytes'], vis['splits'][split]['x_bytes']], 0),
            bytes_mask=torch.cat([corpus['splits'][split]['bytes_mask'], vis['splits'][split]['bytes_mask']], 0),
            label=torch.zeros(corpus['splits'][split]['x_seq'].shape[0] + vis['splits'][split]['x_seq'].shape[0],
                              dtype=torch.long))
    out = dict(splits=splits, protocol=dict(
        selection='in-domain six-dataset train splits plus external VisQUIC; three branches aligned per flow',
        datasets=[e['dataset'] for e in protocol['datasets']] + ['VisQUIC'],
        stat_rows=int(splits['train']['x_stat'].shape[0] + splits['val']['x_stat'].shape[0]),
        normalization='x_stat standardised per dataset and clipped to [-10, 10]; x_seq standardised per source'))
    torch.save(out, OUT)
    print(f'train={tuple(splits["train"]["x_stat"].shape)} val={tuple(splits["val"]["x_stat"].shape)} -> {OUT}')


if __name__ == '__main__':
    main()
