"""Three-source pretraining corpus: in-domain six datasets + VisQUIC + CIC-IDS2017."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'outputs/reproduction/pretrain_all.pt'
B = ROOT / 'outputs/reproduction/pretrain_cicids.pt'
OUT = ROOT / 'outputs/reproduction/pretrain_all3.pt'
KEYS = ('x_stat', 'x_seq', 'attn_mask', 'x_bytes', 'bytes_mask')


def main():
    a = torch.load(A, map_location='cpu')
    b = torch.load(B, map_location='cpu')
    splits = {}
    for split in ('train', 'val'):
        splits[split] = {k: torch.cat([a['splits'][split][k], b['splits'][split][k]], 0) for k in KEYS}
        splits[split]['label'] = torch.zeros(splits[split]['x_stat'].shape[0], dtype=torch.long)
    out = dict(splits=splits, protocol=dict(
        selection='in-domain six datasets + VisQUIC + CIC-IDS2017, three branches aligned per flow',
        sources=['in-domain', 'VisQUIC', 'CIC-IDS2017'],
        counts=dict(train=int(splits['train']['x_stat'].shape[0]), val=int(splits['val']['x_stat'].shape[0])),
        normalization='each source standardised with its own scaler; statistics winsorised at +-10'))
    torch.save(out, OUT)
    print(f'train={tuple(splits["train"]["x_stat"].shape)} val={tuple(splits["val"]["x_stat"].shape)} '
          f'-> {OUT} ({OUT.stat().st_size/1e6:.0f} MB)')


if __name__ == '__main__':
    main()
