"""Masked pretraining for all three branches: statistics, sequence and bytes."""
import argparse, json, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reproduction'))
sys.path.insert(0, str(ROOT / 'scripts'))
from pretrain_fast import MaskedProjectionPretrainer                     # noqa: E402
from model.encoders.stat_encoder import StatConceptEncoder               # noqa: E402
from checkpoint_io import atomic_torch_save                              # noqa: E402


class AllBranchPretrainer(MaskedProjectionPretrainer):
    def __init__(self, stat_dim, concept_groups, stat_hidden=(128, 128, 128, 64), dropout=.2, **kwargs):
        super().__init__(dropout=dropout, **kwargs)
        self.stat_encoder = StatConceptEncoder(stat_dim=stat_dim, num_concepts=6, concept_mode='grouped',
                                               concept_groups=concept_groups, hidden_dims=stat_hidden,
                                               dropout=dropout)
        self.stat_decoder = nn.Linear(6, stat_dim)

    def forward(self, batch, seq_mask_ratio=.15, bytes_mask_ratio=.15, stat_mask_ratio=.15, branch_drop=0.0):
        batch = dict(batch)
        if branch_drop > 0:
            n = batch['x_stat'].shape[0]
            drop_stat = torch.rand(n, device=batch['x_stat'].device) < branch_drop
            drop_seq = torch.rand(n, device=batch['x_stat'].device) < branch_drop
            drop_bytes = torch.rand(n, device=batch['x_stat'].device) < branch_drop
            # A dropped branch contributes no reconstruction loss: the masks are cleared
            # so the masked-reconstruction terms skip those rows entirely.
            if drop_seq.any():
                batch['x_seq'] = batch['x_seq'].clone(); batch['x_seq'][drop_seq] = 0
                batch['attn_mask'] = batch['attn_mask'].clone(); batch['attn_mask'][drop_seq] = 0
            if drop_bytes.any():
                batch['x_bytes'] = batch['x_bytes'].clone(); batch['x_bytes'][drop_bytes] = 65536
                batch['bytes_mask'] = batch['bytes_mask'].clone(); batch['bytes_mask'][drop_bytes] = 0
        else:
            drop_stat = torch.zeros(batch['x_stat'].shape[0], dtype=torch.bool, device=batch['x_stat'].device)
        loss, parts = super().forward(batch, seq_mask_ratio, bytes_mask_ratio)
        # Standardised statistics contain extreme outliers (max 2.2e9 in column 68);
        # clip so that a few rows cannot dominate the reconstruction objective.
        x = batch['x_stat'].float().clamp(-10.0, 10.0)
        chosen = (torch.rand_like(x) < stat_mask_ratio) & (~drop_stat).unsqueeze(1)
        altered = x.clone(); altered[chosen] = 0.0
        concepts = self.stat_encoder(altered)
        recon = self.stat_decoder(concepts)
        stat_loss = F.mse_loss(recon[chosen], x[chosen]) if chosen.any() else torch.zeros((), device=x.device)
        parts['stat_loss'] = float(stat_loss.detach())
        return loss + stat_loss, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--meta', type=Path, required=True, help='cache providing concept groups')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=96)
    ap.add_argument('--limit', type=int, default=0, help='debug: cap training rows')
    ap.add_argument('--mask-ratio', type=float, default=0.15,
                    help='masked-reconstruction ratio for every branch')
    ap.add_argument('--branch-drop', type=float, default=0.0,
                    help='probability of dropping an entire evidence branch during pretraining')
    a = ap.parse_args()
    torch.manual_seed(42)
    cache = torch.load(a.data, map_location='cpu')
    meta = torch.load(str(a.meta), map_location='cpu', mmap=True)['meta']
    groups = meta['concept_groups']
    model = AllBranchPretrainer(stat_dim=meta['stat_dim'], concept_groups=groups,
                                max_seq_len=100, seq_model_dim=64, seq_num_heads=4, seq_layers=2,
                                seq_pooling='cls', bytes_model_dim=64, bytes_num_heads=4, bytes_layers=2,
                                bytes_pooling='cls', bytes_pkt_len=63, num_packets=5, num_groups=3,
                                dropout=.2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    train = {k: v.cuda() for k, v in cache['splits']['train'].items()}
    val = {k: v.cuda() for k, v in cache['splits']['val'].items()}
    if a.limit:
        train = {k: v[:a.limit] for k, v in train.items()}
    n = train['x_stat'].shape[0]
    a.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(a.epochs):
        model.train(); total = 0.0; start = time.perf_counter()
        gen = torch.Generator().manual_seed(42 + epoch)
        order = torch.randperm(n, generator=gen).cuda()
        for offset in range(0, n, a.batch_size):
            take = order[offset:offset + a.batch_size]
            batch = {k: v[take] for k, v in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model(batch, seq_mask_ratio=a.mask_ratio, bytes_mask_ratio=a.mask_ratio, stat_mask_ratio=a.mask_ratio, branch_drop=a.branch_drop)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(take)
        model.eval()
        with torch.inference_mode():
            vl = 0.0
            for offset in range(0, val['x_stat'].shape[0], a.batch_size):
                batch = {k: v[offset:offset + a.batch_size] for k, v in val.items()}
                loss, _ = model(batch, seq_mask_ratio=a.mask_ratio, bytes_mask_ratio=a.mask_ratio, stat_mask_ratio=a.mask_ratio, branch_drop=a.branch_drop)
                vl += float(loss) * batch['x_stat'].shape[0]
        row = dict(epoch=epoch + 1, train_loss=total / n, val_loss=vl / val['x_stat'].shape[0],
                   seconds=time.perf_counter() - start, parts=parts)
        history.append(row)
        print('[stat-pretrain]', json.dumps(row), flush=True)
        atomic_torch_save(dict(stat_encoder=model.stat_encoder.state_dict(),
                               seq_encoder=model.seq_encoder.state_dict(),
                               bytes_encoder=model.bytes_encoder.state_dict()), a.output / 'best_pretrain.pt')
        (a.output / 'history.json').write_text(json.dumps(history, indent=2))
    (a.output / 'result.json').write_text(json.dumps(dict(epochs=a.epochs, history=history), indent=2))
    print('[done]', a.output, flush=True)


if __name__ == '__main__':
    main()
