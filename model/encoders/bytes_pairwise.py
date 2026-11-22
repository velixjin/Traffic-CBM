"""Explicit pairwise-interaction cross-packet encoder.

Packet-level concepts stay exactly as in BytesPacketCrossEncoder.  The three cross-packet
concepts are replaced by explicit bilinear interactions between packet representations:

    h_{p,r}  = mean-pooled token embedding of packet p in region r
    u_{p,r}  = A_r (h_{p,r} - mu_r)
    s_{pq,r} = u_{p,r}^T W_r u_{q,r} / sqrt(D)
    c_r      = phi_r( sum_{p<q} s_{pq,r} )
"""
import torch
from torch import nn

from .bytes_encoder import BytesPacketCrossEncoder, BYTES_VOCAB_SIZE


class BytesPairwiseCrossEncoder(BytesPacketCrossEncoder):
    def __init__(self, *args, dual_dim: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.use_cross
        D = self.model_dim
        self.dual_dim = int(dual_dim)
        # per-region centering + low-rank bilinear interaction
        self.cross_mu = nn.Parameter(torch.zeros(self.num_cross, D))
        self.cross_A = nn.Parameter(torch.eye(D).repeat(self.num_cross, 1, 1) * 0.1)
        self.cross_W = nn.Parameter(torch.zeros(self.num_cross, D, D))
        nn.init.normal_(self.cross_W, std=0.02)
        self.cross_phi = nn.Sequential(
            nn.Linear(self.num_cross, 32), nn.GELU(), nn.Dropout(0.0), nn.Linear(32, self.num_cross)
        )

    def _packet_region_vectors(self, x_bytes, bytes_mask):
        """h[b, p, r, d] = masked mean of the token embeddings in that region."""
        emb = self.byte_embed(x_bytes)                       # B,P,L,D
        B, P, L, D = emb.shape
        G = self.group_len
        mask = bytes_mask.float().unsqueeze(-1) if bytes_mask is not None else torch.ones(B, P, L, 1, device=emb.device)
        regions = []
        for r in range(self.num_cross):
            lo, hi = r * G, (r + 1) * G
            e = emb[:, :, lo:hi] * mask[:, :, lo:hi]
            m = mask[:, :, lo:hi].sum(2).clamp(min=1.0)
            regions.append(e.sum(2) / m)                     # B,P,D
        return torch.stack(regions, 2)                       # B,P,Rcross,D

    def forward(self, x_bytes, bytes_mask=None, c_cond=None):
        B, P, L = x_bytes.shape
        outs = []
        if self.use_packet:
            emb = self.byte_embed(x_bytes)
            flat = emb.reshape(B * P, L, self.model_dim)
            mflat = bytes_mask.reshape(B * P, L) if bytes_mask is not None else None
            h = self.encoder_pkt(flat)
            pooled = self._pool(h, mflat, B * P, L).reshape(B, P, self.model_dim)
            outs.append(self.proj(pooled).squeeze(-1))
        if self.use_cross:
            hv = self._packet_region_vectors(x_bytes, bytes_mask)          # B,P,R,D
            u = torch.einsum('bpRd,Rde->bpRe', hv - self.cross_mu.unsqueeze(0).unsqueeze(0), self.cross_A)
            scale = 1.0 / (self.dual_dim ** 0.5)
            scores = []
            for r in range(self.num_cross):
                W = self.cross_W[r]
                s = torch.einsum('bpd,de,bqe->bpq', u[:, :, r], W, u[:, :, r]) * scale
                tri = torch.triu(torch.ones(P, P, device=s.device), diagonal=1).bool()
                scores.append(s[:, tri].sum(1))
            cross = self.cross_phi(torch.stack(scores, 1))                 # B,R
            outs.append(cross)
        return torch.cat(outs, 1)
