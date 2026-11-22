#!/usr/bin/env python3
"""
Cross-dataset self-supervised pretraining for seq + bytes encoders.

Pretraining tasks
-----------------
1. Sequence masking reconstruction:
   mask packet-level seq features and reconstruct them with SeqConceptEncoder's
   four group-specific transformers.
2. Bytes masked-token reconstruction:
   mask bigram tokens and reconstruct them with BytesPacketCrossEncoder's
   packet/cross transformers.

The checkpoint stores:
  - seq_encoder
  - bytes_encoder
which can later be loaded by train/train.py via --pretrain_ckpt.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from datasets.traffic_dataset import TOKEN_MASK, TrafficDataset, collate_fn, fit_scalers
from model.encoders.sequence_encoder import SeqConceptEncoder
from model.encoders.bytes_encoder import BYTES_VOCAB_SIZE, BytesPacketCrossEncoder


DEFAULT_DATASETS = [
    "USTC-TFC2016",
    "ISCX-VPN-App",
    "ISCX-VPN-Service",
    "CSTNET-TLS13",
    "CipherSpectrum-AES128",
    "CipherSpectrum-mix",
]


def _load_model_cfg(cfg_path: str) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("model", {})


def _build_concat_dataset(
    datasets_root: str,
    dataset_names: Sequence[str],
    split: str,
    max_seq_len: int,
    bytes_len: int,
    bytes_per_packet: bool,
    bytes_pkt_len: int,
    num_packets: int,
) -> ConcatDataset:
    datasets = []
    for name in dataset_names:
        data_dir = os.path.join(datasets_root, name)
        ids_txt = os.path.join(data_dir, f"{split}_ids.txt")
        if not os.path.exists(ids_txt):
            continue
        print(f"[build] {split}: preparing {name}", flush=True)
        _, seq_scaler = fit_scalers(
            data_dir,
            os.path.join(data_dir, "train_ids.txt"),
            fit_stat=False,
            fit_seq=True,
        )
        print(f"[build] {split}: scaler ready for {name}", flush=True)
        ds = TrafficDataset(
            ids_txt=ids_txt,
            data_dir=data_dir,
            use_stat=False,
            use_seq=True,
            use_bytes=True,
            max_seq_len=max_seq_len,
            bytes_len=bytes_len,
            bytes_per_packet=bytes_per_packet,
            bytes_pkt_len=bytes_pkt_len,
            num_packets=num_packets,
            stat_scaler=None,
            seq_scaler=seq_scaler,
        )
        print(f"[build] {split}: dataset ready for {name}, n={len(ds)}", flush=True)
        datasets.append(ds)
    if not datasets:
        raise RuntimeError(f"No datasets found for split={split}")
    return ConcatDataset(datasets)


class SeqBytesPretrainer(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        seq_model_dim: int,
        seq_num_heads: int,
        seq_layers: int,
        seq_pooling: str,
        bytes_model_dim: int,
        bytes_num_heads: int,
        bytes_layers: int,
        bytes_pooling: str,
        bytes_pkt_len: int,
        num_packets: int,
        num_groups: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_encoder = SeqConceptEncoder(
            seq_len=max_seq_len,
            model_dim=seq_model_dim,
            num_heads=seq_num_heads,
            num_layers=seq_layers,
            dropout=dropout,
            pooling=seq_pooling,
        )
        self.bytes_encoder = BytesPacketCrossEncoder(
            num_packets=num_packets,
            num_cross=num_groups,
            bytes_pkt_len=bytes_pkt_len,
            model_dim=bytes_model_dim,
            num_heads=bytes_num_heads,
            num_layers=bytes_layers,
            dropout=dropout,
            pooling=bytes_pooling,
            cross_only=False,
        )
        self.seq_recon_heads = nn.ModuleList(
            [
                nn.Linear(seq_model_dim, len(idx))
                for idx, _ in SeqConceptEncoder.GROUPS
            ]
        )
        self.bytes_recon_head = nn.Linear(bytes_model_dim, BYTES_VOCAB_SIZE)

    def _encode_group_tokens(self, enc, x_group, attn_mask):
        B, N, _ = x_group.shape
        x = enc.input_proj(x_group)
        if enc.pooling == "cls":
            cls = enc.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + enc.pos_embed.unsqueeze(0)
            cls_mask = torch.ones((B, 1), device=x_group.device, dtype=attn_mask.dtype)
            full_mask = torch.cat([cls_mask, attn_mask], dim=1)
            kpm = (full_mask == 0).bool()
            h = enc.encoder(x, src_key_padding_mask=kpm)
            h = enc._safe_to_dense(h, B, N + 1)
            return h[:, 1:, :]
        x = x + enc.pos_embed.unsqueeze(0)
        kpm = (attn_mask == 0).bool()
        h = enc.encoder(x, src_key_padding_mask=kpm)
        h = enc._safe_to_dense(h, B, N)
        return h

    def _encode_bytes_packet_tokens(self, x_flat, mask_flat):
        Bn, L = x_flat.shape
        x = self.bytes_encoder.byte_embed(x_flat)
        if self.bytes_encoder.pooling == "cls":
            cls = self.bytes_encoder.cls_pkt.expand(Bn, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.bytes_encoder.pos_pkt.unsqueeze(0)
            cls_mask = torch.ones((Bn, 1), device=x_flat.device, dtype=mask_flat.dtype)
            full_mask = torch.cat([cls_mask, mask_flat], dim=1)
            kpm = (full_mask == 0).bool()
            h = self.bytes_encoder.encoder_pkt(x, src_key_padding_mask=kpm)
            h = self.bytes_encoder._safe_to_dense(h, Bn, L + 1)
            return h[:, 1:, :]
        x = x + self.bytes_encoder.pos_pkt.unsqueeze(0)
        kpm = (mask_flat == 0).bool()
        h = self.bytes_encoder.encoder_pkt(x, src_key_padding_mask=kpm)
        h = self.bytes_encoder._safe_to_dense(h, Bn, L)
        return h

    def _encode_bytes_cross_tokens(self, x_c, m_c):
        B, L = x_c.shape
        x = self.bytes_encoder.byte_embed(x_c)
        if self.bytes_encoder.pooling == "cls":
            cls = self.bytes_encoder.cls_cross.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.bytes_encoder.pos_cross.unsqueeze(0)
            cls_mask = torch.ones((B, 1), device=x_c.device, dtype=m_c.dtype)
            full_mask = torch.cat([cls_mask, m_c], dim=1)
            kpm = (full_mask == 0).bool()
            h = self.bytes_encoder.encoder_cross(x, src_key_padding_mask=kpm)
            h = self.bytes_encoder._safe_to_dense(h, B, L + 1)
            return h[:, 1:, :]
        x = x + self.bytes_encoder.pos_cross.unsqueeze(0)
        kpm = (m_c == 0).bool()
        h = self.bytes_encoder.encoder_cross(x, src_key_padding_mask=kpm)
        h = self.bytes_encoder._safe_to_dense(h, B, L)
        return h

    def forward(self, batch, seq_mask_ratio: float = 0.15, bytes_mask_ratio: float = 0.15):
        x_seq = batch["x_seq"].float()
        attn_mask = batch["attn_mask"].float()
        x_bytes = batch["x_bytes"].long()
        bytes_mask = batch["bytes_mask"].long()
        device = x_seq.device

        total_seq_loss = torch.tensor(0.0, device=device)
        total_bytes_loss = torch.tensor(0.0, device=device)

        # seq reconstruction
        for head, ((idx, _), enc) in zip(self.seq_recon_heads, zip(SeqConceptEncoder.GROUPS, self.seq_encoder.group_encoders)):
            x_group = x_seq[:, :, idx]
            valid = attn_mask > 0
            mask = (torch.rand_like(attn_mask) < seq_mask_ratio) & valid
            x_masked = x_group.clone()
            x_masked[mask.unsqueeze(-1).expand_as(x_masked)] = 0.0
            h_tokens = self._encode_group_tokens(enc, x_masked, attn_mask)
            pred = head(h_tokens)
            if mask.any():
                total_seq_loss = total_seq_loss + F.mse_loss(
                    pred[mask],
                    x_group[mask],
                )

        # bytes reconstruction: packet path
        valid_bytes = bytes_mask > 0
        mask_bytes = (torch.rand_like(bytes_mask.float()) < bytes_mask_ratio) & valid_bytes
        x_bytes_masked = x_bytes.clone()
        x_bytes_masked[mask_bytes] = TOKEN_MASK

        B, N, L = x_bytes.shape
        x_flat = x_bytes_masked.reshape(B * N, L)
        mask_flat = bytes_mask.reshape(B * N, L)
        target_flat = x_bytes.reshape(B * N, L)
        mlm_flat = mask_bytes.reshape(B * N, L)
        h_pkt = self._encode_bytes_packet_tokens(x_flat, mask_flat)
        logits_pkt = self.bytes_recon_head(h_pkt)
        if mlm_flat.any():
            total_bytes_loss = total_bytes_loss + F.cross_entropy(
                logits_pkt[mlm_flat],
                target_flat[mlm_flat],
            )

        # bytes reconstruction: cross path
        G = self.bytes_encoder.num_cross
        GL = self.bytes_encoder.group_len
        x_g = x_bytes_masked.reshape(B, N, G, GL)
        mask_g = bytes_mask.reshape(B, N, G, GL)
        target_g = x_bytes.reshape(B, N, G, GL)
        mlm_g = mask_bytes.reshape(B, N, G, GL)
        cross_losses = []
        for g in range(G):
            x_c = x_g[:, :, g, :].reshape(B, N * GL)
            m_c = mask_g[:, :, g, :].reshape(B, N * GL)
            t_c = target_g[:, :, g, :].reshape(B, N * GL)
            mlm_c = mlm_g[:, :, g, :].reshape(B, N * GL)
            h_cross = self._encode_bytes_cross_tokens(x_c, m_c)
            logits_cross = self.bytes_recon_head(h_cross)
            if mlm_c.any():
                cross_losses.append(F.cross_entropy(logits_cross[mlm_c], t_c[mlm_c]))
        if cross_losses:
            total_bytes_loss = total_bytes_loss + torch.stack(cross_losses).mean()

        total_loss = total_seq_loss + total_bytes_loss
        return total_loss, {
            "seq_loss": float(total_seq_loss.detach().cpu()),
            "bytes_loss": float(total_bytes_loss.detach().cpu()),
        }


@torch.no_grad()
def evaluate(model, loader, device, seq_mask_ratio, bytes_mask_ratio):
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, _ = model(batch, seq_mask_ratio=seq_mask_ratio, bytes_mask_ratio=bytes_mask_ratio)
        bs = batch["label"].size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(total, 1)


def main():
    ap = argparse.ArgumentParser("Pretrain seq + bytes encoders")
    ap.add_argument("--config", default=os.path.join(ROOT, "train", "config.json"))
    ap.add_argument("--datasets_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--seq_mask_ratio", type=float, default=0.15)
    ap.add_argument("--bytes_mask_ratio", type=float, default=0.15)

    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    model_cfg = _load_model_cfg(args.config)
    seq_cfg = model_cfg.get("seq", {})
    bytes_cfg = model_cfg.get("bytes", {})
    fusion_mode = model_cfg.get("fusion_mode", "6+4+5+3")
    dropout = float(model_cfg.get("dropout", 0.1))
    max_seq_len = int(seq_cfg.get("max_len", 100))
    bytes_len = int(bytes_cfg.get("len", 320))
    bytes_per_packet = bool(bytes_cfg.get("per_packet", False) or fusion_mode in ("6+4+5", "6+4+5+3", "6+4+3"))
    bytes_pkt_len = int(bytes_cfg.get("pkt_len", 63))
    num_packets = int(bytes_cfg.get("num_packets", 5))
    num_groups = int(bytes_cfg.get("num_groups", 3))

    print(f"[pretrain] start building train datasets: {','.join(args.datasets)}", flush=True)
    train_ds = _build_concat_dataset(
        args.datasets_root, args.datasets, "train",
        max_seq_len, bytes_len, bytes_per_packet, bytes_pkt_len, num_packets
    )
    print(f"[pretrain] start building val datasets: {','.join(args.datasets)}", flush=True)
    val_ds = _build_concat_dataset(
        args.datasets_root, args.datasets, "val",
        max_seq_len, bytes_len, bytes_per_packet, bytes_pkt_len, num_packets
    )
    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqBytesPretrainer(
        max_seq_len=max_seq_len,
        seq_model_dim=int(seq_cfg.get("model_dim", 64)),
        seq_num_heads=int(seq_cfg.get("num_heads", 4)),
        seq_layers=int(seq_cfg.get("num_layers", 2)),
        seq_pooling=str(seq_cfg.get("pooling", "mean")),
        bytes_model_dim=int(bytes_cfg.get("model_dim", 64)),
        bytes_num_heads=int(bytes_cfg.get("num_heads", 4)),
        bytes_layers=int(bytes_cfg.get("num_layers", 2)),
        bytes_pooling=str(bytes_cfg.get("pooling", "mean")),
        bytes_pkt_len=bytes_pkt_len,
        num_packets=num_packets,
        num_groups=num_groups,
        dropout=dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val = float("inf")
    best_state = None
    t_start = time.time()

    print(f"[pretrain] datasets={','.join(args.datasets)} train={len(train_ds)} val={len(val_ds)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        seq_loss_sum = 0.0
        bytes_loss_sum = 0.0
        t0 = time.time()
        for step, batch in enumerate(dl_train, 1):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model(batch, seq_mask_ratio=args.seq_mask_ratio, bytes_mask_ratio=args.bytes_mask_ratio)
            loss.backward()
            optimizer.step()

            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            seq_loss_sum += stats["seq_loss"] * bs
            bytes_loss_sum += stats["bytes_loss"] * bs
            if step % args.log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"  [pretrain] step={step}/{len(dl_train)} loss={total_loss/max(total,1):.4f} "
                    f"seq={seq_loss_sum/max(total,1):.4f} bytes={bytes_loss_sum/max(total,1):.4f} "
                    f"speed={total/max(elapsed,1e-6):.0f} samples/s",
                    flush=True,
                )

        val_loss = evaluate(model, dl_val, device, args.seq_mask_ratio, args.bytes_mask_ratio)
        scheduler.step()
        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"train_loss={total_loss/max(total,1):.4f} | val_loss={val_loss:.4f} | {dt:.1f}s",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "seq_encoder": {k: v.detach().cpu() for k, v in model.seq_encoder.state_dict().items()},
                "bytes_encoder": {k: v.detach().cpu() for k, v in model.bytes_encoder.state_dict().items()},
                "meta": {
                    "datasets": list(args.datasets),
                    "max_seq_len": max_seq_len,
                    "bytes_pkt_len": bytes_pkt_len,
                    "num_packets": num_packets,
                    "num_groups": num_groups,
                    "seq_mask_ratio": args.seq_mask_ratio,
                    "bytes_mask_ratio": args.bytes_mask_ratio,
                },
            }

    if best_state is not None:
        ckpt = os.path.join(args.output, "best_pretrain.pt")
        torch.save(best_state, ckpt)
        print(f"[save] best pretrain -> {ckpt}", flush=True)
        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    total_sec = int(time.time() - t_start)
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    print(f"[done] total pretrain time: {h:02d}h {m:02d}m {s:02d}s", flush=True)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
