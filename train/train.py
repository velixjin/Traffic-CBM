#!/usr/bin/env python3
"""
train/train.py - Training script for TrafficCBM.

Usage:
  python -u train/train.py \
    --data_dir  data/USTC-TFC2016 \
    --output    outputs/USTC-TFC2016 \
    --use_stat --use_seq --use_bytes \
    --epochs 40 --batch_size 256

  # tabcbm stat mode
  python -u train/train.py \
    --data_dir data/USTC-TFC2016 \
    --use_stat --concept_mode tabcbm \
    --sparsity_weight 1e-3
"""
import os
import sys
import argparse
import time
import json
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.traffic_dataset import (
    FEATURE_SCALERS_FILENAME,
    TrafficDataset,
    collate_fn,
    fit_scalers,
    save_feature_scalers,
)
from model.encoders.stat_groups import build_cic_groups, GROUP_NAMES
from model.traffic_cbm import TrafficCBM, build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_concept_mask(model, concept_mode, concept_groups, header, feat_idx, output_dir):
    """Save concept mask to concept_mask.md (Markdown table) in the output directory.

    grouped: hard 0/1 matrix with group names as column headers.
    tabcbm:  sigmoid of learned concept_mask, columns named concept_0..K-1.
    Rows = statistical feature names.
    """
    if model.stat_encoder is None or header is None:
        return

    feat_names = [header[i] for i in feat_idx]
    D = len(feat_names)

    if concept_mode == "grouped" and concept_groups is not None:
        K = len(concept_groups)
        col_names = GROUP_NAMES[:K] if K <= len(GROUP_NAMES) else \
                    GROUP_NAMES + [f"concept_{i}" for i in range(len(GROUP_NAMES), K)]
        mask = [[0] * K for _ in range(D)]
        for k, grp in enumerate(concept_groups):
            for i in grp:
                if i < D:
                    mask[i][k] = 1
        fmt_cell = lambda v: " 1 " if v else " - "
    else:  # tabcbm
        import torch
        param = model.stat_encoder.concept_mask
        vals = torch.sigmoid(param).detach().cpu().tolist()  # [K, D]
        K = len(vals)
        col_names = [f"concept_{k}" for k in range(K)]
        mask = [[round(vals[k][i], 3) for k in range(K)] for i in range(D)]
        fmt_cell = lambda v: f"{v:.3f}"

    # compute column widths
    col_w = [max(len(c), 5) for c in col_names]
    feat_w = max(len(n) for n in feat_names)

    def md_row(first, cells, widths):
        parts = [first.ljust(feat_w)] + [str(c).center(w) for c, w in zip(cells, widths)]
        return "| " + " | ".join(parts) + " |"

    lines = []
    lines.append(f"# Concept Mask  (mode: {concept_mode})\n")
    lines.append(md_row("feature", col_names, col_w))
    lines.append("| " + "-" * feat_w + " | " +
                 " | ".join("-" * w for w in col_w) + " |")
    for name, row in zip(feat_names, mask):
        lines.append(md_row(name, [fmt_cell(v) for v in row], col_w))

    out_path = os.path.join(output_dir, "concept_mask.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save] concept_mask -> {out_path}", flush=True)

def _freeze_module(module: Optional[torch.nn.Module]) -> int:
    """Set requires_grad=False for all parameters in module.

    Returns number of frozen parameters.
    """
    if module is None:
        return 0
    n = 0
    for p in module.parameters():
        p.requires_grad = False
        n += p.numel()
    return n


def _load_pretrained_modalities(model, ckpt_path: str, device) -> None:
    """Load pretrained seq/bytes encoder weights if present."""
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid pretrain checkpoint: {ckpt_path}")

    if "seq_encoder" in state and model.seq_encoder is not None:
        missing, unexpected = model.seq_encoder.load_state_dict(state["seq_encoder"], strict=False)
        print(f"[pretrain] loaded seq_encoder from {ckpt_path}", flush=True)
        if missing:
            print(f"[pretrain] seq missing keys: {missing}", flush=True)
        if unexpected:
            print(f"[pretrain] seq unexpected keys: {unexpected}", flush=True)

    if "bytes_encoder" in state and model.bytes_encoder is not None:
        missing, unexpected = model.bytes_encoder.load_state_dict(state["bytes_encoder"], strict=False)
        print(f"[pretrain] loaded bytes_encoder from {ckpt_path}", flush=True)
        if missing:
            print(f"[pretrain] bytes missing keys: {missing}", flush=True)
        if unexpected:
            print(f"[pretrain] bytes unexpected keys: {unexpected}", flush=True)


def _stat_meta_cols():
    return {"src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp"}


def _get_stat_info(data_dir: str):
    """Return (stat_dim, header, feat_idx) from stats.parquet."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_dir, "stats", "stats.parquet"))
    header = list(df.columns)
    skip = _stat_meta_cols()
    feat_idx = [i for i, c in enumerate(header) if c != "flow_id" and c not in skip]
    return len(feat_idx), header, feat_idx


def _build_concept_groups(data_dir: str):
    stat_dim, header, feat_idx = _get_stat_info(data_dir)
    return build_cic_groups(header, feat_idx)


CONCEPT_GROUPS_FILENAME = "concept_groups.json"


def _save_concept_groups(output_dir: str, concept_groups) -> None:
    if concept_groups is None:
        return
    path = os.path.join(output_dir, CONCEPT_GROUPS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(concept_groups, f)
    print(f"[save] concept_groups -> {path}", flush=True)


def _load_concept_groups(ckpt_dir: Optional[str]):
    if not ckpt_dir:
        return None
    path = os.path.join(ckpt_dir, CONCEPT_GROUPS_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    return [[int(i) for i in grp] for grp in groups]


def _resolve_concept_groups(
    data_dir: str,
    ckpt_dir: Optional[str],
    use_stat: bool,
    concept_mode: str,
):
    """Return (concept_groups, source) for reproducible grouped-stat checkpoints."""
    if not use_stat:
        return None, "disabled"
    if concept_mode != "grouped":
        return _build_concept_groups(data_dir), "rebuilt"

    groups = _load_concept_groups(ckpt_dir)
    if groups is not None:
        return groups, "saved"
    return _build_concept_groups(data_dir), "rebuilt"


def _num_classes(data_dir: str) -> int:
    with open(os.path.join(data_dir, "label2id.json"), encoding="utf-8") as f:
        return len(json.load(f))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, num_classes, sparsity_weight=0.0):
    model.eval()
    total_loss, total = 0.0, 0
    cm = torch.zeros(num_classes, num_classes, device=device, dtype=torch.long)

    for batch in loader:
        y = batch["label"].to(device)
        kwargs = _batch_to_kwargs(batch, device)
        logits, _ = model(**kwargs)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

        pred = logits.argmax(dim=1)
        cm += torch.bincount(
            y * num_classes + pred, minlength=num_classes * num_classes
        ).view(num_classes, num_classes)

    avg_loss = total_loss / max(total, 1)
    acc = cm.diagonal().sum().item() / max(cm.sum().item(), 1)

    tp  = cm.diagonal().float()
    sup = cm.sum(dim=1).float()
    pp  = cm.sum(dim=0).float()
    prec = tp / pp.clamp(min=1.0)
    rec  = tp / sup.clamp(min=1.0)
    f1   = 2 * prec * rec / (prec + rec).clamp(min=1e-12)

    return avg_loss, acc, float(prec.mean()), float(rec.mean()), float(f1.mean())


# 6+4|b concept names for gate stats
_GATE_CONCEPT_NAMES = GROUP_NAMES + ["volume", "timing", "direction", "tcp"]


@torch.no_grad()
def _print_gate_stats(model, loader, device):
    """Print mean/std of sigmoid(g_bytes) per concept on test set (6+4|b only)."""
    model.eval()
    all_gates = []
    for batch in loader:
        kwargs = _batch_to_kwargs(batch, device)
        _, extras = model(**kwargs)
        if "g_bytes" not in extras:
            return
        all_gates.append(torch.sigmoid(extras["g_bytes"]))
    gates = torch.cat(all_gates, dim=0)
    mean = gates.mean(dim=0).cpu().tolist()
    std = gates.std(dim=0).cpu().tolist()
    print("\n[gate] sigmoid(g_bytes) on test set:", flush=True)
    for k, name in enumerate(_GATE_CONCEPT_NAMES):
        print(f"  {name:20s}  mean={mean[k]:.4f}  std={std[k]:.4f}", flush=True)
    print("", flush=True)


def _batch_to_kwargs(batch, device):
    """Convert collated batch dict to model forward kwargs."""
    kwargs = {}
    if "x_stat"     in batch: kwargs["x_stat"]     = batch["x_stat"].to(device).float()
    if "x_seq"      in batch: kwargs["x_seq"]       = batch["x_seq"].to(device).float()
    if "attn_mask"  in batch: kwargs["attn_mask"]   = batch["attn_mask"].to(device).float()
    if "x_bytes"    in batch: kwargs["x_bytes"]     = batch["x_bytes"].to(device).long()
    if "bytes_mask" in batch: kwargs["bytes_mask"]  = batch["bytes_mask"].to(device).long()
    return kwargs


def train_one_epoch(model, loader, device, optimizer, loss_fn, log_interval,
                    sparsity_weight=0.0, bytes_aux_weight=0.0):
    model.train()
    total_loss, total = 0.0, 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        y = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)

        kwargs = _batch_to_kwargs(batch, device)
        logits, extras = model(**kwargs)

        loss = loss_fn(logits, y)
        if sparsity_weight > 0:
            loss = loss + sparsity_weight * model.concept_sparsity_loss()

        # 6+4|b: auxiliary CE loss on gate values to prevent gate collapse
        if bytes_aux_weight > 0 and "g_bytes" in extras:
            aux_logits = model.bytes_aux_head(extras["g_bytes"])
            loss = loss + bytes_aux_weight * loss_fn(aux_logits, y)

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"  [train] step={step}/{len(loader)}"
                f"  loss={total_loss/total:.4f}"
                f"  speed={total/max(elapsed,1e-6):.0f} samples/s",
                flush=True,
            )

    return total_loss / max(total, 1)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _flatten_config(cfg: dict) -> dict:
    """Convert hierarchical config (model/train/runtime sections) to flat argparse dict.

    Supports both the new nested format and old flat format (pass-through).
    """
    if not any(k in cfg for k in ("data", "model", "train", "runtime")):
        return cfg   # old flat format — pass through unchanged

    flat: dict = {}
    d = cfg.get("data", {})
    m = cfg.get("model", {})

    for k in ("data_dir", "data_root", "dataset"):
        if k in d:
            flat[k] = d[k]

    # model-level scalars
    for k in ("fusion_mode", "num_concepts", "head", "dropout", "mlp_hidden"):
        if k in m:
            flat[k] = m[k]

    # stat branch
    for src, dst in [
        ("use",         "use_stat"),
        ("concept_mode","concept_mode"),
        ("hidden",      "stat_hidden"),
    ]:
        if src in m.get("stat", {}):
            flat[dst] = m["stat"][src]

    # seq branch
    for src, dst in [
        ("use",        "use_seq"),
        ("max_len",    "max_seq_len"),
        ("model_dim",  "seq_model_dim"),
        ("num_heads",  "seq_num_heads"),
        ("num_layers", "seq_layers"),
        ("pooling",    "seq_pooling"),
    ]:
        if src in m.get("seq", {}):
            flat[dst] = m["seq"][src]

    # bytes branch
    for src, dst in [
        ("use",        "use_bytes"),
        ("use_packet", "use_bytes_packet"),
        ("use_cross",  "use_bytes_cross"),
        ("len",        "bytes_len"),
        ("model_dim",  "bytes_model_dim"),
        ("num_heads",  "bytes_num_heads"),
        ("num_layers", "bytes_layers"),
        ("arch",       "bytes_arch"),
        ("pooling",    "bytes_pooling"),
        ("per_packet", "bytes_per_packet"),
        ("pkt_len",    "bytes_pkt_len"),
        ("num_packets", "num_packets"),
        ("num_groups",  "num_groups"),
    ]:
        if src in m.get("bytes", {}):
            flat[dst] = m["bytes"][src]

    # train & runtime sections map 1-to-1
    flat.update(cfg.get("train", {}))
    flat.update(cfg.get("runtime", {}))

    return flat


def _resolve_bytes_branch_flags(
    fusion_mode: str,
    use_bytes: bool,
    use_bytes_packet: bool = False,
    use_bytes_cross: bool = False,
):
    """Normalize bytes packet/cross switches with backward compatibility."""
    if fusion_mode == "6+4+5+3":
        if use_bytes_packet or use_bytes_cross:
            return True, bool(use_bytes_packet), bool(use_bytes_cross)
        if use_bytes:
            return True, True, True
        return False, False, False

    if fusion_mode == "6+4+5":
        packet_on = bool(use_bytes or use_bytes_packet)
        return packet_on, packet_on, False

    if fusion_mode == "6+4+3":
        cross_on = bool(use_bytes or use_bytes_cross)
        return cross_on, False, cross_on

    return bool(use_bytes), bool(use_bytes_packet), bool(use_bytes_cross)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("TrafficCBM training")

    # Config file (loaded first; CLI flags override)
    ap.add_argument("--config", default=None, help="path to config.json")

    # Data
    ap.add_argument("--data_dir",  default=None,  help="data/{name}/ self-contained directory")
    ap.add_argument("--data_root", default=None,  help="root data directory containing per-dataset subdirs")
    ap.add_argument("--dataset",   default=None,  help="dataset name under --data_root")
    ap.add_argument("--output",    default="outputs/run", help="output directory for checkpoints")

    # Modality toggles
    ap.add_argument("--use_stat",  action="store_true")
    ap.add_argument("--use_seq",   action="store_true")
    ap.add_argument("--use_bytes", action="store_true")
    ap.add_argument("--use_bytes_packet", action="store_true")
    ap.add_argument("--use_bytes_cross",  action="store_true")

    # Model
    ap.add_argument("--concept_mode",   default="grouped", choices=["grouped", "tabcbm"])
    ap.add_argument("--num_concepts",   type=int, default=6)
    ap.add_argument("--stat_hidden",    type=int, nargs="+", default=[64, 64, 32],
                    metavar="DIM", help="stat MLP hidden layer sizes, e.g. --stat_hidden 64 64 32")
    ap.add_argument("--max_seq_len",    type=int, default=100)
    ap.add_argument("--seq_model_dim",  type=int, default=64)
    ap.add_argument("--seq_num_heads",  type=int, default=4)
    ap.add_argument("--seq_layers",     type=int, default=2)
    ap.add_argument("--bytes_len",      type=int, default=320)
    ap.add_argument("--bytes_model_dim", type=int, default=64)
    ap.add_argument("--bytes_num_heads", type=int, default=4)
    ap.add_argument("--bytes_layers",   type=int, default=2)
    ap.add_argument("--bytes_arch",    default="transformer", choices=["transformer", "cnn"])
    ap.add_argument("--seq_pooling",   default="mean", choices=["mean", "max", "cls"])
    ap.add_argument("--bytes_pooling", default="mean", choices=["mean", "max", "cls"])
    ap.add_argument("--fusion_mode",   default="6+1+1", choices=["6+1+1", "6/6/6", "6+4", "6+4|b", "6+4+5", "6+4+5+3", "6+4+3"])
    ap.add_argument("--bytes_per_packet", action="store_true",
                    help="split bytes by SEP into per-packet segments (for 6+4+5)")
    ap.add_argument("--bytes_pkt_len", type=int, default=63,
                    help="tokens per packet when bytes_per_packet=True (63 for 64-byte payload)")
    ap.add_argument("--num_packets",   type=int, default=5,
                    help="number of packet segments for bytes (6+4+5, 6+4+5+3, 6+4+3)")
    ap.add_argument("--num_groups",    type=int, default=3,
                    help="cross-packet groups for bytes (6+4+5+3, 6+4+3: 3 cross-packet concepts)")
    ap.add_argument("--bytes_aux_weight", type=float, default=0.0,
                    help="auxiliary classification loss weight for BytesGateEncoder (6+4|b mode)")
    ap.add_argument("--head",          default="mlp", choices=["mlp", "taylor", "ga2m"])
    ap.add_argument("--mlp_hidden",    type=int, default=None,
                    help="MLP head hidden width (head=mlp only); omit for max(64, 2*fused_dim)")
    ap.add_argument("--dropout",       type=float, default=0.1)

    # Two-stage training
    ap.add_argument("--init_ckpt",    default=None,
                    help="path to Stage-1 checkpoint to load before Stage-2 training "
                         "(uses strict=False so new bytes params are randomly initialised)")
    ap.add_argument("--pretrain_ckpt", default=None,
                    help="path to seq+bytes self-supervised pretrain checkpoint")
    ap.add_argument("--freeze_stat",  action="store_true",
                    help="freeze stat_encoder weights (Stage-2: keep concept backbone fixed)")
    ap.add_argument("--freeze_seq",   action="store_true",
                    help="freeze seq_encoder weights (Stage-2: keep concept backbone fixed)")
    ap.add_argument("--freeze_head",  action="store_true",
                    help="freeze head weights (Stage-2: keep classifier fixed)")

    # Training
    ap.add_argument("--epochs",       type=int,   default=40)
    ap.add_argument("--batch_size",   type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--lr_stat",      type=float, default=None)
    ap.add_argument("--lr_seq",       type=float, default=None)
    ap.add_argument("--lr_bytes",     type=float, default=None)
    ap.add_argument("--weight_decay",     type=float, default=1e-2)
    ap.add_argument("--label_smoothing",  type=float, default=0.0,
                    help="CrossEntropyLoss label smoothing (0=off, 0.1 recommended)")
    ap.add_argument("--sparsity_weight", type=float, default=0.0,
                    help="L1 weight for tabcbm concept mask sparsity")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="stop early if val_loss does not improve for N epochs (0=disabled)")
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--log_interval", type=int,   default=100)

    # --- load config.json defaults before final parse ---
    pre, _ = ap.parse_known_args()
    if pre.config is not None:
        with open(pre.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg = _flatten_config(cfg)   # handle both flat and hierarchical formats
        # boolean flags need special handling for action="store_true"
        bool_flags = {"use_stat", "use_seq", "use_bytes", "use_bytes_packet", "use_bytes_cross",
                      "freeze_stat", "freeze_seq", "freeze_head"}
        defaults = {k: v for k, v in cfg.items() if v is not None and k not in bool_flags}
        ap.set_defaults(**defaults)
        # inject boolean flags as if passed on CLI when true in config
        for flag in bool_flags:
            if cfg.get(flag):
                import sys as _sys
                flag_str = f"--{flag}"
                if flag_str not in _sys.argv:
                    _sys.argv.append(flag_str)

    args = ap.parse_args()
    args.use_bytes, args.use_bytes_packet, args.use_bytes_cross = _resolve_bytes_branch_flags(
        args.fusion_mode,
        args.use_bytes,
        getattr(args, "use_bytes_packet", False),
        getattr(args, "use_bytes_cross", False),
    )

    if not args.data_dir:
        if args.data_root and args.dataset:
            args.data_dir = os.path.join(args.data_root, args.dataset)
        else:
            ap.error("--data_dir is required, or set both --data_root and --dataset (config supported).")
    if not args.use_stat and not args.use_seq and not args.use_bytes:
        ap.error("At least one of --use_stat / --use_seq / --use_bytes must be set.")

    os.makedirs(args.output, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    print(f"[dataset] {dataset_name}", flush=True)
    print(f"[data_dir] {args.data_dir}", flush=True)

    # copy config to output dir for reproducibility
    import shutil as _shutil
    if args.config is not None:
        _shutil.copy2(args.config, os.path.join(args.output, "config.json"))
    else:
        # save current args as config.json so output is always self-contained
        cfg_snapshot = {k: v for k, v in vars(args).items()
                        if k not in ("output", "config")}
        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as _f:
            json.dump(cfg_snapshot, _f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # ---- scalers ----
    print("[data] fitting scalers ...", flush=True)
    stat_s, seq_s = fit_scalers(
        args.data_dir,
        os.path.join(args.data_dir, "train_ids.txt"),
        fit_stat=args.use_stat,
        fit_seq=args.use_seq,
    )
    scaler_path = os.path.join(args.output, FEATURE_SCALERS_FILENAME)
    save_feature_scalers(scaler_path, stat_s, seq_s)
    print(f"[save] feature scalers -> {scaler_path}", flush=True)

    # ---- datasets ----
    bytes_per_packet = getattr(args, "bytes_per_packet", False) or args.fusion_mode in ("6+4+5", "6+4+5+3", "6+4+3")
    def make_ds(split):
        return TrafficDataset(
            ids_txt  = os.path.join(args.data_dir, f"{split}_ids.txt"),
            data_dir = args.data_dir,
            use_stat  = args.use_stat,
            use_seq   = args.use_seq,
            use_bytes = args.use_bytes,
            max_seq_len = args.max_seq_len,
            bytes_len   = args.bytes_len,
            bytes_per_packet = bytes_per_packet,
            bytes_pkt_len    = getattr(args, "bytes_pkt_len", 63),
            num_packets      = getattr(args, "num_packets", 5),
            stat_scaler = stat_s,
            seq_scaler  = seq_s,
        )

    print("[data] loading datasets ...", flush=True)
    ds_train = make_ds("train")
    ds_val   = make_ds("val")
    ds_test  = make_ds("test")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate_fn)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate_fn)

    # ---- model ----
    num_classes = _num_classes(args.data_dir)
    stat_dim, header, feat_idx = _get_stat_info(args.data_dir) if args.use_stat else (None, None, None)
    concept_groups = _build_concept_groups(args.data_dir) if args.use_stat else None
    if args.use_stat and args.concept_mode == "grouped":
        _save_concept_groups(args.output, concept_groups)

    model = build_model(
        num_classes    = num_classes,
        stat_dim       = stat_dim,
        num_concepts   = args.num_concepts,
        concept_mode   = args.concept_mode,
        concept_groups = concept_groups,
        max_seq_len    = args.max_seq_len,
        bytes_len      = args.bytes_len,
        use_stat       = args.use_stat,
        use_seq        = args.use_seq,
        use_bytes      = args.use_bytes,
        use_bytes_packet = args.use_bytes_packet,
        use_bytes_cross  = args.use_bytes_cross,
        head           = args.head,
        stat_hidden    = tuple(args.stat_hidden),
        seq_model_dim  = args.seq_model_dim,
        seq_num_heads  = args.seq_num_heads,
        seq_layers     = args.seq_layers,
        bytes_model_dim = args.bytes_model_dim,
        bytes_num_heads = args.bytes_num_heads,
        bytes_layers   = args.bytes_layers,
        bytes_arch     = args.bytes_arch,
        seq_pooling    = args.seq_pooling,
        bytes_pooling  = args.bytes_pooling,
        fusion_mode    = args.fusion_mode,
        dropout        = args.dropout,
        bytes_per_packet = getattr(args, "bytes_per_packet", False) or args.fusion_mode in ("6+4+5", "6+4+5+3", "6+4+3"),
        bytes_pkt_len    = getattr(args, "bytes_pkt_len", 63),
        num_packets      = getattr(args, "num_packets", 5),
        num_groups       = getattr(args, "num_groups", 3),
        mlp_hidden       = getattr(args, "mlp_hidden", None),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] fused_dim={model.fused_dim}  params={total_params:,}  classes={num_classes}", flush=True)
    print(f"[model] concept_mode={args.concept_mode}  fusion_mode={args.fusion_mode}  head={args.head}", flush=True)
    if args.head == "mlp" and getattr(args, "mlp_hidden", None) is not None:
        print(f"[model] mlp_hidden={args.mlp_hidden}", flush=True)
    print(f"[model] branches: stat={args.use_stat} seq={args.use_seq} bytes={args.use_bytes} pkt={args.use_bytes_packet} cross={args.use_bytes_cross}", flush=True)
    print(f"[train] label_smoothing={args.label_smoothing}  dropout={args.dropout}  bytes_aux_weight={args.bytes_aux_weight}", flush=True)

    # ---- optional seq+bytes self-supervised pretraining ----
    if args.pretrain_ckpt is not None:
        _load_pretrained_modalities(model, args.pretrain_ckpt, device)

    # ---- Stage-2: load Stage-1 checkpoint ----
    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init]  loaded Stage-1 ckpt: {args.init_ckpt}", flush=True)
        if missing:
            print(f"[init]  missing keys (new params, will be randomly init): {missing}", flush=True)
        if unexpected:
            print(f"[init]  unexpected keys (ignored): {unexpected}", flush=True)

    # ---- Stage-2: freeze modules ----
    # Freeze BEFORE building optimizer so frozen params are excluded automatically.
    if args.freeze_stat:
        n = _freeze_module(model.stat_encoder)
        print(f"[freeze] stat_encoder  frozen  ({n:,} params)", flush=True)
    if args.freeze_seq:
        n = _freeze_module(model.seq_encoder)
        print(f"[freeze] seq_encoder   frozen  ({n:,} params)", flush=True)
    if args.freeze_head:
        n = _freeze_module(model.head)
        print(f"[freeze] head          frozen  ({n:,} params)", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {trainable:,} / {total_params:,}", flush=True)

    # ---- optimizer: only collect trainable parameters (respects freeze flags) ----
    def _trainable(module):
        if module is None:
            return []
        return [p for p in module.parameters() if p.requires_grad]

    param_groups = []
    if _trainable(model.head):
        param_groups.append({"params": _trainable(model.head), "lr": args.lr})
    if _trainable(model.stat_encoder):
        param_groups.append({"params": _trainable(model.stat_encoder),
                              "lr": args.lr_stat or args.lr})
    if _trainable(model.seq_encoder):
        param_groups.append({"params": _trainable(model.seq_encoder),
                              "lr": args.lr_seq or args.lr})
    if _trainable(model.bytes_encoder):
        param_groups.append({"params": _trainable(model.bytes_encoder),
                              "lr": args.lr_bytes or args.lr})
    # bytes_aux_head (6+4|b only) — always trainable if it exists
    if hasattr(model, "bytes_aux_head") and _trainable(model.bytes_aux_head):
        param_groups.append({"params": _trainable(model.bytes_aux_head), "lr": args.lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found. Check --freeze_* flags.")

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---- training loop ----
    best_val_loss = float("inf")
    best_state    = None
    epochs_no_improve = 0

    print(f"[train] starting: {args.epochs} epochs, {len(ds_train)} train / {len(ds_val)} val", flush=True)
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, dl_train, device, optimizer, loss_fn,
            args.log_interval, sparsity_weight=args.sparsity_weight,
            bytes_aux_weight=args.bytes_aux_weight,
        )
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(
            model, dl_val, device, loss_fn, num_classes,
        )
        scheduler.step()
        dt  = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d} | "
            f"lr={cur_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} | {dt:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0:
                print(
                    f"[early_stop] no improvement for {epochs_no_improve}/{args.early_stop_patience} epochs",
                    flush=True,
                )
                if epochs_no_improve >= args.early_stop_patience:
                    print(
                        f"[early_stop] stop at epoch {epoch:03d} (best_val_loss={best_val_loss:.4f})",
                        flush=True,
                    )
                    break

    # ---- save & test ----
    if best_state is not None:
        ckpt = os.path.join(args.output, "best.pt")
        torch.save(best_state, ckpt)
        print(f"[save] best model -> {ckpt}", flush=True)
        model.load_state_dict(best_state)

    test_loss, test_acc, test_p, test_r, test_f1 = evaluate(
        model, dl_test, device, loss_fn, num_classes,
    )
    total_sec = time.time() - t_train_start
    h, rem = divmod(int(total_sec), 3600)
    m, s   = divmod(rem, 60)
    print(
        f"[test] loss={test_loss:.4f} | "
        f"acc={test_acc:.4f} | p={test_p:.4f} | r={test_r:.4f} | f1={test_f1:.4f}",
        flush=True,
    )
    print(f"[done] total training time: {h:02d}h {m:02d}m {s:02d}s", flush=True)

    if args.fusion_mode == "6+4|b":
        _print_gate_stats(model, dl_test, device)

    if args.use_stat:
        _save_concept_mask(model, args.concept_mode, concept_groups,
                           header, feat_idx, args.output)


if __name__ == "__main__":
    main()
