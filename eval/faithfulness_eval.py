#!/usr/bin/env python3
"""
Faithfulness evaluation: compare classification performance when ablating
top-ranked vs bottom-ranked concepts (same ablation definition as analyze_contrib).

- Ranking: mean |Δ logit_true| per concept (ablation_contrib), same as ablation bar.
- Metrics: accuracy (and macro-F1 if sklearn available).
- Ablate: set selected concept dimensions to baseline (mean or zero over split).

Usage:
  python eval/faithfulness_eval.py \\
    --ckpt_dir outputs/CipherSpectrum-AES128/20260314_153627 \\
    --data_dir data/CipherSpectrum-AES128 \\
    --split test \\
    --output_dir outputs/contrib/CipherSpectrum-AES128 \\
    --ks 3,5,10 \\
    --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import analyze_contrib as ac
from datasets.traffic_dataset import TrafficDataset, collate_fn, fit_or_load_scalers
from model.traffic_cbm import build_model
from train.train import (
    _load_concept_groups,
    _resolve_concept_groups,
    _build_concept_groups,
    _flatten_config,
    _get_stat_info,
    _num_classes,
    _resolve_bytes_branch_flags,
    evaluate as train_evaluate,
)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        return float("nan")


def _metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """Same as train.evaluate: acc = diag(cm)/sum(cm), macro_f1 = mean per-class f1."""
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean().item()
    y_t = labels.detach().cpu().numpy()
    y_p = pred.detach().cpu().numpy()
    nc = int(logits.size(1))
    f1 = _macro_f1(y_t, y_p, nc)
    return acc, f1


@torch.no_grad()
def _accuracy_and_logits(
    model: torch.nn.Module,
    concepts: torch.Tensor,
    labels: torch.Tensor,
    ablate_idx: Optional[List[int]] = None,
    baseline: Optional[torch.Tensor] = None,
) -> Tuple[float, float, torch.Tensor]:
    """Return (acc, macro_f1, logits) after optional ablation on concept dimensions."""
    device = concepts.device
    c = concepts.clone()
    if ablate_idx and baseline is not None:
        for k in ablate_idx:
            c[:, k] = baseline[k]
    logits = ac._head_forward(model, c)
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean().item()
    y_t = labels.detach().cpu().numpy()
    y_p = pred.detach().cpu().numpy()
    nc = int(logits.size(1))
    f1 = _macro_f1(y_t, y_p, nc)
    return acc, f1, logits


@torch.no_grad()
def _mean_abs_delta_batches(
    model: torch.nn.Module,
    concepts: torch.Tensor,
    labels: torch.Tensor,
    target_idx: torch.Tensor,
    baseline: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Accumulate mean |Δ| per concept (same as ablation bar ordering)."""
    device = concepts.device
    n, c_dim = concepts.shape
    total_abs = torch.zeros(c_dim, device=device, dtype=torch.float64)
    total_n = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_c = concepts[start:end]
        batch_t = target_idx[start:end]
        deltas = ac.ablation_contrib(model, batch_c, batch_t, baseline)
        total_abs += deltas.abs().sum(dim=0).double().to(device)
        total_n += end - start
    mean_abs = (total_abs / max(total_n, 1)).cpu().numpy()
    return mean_abs


def _load_model_and_data(args, device):
    if args.ckpt_dir:
        cfg_path = args.config or os.path.join(args.ckpt_dir, "config.json")
        ckpt_path = args.ckpt or os.path.join(args.ckpt_dir, "best.pt")
    else:
        cfg_path = args.config
        ckpt_path = args.ckpt
    if not cfg_path or not ckpt_path:
        raise ValueError("Need --ckpt_dir or both --ckpt and --config")
    cfg = _flatten_config(json.load(open(cfg_path, "r", encoding="utf-8")))

    use_stat = cfg.get("use_stat", True)
    use_seq = cfg.get("use_seq", True)
    use_bytes = cfg.get("use_bytes", True)
    fusion_mode = cfg.get("fusion_mode", "6+1+1")

    ckpt_dir = args.ckpt_dir or (
        os.path.dirname(os.path.abspath(args.ckpt)) if getattr(args, "ckpt", None) else None
    )
    stat_s, seq_s = fit_or_load_scalers(
        args.data_dir,
        os.path.join(args.data_dir, "train_ids.txt"),
        fit_stat=use_stat,
        fit_seq=use_seq,
        ckpt_dir=ckpt_dir,
    )
    use_bytes, use_bytes_packet, use_bytes_cross = _resolve_bytes_branch_flags(
        fusion_mode,
        bool(cfg.get("use_bytes", False)),
        bool(cfg.get("use_bytes_packet", False)),
        bool(cfg.get("use_bytes_cross", False)),
    )
    bytes_per_packet = bool(cfg.get("bytes_per_packet", False) or fusion_mode in ("6+4+5", "6+4+5+3", "6+4+3"))
    num_packets = int(cfg.get("num_packets", 5))
    num_groups = int(cfg.get("num_groups", 3))

    ds = TrafficDataset(
        ids_txt=os.path.join(args.data_dir, f"{args.split}_ids.txt"),
        data_dir=args.data_dir,
        use_stat=use_stat,
        use_seq=use_seq,
        use_bytes=use_bytes,
        max_seq_len=int(cfg.get("max_seq_len", 100)),
        bytes_len=int(cfg.get("bytes_len", 320)),
        bytes_per_packet=bytes_per_packet,
        bytes_pkt_len=int(cfg.get("bytes_pkt_len", 63)),
        num_packets=num_packets,
        stat_scaler=stat_s,
        seq_scaler=seq_s,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
    )

    num_classes = _num_classes(args.data_dir)
    stat_dim, _, _ = _get_stat_info(args.data_dir) if use_stat else (None, None, None)
    concept_mode = cfg.get("concept_mode", "grouped")
    concept_groups, concept_groups_src = _resolve_concept_groups(
        args.data_dir, ckpt_dir, use_stat, concept_mode
    )
    if use_stat and concept_mode == "grouped":
        if concept_groups_src == "saved":
            print("  [concept_groups] loaded from checkpoint", flush=True)
        else:
            print(
                "  [warn] concept_groups.json not found; rebuilding grouped stat feature order from current code/data. "
                "Old checkpoints may not be exactly reproducible.",
                flush=True,
            )
    model = build_model(
        num_classes=num_classes,
        stat_dim=stat_dim,
        num_concepts=cfg.get("num_concepts", 6),
        concept_mode=concept_mode,
        concept_groups=concept_groups,
        max_seq_len=cfg.get("max_seq_len", 100),
        bytes_len=cfg.get("bytes_len", 320),
        use_stat=use_stat,
        use_seq=use_seq,
        use_bytes=use_bytes,
        use_bytes_packet=use_bytes_packet,
        use_bytes_cross=use_bytes_cross,
        head=cfg.get("head", "mlp"),
        stat_hidden=tuple(cfg.get("stat_hidden", [64, 64, 32])),
        seq_model_dim=cfg.get("seq_model_dim", 64),
        seq_num_heads=cfg.get("seq_num_heads", 4),
        seq_layers=cfg.get("seq_layers", 2),
        bytes_model_dim=cfg.get("bytes_model_dim", 64),
        bytes_num_heads=cfg.get("bytes_num_heads", 4),
        bytes_layers=cfg.get("bytes_layers", 2),
        bytes_arch=cfg.get("bytes_arch", "transformer"),
        seq_pooling=cfg.get("seq_pooling", "mean"),
        bytes_pooling=cfg.get("bytes_pooling", "mean"),
        fusion_mode=fusion_mode,
        dropout=cfg.get("dropout", 0.1),
        bytes_per_packet=bytes_per_packet,
        bytes_pkt_len=cfg.get("bytes_pkt_len", 63),
        num_packets=num_packets,
        num_groups=num_groups,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    concept_names = ac._concept_names(
        fusion_mode,
        num_packets=num_packets,
        num_groups=num_groups,
        use_stat=use_stat,
        use_seq=use_seq,
        use_bytes=use_bytes,
        use_bytes_packet=use_bytes_packet,
        use_bytes_cross=use_bytes_cross,
    )

    return model, loader, cfg, concept_names


def main():
    ap = argparse.ArgumentParser("Faithfulness: top-k vs bottom-k concept ablation vs accuracy")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target", default="true", choices=["true", "pred"], help="ranking target logit (same as analyze_contrib)")
    ap.add_argument("--baseline", default="mean", choices=["mean", "zero"])
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ks",
        type=str,
        default="3,5,10",
        help="comma-separated k for top-k / bottom-k / random-k",
    )
    ap.add_argument("--num_random", type=int, default=5, help="random k trials (different seeds) to average")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir or args.ckpt_dir
    if not args.ckpt_dir and not args.ckpt:
        ap.error("--ckpt_dir or --ckpt required")
    os.makedirs(output_dir, exist_ok=True)

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if not ks:
        ks = [3, 5, 10]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loader, cfg, concept_names = _load_model_and_data(args, device)

    logits, labels, concepts = ac.collect_outputs(model, loader, device)
    pred = logits.argmax(dim=1)
    target_idx = labels if args.target == "true" else pred

    # Use the same accuracy and macro-F1 computation as the training evaluation.
    num_classes = int(logits.size(1))
    ls = float(cfg.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)
    _, full_acc, _, _, full_f1 = train_evaluate(
        model, loader, device, loss_fn, num_classes, sparsity_weight=0.0
    )
    acc_from_logits, f1_from_logits = _metrics_from_logits(logits, labels)
    if abs(full_acc - acc_from_logits) > 1e-4:
        print(
            f"[warn] collect_outputs logits acc ({acc_from_logits:.6f}) vs train.evaluate ({full_acc:.6f})",
            flush=True,
        )
    print(
        f"[verify] collect_outputs vs train.evaluate: acc_logits={acc_from_logits:.4f} acc_eval={full_acc:.4f} (should match)",
        flush=True,
    )

    c_dim = concepts.size(1)
    baseline = concepts.mean(dim=0) if args.baseline == "mean" else torch.zeros(c_dim, device=device, dtype=concepts.dtype)
    concepts = concepts.to(device)
    labels = labels.to(device)
    target_idx = target_idx.to(device)
    baseline = baseline.to(device)

    logits_re = ac._head_forward(model, concepts)
    max_diff = (logits.to(device) - logits_re).abs().max().item()
    print(f"[check] max|logits_forward - head(c)| = {max_diff:.2e}  (expect ~0)", flush=True)
    if max_diff > 1e-3:
        print(
            "[warn] forward logits vs head(concepts) mismatch; ablation metrics may be inconsistent",
            flush=True,
        )

    print(f"[faithfulness] samples={concepts.size(0)}  concepts={c_dim}  baseline={args.baseline}", flush=True)
    print(f"[faithfulness] ranking = mean |Δ logit_target| (ablation), same as ablation bar", flush=True)

    mean_abs = _mean_abs_delta_batches(
        model, concepts, labels, target_idx, baseline, args.batch_size
    )
    order = np.argsort(-mean_abs).tolist()
    rows: List[Dict] = []

    row0 = {
        "setting": "full",
        "k": "",
        "ablated": "",
        "accuracy": full_acc,
        "macro_f1": full_f1,
    }
    rows.append(row0)
    print(f"  full          acc={full_acc:.4f}  macro_f1={full_f1:.4f}", flush=True)

    for k in ks:
        if k > c_dim:
            print(f"  [skip] k={k} > num_concepts={c_dim}", flush=True)
            continue
        if 2 * k > c_dim:
            print(
                f"  [warn] k={k}: top-{k} and bottom-{k} overlap (C={c_dim}); interpret with care",
                flush=True,
            )

        top_idx = order[:k]
        bot_idx = order[-k:]

        acc_t, f1_t, _ = _accuracy_and_logits(model, concepts, labels, top_idx, baseline)
        acc_b, f1_b, _ = _accuracy_and_logits(model, concepts, labels, bot_idx, baseline)

        rows.append(
            {
                "setting": "ablate_top",
                "k": k,
                "ablated": ",".join(concept_names[i] for i in top_idx),
                "accuracy": acc_t,
                "macro_f1": f1_t,
            }
        )
        rows.append(
            {
                "setting": "ablate_bottom",
                "k": k,
                "ablated": ",".join(concept_names[i] for i in bot_idx),
                "accuracy": acc_b,
                "macro_f1": f1_b,
            }
        )
        print(
            f"  top-{k:2d}        acc={acc_t:.4f}  macro_f1={f1_t:.4f}  (drop acc {full_acc - acc_t:+.4f})",
            flush=True,
        )
        print(
            f"  bottom-{k:2d}     acc={acc_b:.4f}  macro_f1={f1_b:.4f}  (drop acc {full_acc - acc_b:+.4f})",
            flush=True,
        )

        rng = random.Random(args.seed)
        acc_r_list = []
        f1_r_list = []
        for t in range(args.num_random):
            # sample k distinct indices without replacement
            pool = list(range(c_dim))
            rng.shuffle(pool)
            ridx = pool[:k]
            acc_r, f1_r, _ = _accuracy_and_logits(model, concepts, labels, ridx, baseline)
            acc_r_list.append(acc_r)
            f1_r_list.append(f1_r)
        acc_rm = float(np.mean(acc_r_list))
        f1_rm = float(np.mean(f1_r_list))
        rows.append(
            {
                "setting": "ablate_random_mean",
                "k": k,
                "ablated": f"{args.num_random}_trials",
                "accuracy": acc_rm,
                "macro_f1": f1_rm,
            }
        )
        print(
            f"  random-{k:2d} (mean {args.num_random} trials) acc={acc_rm:.4f}  macro_f1={f1_rm:.4f}",
            flush=True,
        )

    prefix = os.path.join(output_dir, f"faithfulness_{args.split}")
    txt_path = prefix + ".txt"
    csv_path = prefix + ".csv"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("concept_rank (by mean |ablation delta| on target logit)\n")
        for r, idx in enumerate(order, 1):
            f.write(f"  {r:3d}  {concept_names[idx]:<40}  {mean_abs[idx]:.6f}\n")
        f.write("\n")
        for row in rows:
            f.write(str(row) + "\n")
    print(f"[save] {txt_path}", flush=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "k", "accuracy", "macro_f1", "ablated"])
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "setting": row["setting"],
                    "k": row["k"],
                    "accuracy": row["accuracy"],
                    "macro_f1": row["macro_f1"],
                    "ablated": row["ablated"],
                }
            )
    print(f"[save] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
