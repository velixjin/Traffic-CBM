#!/usr/bin/env python3
"""
Sufficiency evaluation: compare classification performance when keeping only
top-k ranked concepts and replacing all remaining concepts with a baseline.

- Ranking: mean |Δ logit_target| per concept (same as analyze_contrib / faithfulness_eval).
- Metrics: accuracy and macro-F1.
- Keep-only: preserve selected top-k concept dimensions, set all others to baseline.

Usage:
  python eval/sufficiency_eval.py \
    --ckpt_dir outputs/CipherSpectrum-AES128/20260320_171456 \
    --data_dir data/CipherSpectrum-AES128 \
    --split test \
    --output_dir outputs/contrib/CipherSpectrum-AES128 \
    --ks 3,5,10 \
    --seed 42
"""
from __future__ import annotations

import argparse
import csv
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

import analyze_contrib as ac
import faithfulness_eval as fe
from train.train import evaluate as train_evaluate


@torch.no_grad()
def _accuracy_keep_only(
    model: torch.nn.Module,
    concepts: torch.Tensor,
    labels: torch.Tensor,
    keep_idx: Optional[List[int]],
    baseline: torch.Tensor,
) -> Tuple[float, float, torch.Tensor]:
    """Return (acc, macro_f1, logits) after keeping only selected concept dimensions."""
    c = baseline.unsqueeze(0).expand_as(concepts).clone()
    if keep_idx:
        c[:, keep_idx] = concepts[:, keep_idx]
    logits = ac._head_forward(model, c)
    acc, f1 = fe._metrics_from_logits(logits, labels)
    return acc, f1, logits


def main():
    ap = argparse.ArgumentParser("Sufficiency: keep only top-k concepts vs accuracy")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--target",
        default="true",
        choices=["true", "pred"],
        help="ranking target logit (same as analyze_contrib)",
    )
    ap.add_argument("--baseline", default="mean", choices=["mean", "zero"])
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ks",
        type=str,
        default="3,5,10",
        help="comma-separated k for top-k only evaluation",
    )
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
    model, loader, cfg, concept_names = fe._load_model_and_data(args, device)

    logits, labels, concepts = ac.collect_outputs(model, loader, device)
    pred = logits.argmax(dim=1)
    target_idx = labels if args.target == "true" else pred

    num_classes = int(logits.size(1))
    ls = float(cfg.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)
    _, full_acc, _, _, full_f1 = train_evaluate(
        model, loader, device, loss_fn, num_classes, sparsity_weight=0.0
    )
    acc_from_logits, f1_from_logits = fe._metrics_from_logits(logits, labels)
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
    baseline = (
        concepts.mean(dim=0)
        if args.baseline == "mean"
        else torch.zeros(c_dim, device=device, dtype=concepts.dtype)
    )
    concepts = concepts.to(device)
    labels = labels.to(device)
    target_idx = target_idx.to(device)
    baseline = baseline.to(device)

    logits_re = ac._head_forward(model, concepts)
    max_diff = (logits.to(device) - logits_re).abs().max().item()
    print(f"[check] max|logits_forward - head(c)| = {max_diff:.2e}  (expect ~0)", flush=True)
    if max_diff > 1e-3:
        print(
            "[warn] forward logits vs head(concepts) mismatch; sufficiency metrics may be inconsistent",
            flush=True,
        )

    print(f"[sufficiency] samples={concepts.size(0)}  concepts={c_dim}  baseline={args.baseline}", flush=True)
    print(f"[sufficiency] ranking = mean |Δ logit_target| (ablation), same as ablation bar", flush=True)

    mean_abs = fe._mean_abs_delta_batches(
        model, concepts, labels, target_idx, baseline, args.batch_size
    )
    order = np.argsort(-mean_abs).tolist()
    rows: List[Dict] = []

    rows.append(
        {
            "setting": "Full Model",
            "k": "",
            "accuracy": full_acc,
            "macro_f1": full_f1,
            "kept": "",
        }
    )
    print(f"  full           acc={full_acc:.4f}  macro_f1={full_f1:.4f}", flush=True)

    for k in ks:
        if k > c_dim:
            print(f"  [skip] k={k} > num_concepts={c_dim}", flush=True)
            continue

        keep_idx = order[:k]
        acc_k, f1_k, _ = _accuracy_keep_only(model, concepts, labels, keep_idx, baseline)
        rows.append(
            {
                "setting": f"Top-{k} Only",
                "k": k,
                "accuracy": acc_k,
                "macro_f1": f1_k,
                "kept": ",".join(concept_names[i] for i in keep_idx),
            }
        )
        print(
            f"  top-{k:2d} only    acc={acc_k:.4f}  macro_f1={f1_k:.4f}  (drop acc {full_acc - acc_k:+.4f})",
            flush=True,
        )

    prefix = os.path.join(output_dir, f"sufficiency_{args.split}")
    txt_path = prefix + ".txt"
    csv_path = prefix + ".csv"
    tex_path = prefix + "_table_rows.tex"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("concept_rank (by mean |ablation delta| on target logit)\n")
        for r, idx in enumerate(order, 1):
            f.write(f"  {r:3d}  {concept_names[idx]:<40}  {mean_abs[idx]:.6f}\n")
        f.write("\n")
        for row in rows:
            f.write(str(row) + "\n")
    print(f"[save] {txt_path}", flush=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "k", "accuracy", "macro_f1", "kept"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[save] {csv_path}", flush=True)

    with open(tex_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                f"{row['setting']} & {row['accuracy']:.4f} & {row['macro_f1']:.4f} \\\\\n"
            )
    print(f"[save] {tex_path}", flush=True)


if __name__ == "__main__":
    main()
