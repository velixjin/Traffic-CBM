#!/usr/bin/env python3
"""
Evaluate a checkpoint using the same metrics as the training runner.

Usage with the same checkpoint and feature directory as the faithfulness evaluation:
  python eval/eval_ckpt.py \\
    --ckpt_dir outputs/CipherSpectrum-AES128/20260316_171441 \\
    --data_dir ./data/CipherSpectrum-AES128 \\
    --split test

Output uses the same format as the final evaluation in train.log:
  [test] loss=... | acc=... | p=... | r=... | f1=...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import torch
import torch.nn as nn

import faithfulness_eval as fe
from train.train import _flatten_config, _num_classes, evaluate, _load_concept_groups


def main():
    ap = argparse.ArgumentParser("Evaluate checkpoint (same metrics as train.py [test]/[val])")
    ap.add_argument("--ckpt_dir", default=None, help="directory with config.json + best.pt")
    ap.add_argument("--ckpt", default=None, help="override checkpoint path")
    ap.add_argument("--config", default=None, help="override config path")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="default: train.batch_size from config, else 256 (same as train default)",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.ckpt_dir and not (args.ckpt and args.config):
        ap.error("Need --ckpt_dir or both --ckpt and --config")

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg_path = args.config or os.path.join(args.ckpt_dir, "config.json")
    flat = _flatten_config(json.load(open(cfg_path, "r", encoding="utf-8")))
    if args.batch_size is None:
        args.batch_size = int(flat.get("batch_size", 256))

    ckpt_dir = args.ckpt_dir or (
        os.path.dirname(os.path.abspath(args.ckpt)) if getattr(args, "ckpt", None) else None
    )
    if flat.get("use_stat", True) and flat.get("concept_mode", "grouped") == "grouped":
        if _load_concept_groups(ckpt_dir) is None:
            print(
                "[warn] concept_groups.json not found in checkpoint dir; grouped-stat checkpoints "
                "from old runs may not be exactly reproducible in a fresh process.",
                flush=True,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    print(f"[data_dir] {args.data_dir}", flush=True)
    print(f"[split] {args.split}  batch_size={args.batch_size}", flush=True)

    model, loader, cfg, _ = fe._load_model_and_data(args, device)

    num_classes = _num_classes(args.data_dir)
    ls = float(cfg.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)

    avg_loss, acc, p, r, f1 = evaluate(
        model, loader, device, loss_fn, num_classes, sparsity_weight=0.0
    )

    tag = args.split
    print(
        f"[{tag}] loss={avg_loss:.4f} | acc={acc:.4f} | p={p:.4f} | r={r:.4f} | f1={f1:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
