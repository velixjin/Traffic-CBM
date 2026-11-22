#!/usr/bin/env python3
"""
Select instance-level case-study samples for Traffic-CBM.

This script mirrors the checkpoint/data arguments used by faithfulness_eval.py,
but focuses on instance selection instead of aggregate robustness metrics.

It groups candidate samples into:
1. confident correct
2. borderline correct
3. misclassified

For each selected sample it reports:
- flow id / true label / predicted label
- confidence, margin, true-class probability
- top concept ablation contributions
- optionally top GA2M main effects and pair interactions

Example
-------
CUDA_VISIBLE_DEVICES=1 python eval/case_study_samples.py \
  --ckpt_dir outputs/CipherSpectrum-AES128/20260320_171456 \
  --data_dir ./data/CipherSpectrum-AES128 \
  --split test \
  --output_dir outputs/contrib/CipherSpectrum-AES128 \
  --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import numpy as np
import torch

import analyze_contrib as ac
import faithfulness_eval as fe
from model.traffic_cbm import GA2MHead


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def _load_id_maps(data_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    path = os.path.join(data_dir, "label2id.json")
    with open(path, "r", encoding="utf-8") as f:
        label2id = {str(k): int(v) for k, v in json.load(f).items()}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def _load_flow_ids(data_dir: str, split: str) -> List[str]:
    path = os.path.join(data_dir, f"{split}_ids.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _format_terms(
    values: np.ndarray,
    names: Sequence[str],
    top_k: int,
    by_abs: bool = True,
) -> List[str]:
    if values.size == 0:
        return []
    score = np.abs(values) if by_abs else values
    order = np.argsort(-score)[:top_k]
    return [f"{names[i]}({values[i]:+.4f})" for i in order]


def _pair_term_names(concept_names: Sequence[str], pair_indices: Sequence[Tuple[int, int]]) -> List[str]:
    return [f"{concept_names[i]} x {concept_names[j]}" for i, j in pair_indices]


def _pick_unique(
    indices: Iterable[int],
    labels: np.ndarray,
    preds: np.ndarray,
    mode: str,
    limit: int,
) -> List[int]:
    picked: List[int] = []
    seen = set()
    for idx in indices:
        key = int(labels[idx]) if mode != "misclassified" else (int(labels[idx]), int(preds[idx]))
        if key in seen:
            continue
        seen.add(key)
        picked.append(int(idx))
        if len(picked) >= limit:
            break
    return picked


def _candidate_indices(
    probs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    limit_per_group: int,
) -> Dict[str, List[int]]:
    conf = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    true_prob = probs[np.arange(labels.shape[0]), labels]
    correct = preds == labels

    confident_pool = sorted(np.where(correct)[0].tolist(), key=lambda i: (conf[i], margin[i]), reverse=True)
    borderline_pool = sorted(np.where(correct)[0].tolist(), key=lambda i: (margin[i], conf[i]))
    misclassified_pool = sorted(np.where(~correct)[0].tolist(), key=lambda i: (true_prob[i], margin[i]))

    return {
        "confident_correct": _pick_unique(confident_pool, labels, preds, "confident", limit_per_group),
        "borderline_correct": _pick_unique(borderline_pool, labels, preds, "borderline", limit_per_group),
        "misclassified": _pick_unique(misclassified_pool, labels, preds, "misclassified", limit_per_group),
    }


def _class_stats(labels: np.ndarray, preds: np.ndarray, id2label: Dict[int, str]) -> List[str]:
    lines = []
    for c in sorted(np.unique(labels).tolist()):
        mask = labels == c
        acc = float((preds[mask] == labels[mask]).mean())
        n = int(mask.sum())
        err = int((preds[mask] != labels[mask]).sum())
        lines.append(f"{id2label[int(c)]:<24} acc={acc:.3f}  n={n:>3d}  errors={err:>3d}")
    lines.sort(key=lambda x: float(x.split("acc=")[1].split()[0]))
    return lines


def main():
    ap = argparse.ArgumentParser("Select case-study samples for TrafficCBM")
    ap.add_argument("--ckpt_dir", default=None, help="directory containing best.pt and config.json")
    ap.add_argument("--ckpt", default=None, help="checkpoint path")
    ap.add_argument("--config", default=None, help="config.json path")
    ap.add_argument("--data_dir", required=True, help="data directory for the target dataset")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target", default="true", choices=["true", "pred"], help="which class logit to attribute")
    ap.add_argument("--baseline", default="mean", choices=["mean", "zero"])
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_group", type=int, default=5, help="number of selected samples per bucket")
    ap.add_argument("--top_concepts", type=int, default=5, help="top concepts to print for each sample")
    ap.add_argument("--top_pairs", type=int, default=3, help="top GA2M pair interactions to print")
    ap.add_argument("--cache_dir", default=None, help="directory containing contrib_<split>_*.npy")
    ap.add_argument("--ga2m_details", action="store_true",
                    help="load the checkpoint and compute GA2M main/pair terms for selected samples")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir or args.cache_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    label2id, id2label = _load_id_maps(args.data_dir)
    flow_ids = _load_flow_ids(args.data_dir, args.split)
    cache_dir = args.cache_dir or output_dir

    cfg_path = args.config or (os.path.join(args.ckpt_dir, "config.json") if args.ckpt_dir else None)
    if not cfg_path:
        raise ValueError("Need --ckpt_dir or --config so concept names can be resolved.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = fe._flatten_config(json.load(f))

    prefix = os.path.join(cache_dir, f"contrib_{args.split}")
    logits_path = prefix + "_logits.npy"
    labels_path = prefix + "_labels.npy"
    concepts_path = prefix + "_concepts.npy"
    deltas_path = prefix + "_ablation_delta.npy"

    if all(os.path.exists(p) for p in (logits_path, labels_path, concepts_path, deltas_path)):
        logits = np.load(logits_path)
        labels = np.load(labels_path)
        concepts_np = np.load(concepts_path)
        deltas = np.load(deltas_path)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, loader, cfg_runtime, _ = fe._load_model_and_data(args, device)
        cfg = cfg_runtime
        logits_t, labels_t, concepts_t = ac.collect_outputs(model, loader, device)
        logits = logits_t.numpy()
        labels = labels_t.numpy()
        concepts_np = concepts_t.numpy()
        target_idx = labels_t if args.target == "true" else logits_t.argmax(dim=1)
        baseline = concepts_t.mean(dim=0) if args.baseline == "mean" else torch.zeros(concepts_t.size(1), device=device)
        deltas = ac.ablation_contrib(model, concepts_t.to(device), target_idx.to(device), baseline).numpy()

    if len(flow_ids) != len(labels):
        raise RuntimeError(
            f"Mismatch between ids ({len(flow_ids)}) and model outputs ({len(labels)})."
        )

    preds = logits.argmax(axis=1)
    probs = _softmax_np(logits)
    conf = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    true_prob = probs[np.arange(labels.shape[0]), labels]
    concept_names = ac._concept_names(
        cfg.get("fusion_mode", "6+1+1"),
        num_packets=int(cfg.get("num_packets", 5)),
        num_groups=int(cfg.get("num_groups", 3)),
        use_stat=cfg.get("use_stat", True),
        use_seq=cfg.get("use_seq", True),
        use_bytes=cfg.get("use_bytes", True),
        use_bytes_packet=cfg.get("use_bytes_packet", False),
        use_bytes_cross=cfg.get("use_bytes_cross", False),
    )

    main_vals = pair_vals = None
    pair_term_names: List[str] = []
    selected = _candidate_indices(probs, labels, preds, args.per_group)
    selected_flat = [idx for bucket in selected.values() for idx in bucket]

    if args.ga2m_details and selected_flat:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, cfg_runtime, _ = fe._load_model_and_data(args, device)
        cfg = cfg_runtime
        concepts_sel = torch.from_numpy(concepts_np[selected_flat]).to(device)
        target_sel = torch.from_numpy(
            labels[selected_flat] if args.target == "true" else preds[selected_flat]
        ).to(device)
        if isinstance(model.head, GA2MHead):
            exact = ac.ga2m_exact_contrib(model, concepts_sel, target_sel)
            if exact is not None:
                main_t, pair_t, pair_indices = exact
                main_vals = main_t.numpy()
                pair_vals = pair_t.numpy()
                pair_term_names = _pair_term_names(concept_names, pair_indices)
        selected_pos = {idx: pos for pos, idx in enumerate(selected_flat)}
    else:
        selected_pos = {}

    rows = []
    for bucket, idxs in selected.items():
        for idx in idxs:
            row = {
                "bucket": bucket,
                "idx": int(idx),
                "flow_id": flow_ids[idx],
                "true_label": id2label[int(labels[idx])],
                "pred_label": id2label[int(preds[idx])],
                "correct": bool(preds[idx] == labels[idx]),
                "confidence": float(conf[idx]),
                "margin": float(margin[idx]),
                "true_prob": float(true_prob[idx]),
                "top_concepts": " | ".join(_format_terms(deltas[idx], concept_names, args.top_concepts)),
                "top_main_effects": "",
                "top_pair_effects": "",
            }
            pos = selected_pos.get(idx)
            if main_vals is not None and pos is not None:
                row["top_main_effects"] = " | ".join(
                    _format_terms(main_vals[pos], concept_names, args.top_concepts)
                )
            if pair_vals is not None and pair_vals.shape[1] > 0 and pos is not None:
                row["top_pair_effects"] = " | ".join(
                    _format_terms(pair_vals[pos], pair_term_names, args.top_pairs)
                )
            rows.append(row)

    out_prefix = os.path.join(output_dir, f"case_study_{args.split}")
    csv_path = out_prefix + ".csv"
    txt_path = out_prefix + ".txt"
    json_path = out_prefix + ".json"

    # CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        header = [
            "bucket",
            "idx",
            "flow_id",
            "true_label",
            "pred_label",
            "correct",
            "confidence",
            "margin",
            "true_prob",
            "top_concepts",
            "top_main_effects",
            "top_pair_effects",
        ]
        f.write(",".join(header) + "\n")
        for row in rows:
            vals = []
            for key in header:
                text = str(row[key]).replace('"', '""')
                vals.append(f'"{text}"')
            f.write(",".join(vals) + "\n")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Text report
    lines = []
    lines.append("=" * 96)
    lines.append("Case Study Sample Selection")
    lines.append("=" * 96)
    lines.append(f"checkpoint = {args.ckpt_dir or args.ckpt}")
    lines.append(f"data_dir   = {args.data_dir}")
    lines.append(f"split      = {args.split}")
    lines.append(f"samples    = {len(labels)}")
    lines.append(f"accuracy   = {(preds == labels).mean():.4f}")
    lines.append(f"cache_dir  = {cache_dir}")
    lines.append(f"ga2m_terms = {args.ga2m_details}")
    lines.append("")
    lines.append("Lowest-accuracy classes")
    lines.append("-" * 96)
    for line in _class_stats(labels, preds, id2label)[: min(12, len(np.unique(labels)))]:
        lines.append(line)

    for bucket in ("confident_correct", "borderline_correct", "misclassified"):
        lines.append("")
        lines.append(bucket)
        lines.append("-" * 96)
        for row in rows:
            if row["bucket"] != bucket:
                continue
            lines.append(
                f"[idx={row['idx']}] {row['true_label']} -> {row['pred_label']}  "
                f"correct={row['correct']}  conf={row['confidence']:.4f}  "
                f"margin={row['margin']:.4f}  true_prob={row['true_prob']:.4f}"
            )
            lines.append(f"flow_id: {row['flow_id']}")
            lines.append(f"top_concepts: {row['top_concepts']}")
            if row["top_main_effects"]:
                lines.append(f"top_main_effects: {row['top_main_effects']}")
            if row["top_pair_effects"]:
                lines.append(f"top_pair_effects: {row['top_pair_effects']}")
            lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    print(report, flush=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[save] report -> {txt_path}", flush=True)
    print(f"[save] csv    -> {csv_path}", flush=True)
    print(f"[save] json   -> {json_path}", flush=True)


if __name__ == "__main__":
    main()
