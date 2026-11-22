#!/usr/bin/env python3
"""
Collect timing-perturbation sensitivity results for timing-high vs timing-low samples.

This script reuses eval/infer_pcap.py as the canonical single-PCAP inference path.
It can either:
1. select samples directly from data_dir + ckpt_dir (+ contrib cache if available), or
2. reuse an existing case_study_test.json.

Then it:
1. selects timing-high and timing-low correct samples
2. runs baseline + random-IAT jitter sweeps for each sample
3. aggregates the outputs into csv/json for case-study analysis

Example
-------
CUDA_VISIBLE_DEVICES=6 python eval/collect_timing_sensitivity.py \
  --ckpt_dir outputs/CipherSpectrum-AES128/20260320_171456 \
  --data_dir ./data/CipherSpectrum-AES128 \
  --flows_dir ./dataset/CipherSpectrum-AES128/flows_filtered \
  --output_dir outputs/contrib/CipherSpectrum-AES128/timing_sensitivity \
  --split test \
  --cache_dir outputs/contrib/CipherSpectrum-AES128 \
  --jitter_stds 0.05,0.10,0.20 \
  --top_n 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(ROOT, "eval")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import analyze_contrib as ac
import faithfulness_eval as fe


TERM_RE = re.compile(r"\s*(.+?)\(([+-]?\d+(?:\.\d+)?)\)\s*$")


def _parse_terms(text: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if not text:
        return out
    for chunk in text.split("|"):
        m = TERM_RE.match(chunk.strip())
        if not m:
            continue
        out.append((m.group(1).strip(), float(m.group(2))))
    return out


def _term_rank(terms: Sequence[Tuple[str, float]], name: str) -> Optional[int]:
    for i, (term_name, _) in enumerate(terms, start=1):
        if term_name == name:
            return i
    return None


def _term_value(terms: Sequence[Tuple[str, float]], name: str) -> Optional[float]:
    for term_name, value in terms:
        if term_name == name:
            return float(value)
    return None


def _load_label_map(data_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(os.path.join(data_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = {str(k): int(v) for k, v in json.load(f).items()}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def _load_flow_ids(data_dir: str, split: str) -> List[str]:
    with open(os.path.join(data_dir, f"{split}_ids.txt"), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _softmax_np(logits: Sequence[float]) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - arr.max()
    ex = np.exp(arr)
    return ex / ex.sum()


def _resolve_flows_dir(flows_dir: Optional[str], data_dir: str) -> str:
    if flows_dir:
        return os.path.abspath(flows_dir)
    dataset_name = os.path.basename(os.path.abspath(data_dir.rstrip("/")))
    guess = os.path.join(os.path.dirname(ROOT), "dataset", dataset_name, "flows_filtered")
    if not os.path.isdir(guess):
        raise FileNotFoundError(
            f"Could not infer flows_dir from data_dir={data_dir}. "
            f"Please pass --flows_dir explicitly."
        )
    return guess


def _resolve_pcap_path(flows_dir: str, flow_id: str) -> str:
    label, stem = flow_id.split("/", 1)
    path = os.path.join(flows_dir, label, f"{stem}.pcap")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PCAP not found for flow_id={flow_id}: {path}")
    return path


def _pick_samples(
    samples: Sequence[dict],
    top_n: int,
    min_conf: float,
    high_rank_max: int,
    low_bottom_k: int,
) -> Dict[str, List[dict]]:
    enriched = []
    for sample in samples:
        if not bool(sample.get("correct", False)):
            continue
        conf = float(sample.get("confidence", 0.0))
        if conf < min_conf:
            continue
        top_terms = _parse_terms(str(sample.get("top_concepts", "")))
        main_terms = _parse_terms(str(sample.get("top_main_effects", "")))
        timing_rank_top = _term_rank(top_terms, "seq/timing")
        timing_rank_main = _term_rank(main_terms, "seq/timing")
        timing_val_top = _term_value(top_terms, "seq/timing")
        timing_val_main = _term_value(main_terms, "seq/timing")
        enriched.append(
            {
                **sample,
                "timing_rank_top": timing_rank_top,
                "timing_rank_main": timing_rank_main,
                "timing_val_top": timing_val_top,
                "timing_val_main": timing_val_main,
            }
        )

    high = [
        s for s in enriched
        if (
            (s["timing_rank_top"] is not None and s["timing_rank_top"] <= high_rank_max) or
            (s["timing_rank_main"] is not None and s["timing_rank_main"] <= high_rank_max)
        )
    ]
    high.sort(
        key=lambda s: (
            s["timing_rank_top"] if s["timing_rank_top"] is not None else 999,
            s["timing_rank_main"] if s["timing_rank_main"] is not None else 999,
            -abs(s["timing_val_top"] or 0.0),
            -abs(s["timing_val_main"] or 0.0),
            -float(s["confidence"]),
        )
    )

    low = [
        s for s in enriched
        if (
            s.get("timing_rank_all") is not None and
            s.get("num_concepts") is not None and
            int(s["timing_rank_all"]) >= int(s["num_concepts"]) - int(low_bottom_k) + 1
        )
    ]
    low.sort(
        key=lambda s: (
            -float(s["confidence"]),
            -float(s["margin"]),
            -(int(s["timing_rank_all"]) if s.get("timing_rank_all") is not None else -1),
            abs(s["timing_val_top"] or 0.0),
            abs(s["timing_val_main"] or 0.0),
        )
    )

    def _dedup_by_label(items: Iterable[dict]) -> List[dict]:
        seen = set()
        out = []
        for item in items:
            label = str(item["true_label"])
            if label in seen:
                continue
            seen.add(label)
            out.append(item)
            if len(out) >= top_n:
                break
        return out

    return {
        "timing_high": _dedup_by_label(high),
        "timing_low": _dedup_by_label(low),
    }


def _load_cfg(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
        return fe._flatten_config(json.load(f))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def _format_terms(values: np.ndarray, names: Sequence[str], top_k: int) -> str:
    if values.size == 0:
        return ""
    order = np.argsort(-np.abs(values))[:top_k]
    return " | ".join(f"{names[i]}({values[i]:+.4f})" for i in order)


def _load_or_compute_samples(
    *,
    ckpt_dir: str,
    data_dir: str,
    split: str,
    cache_dir: Optional[str],
    output_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    top_concepts: int,
) -> List[dict]:
    cfg = _load_cfg(ckpt_dir)
    flow_ids = _load_flow_ids(data_dir, split)
    cache_root = cache_dir or output_dir
    prefix = os.path.join(cache_root, f"contrib_{split}")
    logits_path = prefix + "_logits.npy"
    labels_path = prefix + "_labels.npy"
    concepts_path = prefix + "_concepts.npy"
    deltas_path = prefix + "_ablation_delta.npy"

    if all(os.path.exists(p) for p in (logits_path, labels_path, concepts_path, deltas_path)):
        logits = np.load(logits_path)
        labels = np.load(labels_path)
        deltas = np.load(deltas_path)
    else:
        class _Args:
            pass
        a = _Args()
        a.ckpt_dir = ckpt_dir
        a.ckpt = None
        a.config = None
        a.data_dir = data_dir
        a.split = split
        a.batch_size = batch_size
        a.num_workers = num_workers
        a.seed = seed
        a.target = "true"
        a.baseline = "mean"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, loader, cfg_runtime, _ = fe._load_model_and_data(a, device)
        cfg = cfg_runtime
        logits_t, labels_t, concepts_t = ac.collect_outputs(model, loader, device)
        logits = logits_t.numpy()
        labels = labels_t.numpy()
        baseline = concepts_t.mean(dim=0)
        deltas = ac.ablation_contrib(model, concepts_t.to(device), labels_t.to(device), baseline.to(device)).numpy()

    if len(flow_ids) != len(labels):
        raise RuntimeError(f"Mismatch between ids ({len(flow_ids)}) and labels ({len(labels)})")

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
    _, id2label = _load_label_map(data_dir)
    probs = _softmax_rows(logits)
    preds = logits.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    conf = probs.max(axis=1)
    true_prob = probs[np.arange(labels.shape[0]), labels]

    rows: List[dict] = []
    for idx in range(len(flow_ids)):
        rows.append(
            {
                "idx": int(idx),
                "flow_id": flow_ids[idx],
                "true_label": id2label[int(labels[idx])],
                "pred_label": id2label[int(preds[idx])],
                "correct": bool(preds[idx] == labels[idx]),
                "confidence": float(conf[idx]),
                "margin": float(margin[idx]),
                "true_prob": float(true_prob[idx]),
                "top_concepts": _format_terms(deltas[idx], concept_names, top_concepts),
                "top_main_effects": "",
                "top_pair_effects": "",
                "timing_rank_all": (
                    int(np.argsort(-np.abs(deltas[idx])).tolist().index(concept_names.index("seq/timing")) + 1)
                    if "seq/timing" in concept_names else None
                ),
                "num_concepts": int(len(concept_names)),
            }
        )
    return rows


def _extract_concept_value(infer_json: dict, concept_name: str) -> Optional[float]:
    names = infer_json.get("concept_names")
    c_final = infer_json.get("c_final")
    if isinstance(names, list) and isinstance(c_final, list):
        try:
            idx = names.index(concept_name)
            return float(c_final[idx])
        except Exception:
            return None
    if concept_name == "seq/timing":
        vals = infer_json.get("c_seq")
        if isinstance(vals, list) and len(vals) >= 2:
            return float(vals[1])
    return None


def _run_infer(
    python_bin: str,
    infer_script: str,
    pcap_path: str,
    flow_id: str,
    ckpt_dir: str,
    data_dir: str,
    out_json: str,
    jitter_std: float,
    jitter_seed: int,
) -> dict:
    cmd = [
        python_bin,
        infer_script,
        "--pcap", pcap_path,
        "--ckpt_dir", ckpt_dir,
        "--data_dir", data_dir,
        "--flow_id", flow_id,
        "--output_json", out_json,
        "--timestamp_jitter_std", f"{jitter_std:.10f}",
        "--timestamp_jitter_seed", str(jitter_seed),
    ]
    subprocess.run(cmd, check=True)
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_run(
    sample: dict,
    infer_json: dict,
    label2id: Dict[str, int],
    group: str,
    jitter_std: float,
    jitter_seed: int,
) -> dict:
    logits = np.asarray(infer_json["logits"], dtype=np.float64)
    probs = _softmax_np(logits)
    pred_idx = int(np.argmax(probs))
    true_idx = int(label2id[str(sample["true_label"])])
    top2 = np.sort(probs)[-2:]
    margin = float(top2[-1] - top2[-2]) if probs.size >= 2 else float("nan")
    return {
        "group": group,
        "idx": int(sample["idx"]),
        "flow_id": str(sample["flow_id"]),
        "true_label": str(sample["true_label"]),
        "pred_label": str(infer_json["pred_label"]),
        "correct": bool(str(sample["true_label"]) == str(infer_json["pred_label"])),
        "jitter_std": float(jitter_std),
        "jitter_seed": int(jitter_seed),
        "pred_conf": float(probs[pred_idx]),
        "true_prob": float(probs[true_idx]),
        "margin": margin,
        "seq_timing": _extract_concept_value(infer_json, "seq/timing"),
        "seq_volume": _extract_concept_value(infer_json, "seq/volume"),
        "bytes_cross_middle": _extract_concept_value(infer_json, "bytes/cross_middle"),
        "stat_packet_time": _extract_concept_value(infer_json, "stat/packet_time"),
        "stat_flow_bytes": _extract_concept_value(infer_json, "stat/flow_bytes"),
        "raw_iat_mean": float(infer_json.get("raw_seq_stats", {}).get("iat_mean"))
        if isinstance(infer_json.get("raw_seq_stats"), dict)
        else None,
        "timing_rank_top": sample.get("timing_rank_top"),
        "timing_rank_main": sample.get("timing_rank_main"),
        "timing_val_top": sample.get("timing_val_top"),
        "timing_val_main": sample.get("timing_val_main"),
    }


def _write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _build_aggregate_rows(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    if not rows:
        return [], []

    base_by_sample: Dict[Tuple[str, int, int], dict] = {}
    for row in rows:
        if float(row["jitter_std"]) == 0.0:
            base_by_sample[(str(row["group"]), int(row["idx"]), int(row["jitter_seed"]))] = row

    enriched: List[dict] = []
    for row in rows:
        item = dict(row)
        key = (str(row["group"]), int(row["idx"]), int(row["jitter_seed"]))
        base = base_by_sample.get(key)
        if base is not None:
            item["pred_conf_drop"] = float(base["pred_conf"]) - float(row["pred_conf"])
            item["true_prob_drop"] = float(base["true_prob"]) - float(row["true_prob"])
            item["margin_drop"] = float(base["margin"]) - float(row["margin"])
            seq_timing = row.get("seq_timing")
            base_timing = base.get("seq_timing")
            item["seq_timing_shift"] = (
                float(seq_timing) - float(base_timing)
                if seq_timing is not None and base_timing is not None else None
            )
            bytes_mid = row.get("bytes_cross_middle")
            base_bytes_mid = base.get("bytes_cross_middle")
            item["bytes_cross_middle_shift"] = (
                float(bytes_mid) - float(base_bytes_mid)
                if bytes_mid is not None and base_bytes_mid is not None else None
            )
        else:
            item["pred_conf_drop"] = None
            item["true_prob_drop"] = None
            item["margin_drop"] = None
            item["seq_timing_shift"] = None
            item["bytes_cross_middle_shift"] = None
        enriched.append(item)

    group_bins: Dict[Tuple[str, float], List[dict]] = {}
    sample_bins: Dict[Tuple[str, int, str, float], List[dict]] = {}
    for row in enriched:
        group_bins.setdefault((str(row["group"]), float(row["jitter_std"])), []).append(row)
        sample_bins.setdefault(
            (str(row["group"]), int(row["idx"]), str(row["true_label"]), float(row["jitter_std"])),
            []
        ).append(row)

    group_rows: List[dict] = []
    for (group, jitter_std), items in sorted(group_bins.items(), key=lambda x: (x[0][0], x[0][1])):
        group_rows.append(
            {
                "group": group,
                "jitter_std": jitter_std,
                "num_runs": len(items),
                "avg_pred_conf": _mean_or_none([r.get("pred_conf") for r in items]),
                "avg_true_prob": _mean_or_none([r.get("true_prob") for r in items]),
                "avg_margin": _mean_or_none([r.get("margin") for r in items]),
                "avg_pred_conf_drop": _mean_or_none([r.get("pred_conf_drop") for r in items]),
                "avg_true_prob_drop": _mean_or_none([r.get("true_prob_drop") for r in items]),
                "avg_margin_drop": _mean_or_none([r.get("margin_drop") for r in items]),
                "avg_seq_timing": _mean_or_none([r.get("seq_timing") for r in items]),
                "avg_seq_timing_shift": _mean_or_none([r.get("seq_timing_shift") for r in items]),
                "avg_bytes_cross_middle_shift": _mean_or_none([r.get("bytes_cross_middle_shift") for r in items]),
                "flips": sum(1 for r in items if not bool(r["correct"])),
            }
        )

    sample_rows: List[dict] = []
    for (group, idx, true_label, jitter_std), items in sorted(sample_bins.items(), key=lambda x: (x[0][0], x[0][1], x[0][3])):
        sample_rows.append(
            {
                "group": group,
                "idx": idx,
                "true_label": true_label,
                "jitter_std": jitter_std,
                "num_runs": len(items),
                "avg_pred_conf": _mean_or_none([r.get("pred_conf") for r in items]),
                "avg_true_prob": _mean_or_none([r.get("true_prob") for r in items]),
                "avg_margin": _mean_or_none([r.get("margin") for r in items]),
                "avg_pred_conf_drop": _mean_or_none([r.get("pred_conf_drop") for r in items]),
                "avg_true_prob_drop": _mean_or_none([r.get("true_prob_drop") for r in items]),
                "avg_margin_drop": _mean_or_none([r.get("margin_drop") for r in items]),
                "avg_seq_timing": _mean_or_none([r.get("seq_timing") for r in items]),
                "avg_seq_timing_shift": _mean_or_none([r.get("seq_timing_shift") for r in items]),
                "avg_bytes_cross_middle_shift": _mean_or_none([r.get("bytes_cross_middle_shift") for r in items]),
                "flips": sum(1 for r in items if not bool(r["correct"])),
            }
        )

    return group_rows, sample_rows


def main() -> None:
    ap = argparse.ArgumentParser("Collect timing perturbation sensitivity results")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--flows_dir", default=None, help="flows_filtered root; inferred when possible")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--case_json", default=None, help="optional existing case_study_test.json; otherwise select directly from data/cache")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--cache_dir", default=None, help="directory containing contrib_<split>_*.npy")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--infer_script", default=os.path.join(EVAL_DIR, "infer_pcap.py"))
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--jitter_stds", default="0.05,0.10,0.20", help="comma-separated stds in seconds")
    ap.add_argument("--jitter_seed", type=int, default=42, help="single random seed, used when --jitter_seeds is not provided")
    ap.add_argument("--jitter_seeds", default=None, help="comma-separated random seeds to average over, e.g. 40,41,42,43,44")
    ap.add_argument("--top_n", type=int, default=3, help="number of timing-high and timing-low samples")
    ap.add_argument("--min_conf", type=float, default=0.95)
    ap.add_argument("--high_rank_max", type=int, default=3)
    ap.add_argument("--low_bottom_k", type=int, default=5, help="select timing-low samples whose timing rank is within the bottom k concepts")
    ap.add_argument("--top_concepts", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs_dir = os.path.join(args.output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    flows_dir = _resolve_flows_dir(args.flows_dir, args.data_dir)
    label2id, _ = _load_label_map(args.data_dir)

    if args.case_json:
        with open(args.case_json, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = _load_or_compute_samples(
            ckpt_dir=args.ckpt_dir,
            data_dir=args.data_dir,
            split=args.split,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            top_concepts=args.top_concepts,
        )

    groups = _pick_samples(
        samples=samples,
        top_n=args.top_n,
        min_conf=args.min_conf,
        high_rank_max=args.high_rank_max,
        low_bottom_k=args.low_bottom_k,
    )

    selection_path = os.path.join(args.output_dir, "selected_samples.json")
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    jitter_stds = [0.0] + [float(x) for x in args.jitter_stds.split(",") if x.strip()]
    jitter_seeds = (
        [int(x) for x in args.jitter_seeds.split(",") if x.strip()]
        if args.jitter_seeds else [int(args.jitter_seed)]
    )
    all_rows: List[dict] = []

    for group_name, group_samples in groups.items():
        for sample in group_samples:
            pcap_path = _resolve_pcap_path(flows_dir, str(sample["flow_id"]))
            for jitter_std in jitter_stds:
                seeds_for_this_jitter = [jitter_seeds[0]] if jitter_std == 0.0 else jitter_seeds
                for jitter_seed in seeds_for_this_jitter:
                    safe_std = str(jitter_std).replace(".", "p")
                    out_json = os.path.join(
                        runs_dir,
                        f"{group_name}_idx{int(sample['idx'])}_jitter{safe_std}_seed{jitter_seed}.json",
                    )
                    infer_json = _run_infer(
                        python_bin=args.python_bin,
                        infer_script=args.infer_script,
                        pcap_path=pcap_path,
                        flow_id=str(sample["flow_id"]),
                        ckpt_dir=args.ckpt_dir,
                        data_dir=args.data_dir,
                        out_json=out_json,
                        jitter_std=jitter_std,
                        jitter_seed=jitter_seed,
                    )
                    row = _summarize_run(
                        sample=sample,
                        infer_json=infer_json,
                        label2id=label2id,
                        group=group_name,
                        jitter_std=jitter_std,
                        jitter_seed=jitter_seed,
                    )
                    all_rows.append(row)

    csv_path = os.path.join(args.output_dir, "timing_sensitivity.csv")
    _write_csv(csv_path, all_rows)

    json_path = os.path.join(args.output_dir, "timing_sensitivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    group_rows, sample_rows = _build_aggregate_rows(all_rows)
    group_csv_path = os.path.join(args.output_dir, "timing_sensitivity_group_mean.csv")
    sample_csv_path = os.path.join(args.output_dir, "timing_sensitivity_sample_mean.csv")
    _write_csv(group_csv_path, group_rows)
    _write_csv(sample_csv_path, sample_rows)

    lines = []
    lines.append("Timing Sensitivity Collection")
    lines.append(f"flows_dir: {flows_dir}")
    lines.append(f"split: {args.split}")
    lines.append(f"selected_samples: {selection_path}")
    lines.append("jitter_seeds: " + ",".join(str(x) for x in jitter_seeds))
    lines.append("")
    for group_name, group_samples in groups.items():
        lines.append(f"[{group_name}]")
        for sample in group_samples:
            lines.append(
                f"idx={int(sample['idx'])}  label={sample['true_label']}  "
                f"conf={float(sample['confidence']):.4f}  "
                f"timing_rank_top={sample.get('timing_rank_top')}  "
                f"timing_rank_main={sample.get('timing_rank_main')}  "
                f"timing_rank_all={sample.get('timing_rank_all')}/{sample.get('num_concepts')}"
            )
        lines.append("")
    txt_path = os.path.join(args.output_dir, "timing_sensitivity_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[save] selected -> {selection_path}")
    print(f"[save] csv -> {csv_path}")
    print(f"[save] json -> {json_path}")
    print(f"[save] group_mean_csv -> {group_csv_path}")
    print(f"[save] sample_mean_csv -> {sample_csv_path}")
    print(f"[save] txt -> {txt_path}")


if __name__ == "__main__":
    main()
