#!/usr/bin/env python3
"""
Infer a single PCAP directly with Traffic-CBM.

This script is a deployment-style single entrypoint:
1. read one pcap
2. extract stats / seq / bytes in memory
3. apply the same scalers as the checkpoint
4. run the model
5. print prediction and concept values

It does not depend on parquet / manifest generation for the target sample.

Example
-------
CUDA_VISIBLE_DEVICES=5 python eval/infer_pcap.py \
  --pcap outputs/contrib/CipherSpectrum-AES128/garmin_idx605_filtered.pcap \
  --ckpt_dir outputs/CipherSpectrum-AES128/20260320_171456 \
  --data_dir ./data/CipherSpectrum-AES128
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import numpy as np
import torch
from scapy.utils import rdpcap, wrpcap  # type: ignore

import analyze_contrib as ac
from data_process.build_bytes import packets_to_tokens
from data_process.build_seq import extract_seq
from data_process.build_stats import _process_one as _extract_stats_worker, VENDOR_DIR
from datasets.traffic_dataset import (
    FEATURE_SCALERS_FILENAME,
    SEQ_CONT_IDX,
    fit_or_load_scalers,
    _load_stats,
)
from model.traffic_cbm import GA2MHead, build_model
from train.train import (
    _flatten_config,
    _get_stat_info,
    _num_classes,
    _resolve_bytes_branch_flags,
    _resolve_concept_groups,
)


def _load_cfg(ckpt_dir: Optional[str], config_path: Optional[str]) -> dict:
    path = config_path or (os.path.join(ckpt_dir, "config.json") if ckpt_dir else None)
    if not path:
        raise ValueError("Need --ckpt_dir or --config")
    with open(path, "r", encoding="utf-8") as f:
        return _flatten_config(json.load(f))


def _load_label_map(data_dir: str) -> Dict[int, str]:
    with open(os.path.join(data_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    return {int(v): str(k) for k, v in label2id.items()}


def _format_top_probs(probs: np.ndarray, id2label: Dict[int, str], top_k: int) -> List[str]:
    order = np.argsort(-probs)[:top_k]
    return [f"{id2label[int(i)]}: {probs[i]:.4f}" for i in order]


def _infer_flow_id(pcap_path: str, label: Optional[str]) -> str:
    stem = Path(pcap_path).stem
    if label:
        return f"{label}/{stem}"
    parent = Path(pcap_path).parent.name
    return f"{parent}/{stem}"


def _detect_bytes_sep(data_dir: str, flow_id: Optional[str] = None) -> Optional[bool]:
    path = os.path.join(data_dir, "bytes", "bytes.parquet")
    if not os.path.isfile(path):
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["flow_id", "tokens"])
        if df.empty:
            return None
        if flow_id is not None:
            sub = df[df["flow_id"] == flow_id]
            if not sub.empty:
                toks = list(sub.iloc[0]["tokens"])
                return 65537 in toks
        toks = list(df.iloc[0]["tokens"])
        return 65537 in toks
    except Exception:
        return None


def _extract_stat_vector(
    pcap_path: str,
    data_dir: str,
    stat_scaler,
    max_packets: int,
    flow_id: str,
) -> np.ndarray:
    header_all, feat_idx, feat_cols, _ = _load_stats(os.path.join(data_dir, "stats", "stats.parquet"))
    fid, hdr, row = _extract_stats_worker((flow_id, pcap_path, str(VENDOR_DIR), int(max_packets)))
    if hdr is None or row is None:
        raise RuntimeError(f"Failed to extract stats from {pcap_path}")
    row_map: Dict[str, float] = {}
    for name, value in zip(hdr, row):
        try:
            row_map[str(name)] = float(value)
        except Exception:
            row_map[str(name)] = 0.0
    x = np.array([row_map.get(col, 0.0) for col in feat_cols], dtype=np.float32)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if stat_scaler is not None:
        x = stat_scaler.transform(x)
    return x


def _extract_seq_tensor(
    pcap_path: str,
    seq_scaler,
    max_seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = extract_seq(Path(pcap_path), max_packets=0)
    if arr is None:
        raise RuntimeError(f"Failed to extract seq features from {pcap_path}")
    raw_arr = arr.copy()
    if seq_scaler is not None:
        cont = arr[:, SEQ_CONT_IDX].copy()
        arr = arr.copy()
        arr[:, SEQ_CONT_IDX] = seq_scaler.transform(cont)
    n = arr.shape[0]
    t = int(max_seq_len)
    if n >= t:
        x_seq = arr[:t]
        attn = np.ones(t, dtype=np.float32)
    else:
        pad = np.zeros((t - n, 7), dtype=np.float32)
        x_seq = np.concatenate([arr, pad], axis=0)
        attn = np.array([1.0] * n + [0.0] * (t - n), dtype=np.float32)
    return raw_arr, x_seq, attn


def _tokens_to_packet_tensor(
    toks: Sequence[int],
    num_packets: int,
    pkt_len: int,
    sep_token: int,
    pad_token: int,
) -> Tuple[np.ndarray, np.ndarray]:
    toks = np.asarray(toks, dtype=np.int64)
    segs: List[np.ndarray] = []
    if sep_token in toks:
        start = 0
        for i in range(len(toks)):
            if toks[i] == sep_token:
                segs.append(np.asarray(toks[start:i], dtype=np.int64))
                start = i + 1
        if start < len(toks):
            segs.append(np.asarray(toks[start:], dtype=np.int64))
        segs = [s for s in segs if len(s) > 0][:num_packets]
    else:
        chunk = max(1, (len(toks) + num_packets - 1) // num_packets)
        for k in range(num_packets):
            s, e = k * chunk, min((k + 1) * chunk, len(toks))
            segs.append(np.asarray(toks[s:e], dtype=np.int64))

    rows = []
    masks = []
    for k in range(num_packets):
        if k < len(segs):
            seg = segs[k]
            if len(seg) >= pkt_len:
                row = seg[:pkt_len]
                msk = np.ones(pkt_len, dtype=np.int64)
            else:
                row = np.concatenate([seg, np.full(pkt_len - len(seg), pad_token, dtype=np.int64)])
                msk = np.array([1] * len(seg) + [0] * (pkt_len - len(seg)), dtype=np.int64)
        else:
            row = np.full(pkt_len, pad_token, dtype=np.int64)
            msk = np.zeros(pkt_len, dtype=np.int64)
        rows.append(row)
        masks.append(msk)
    return np.stack(rows), np.stack(masks)


def _extract_bytes_tensor(
    pcap_path: str,
    bytes_len: int,
    bytes_per_packet: bool,
    bytes_pkt_len: int,
    num_packets: int,
    pkt_n: int,
    payload_len: int,
    start_offset: int,
    insert_sep: bool,
    sanitize_mode: str,
    preserve_tcp_timestamp: bool,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    packets = rdpcap(str(pcap_path))
    toks = packets_to_tokens(
        packets,
        pkt_n=int(pkt_n),
        payload_len=int(payload_len),
        start_offset=int(start_offset),
        insert_sep=bool(insert_sep),
        sanitize_mode=str(sanitize_mode),
        preserve_tcp_timestamp=bool(preserve_tcp_timestamp),
    )
    if not toks:
        raise RuntimeError(f"Failed to extract bytes tokens from {pcap_path}")

    token_pad = 65536
    token_sep = 65537

    if bytes_per_packet:
        x_bytes, bytes_mask = _tokens_to_packet_tensor(
            toks=toks,
            num_packets=int(num_packets),
            pkt_len=int(bytes_pkt_len),
            sep_token=token_sep,
            pad_token=token_pad,
        )
    else:
        toks_np = np.asarray(toks, dtype=np.int64)
        l = int(bytes_len)
        if len(toks_np) >= l:
            x_bytes = toks_np[:l]
        else:
            x_bytes = np.concatenate([toks_np, np.full(l - len(toks_np), token_pad, dtype=np.int64)])
        bytes_mask = (x_bytes != token_pad).astype(np.int64)
    return list(map(int, toks)), x_bytes, bytes_mask


def _apply_timestamp_perturbation(
    packets,
    scale: float,
    shift: float,
    front_n: int,
    jitter_std: float,
    jitter_seed: int,
) -> int:
    if not packets or (
        abs(scale - 1.0) < 1e-12 and
        abs(shift) < 1e-12 and
        abs(jitter_std) < 1e-12
    ):
        return 0

    times = [float(getattr(pkt, "time", 0.0)) for pkt in packets]
    if len(times) <= 1:
        return 0

    rng = np.random.default_rng(int(jitter_seed))
    new_times = [times[0]]
    touched = 0
    for i in range(1, len(times)):
        old_iat = max(0.0, times[i] - times[i - 1])
        apply_here = front_n <= 0 or i < front_n
        if apply_here:
            new_iat = old_iat * float(scale) + float(shift)
            if abs(jitter_std) >= 1e-12:
                new_iat += float(rng.normal(loc=0.0, scale=float(jitter_std)))
            new_iat = max(1e-9, new_iat)
            touched += 1
        else:
            new_iat = old_iat
        new_times.append(new_times[-1] + new_iat)

    for pkt, new_t in zip(packets, new_times):
        pkt.time = new_t
    return touched


def main():
    ap = argparse.ArgumentParser("Infer a single PCAP with Traffic-CBM")
    ap.add_argument("--pcap", required=True, help="path to a single flow pcap")
    ap.add_argument("--ckpt_dir", default=None, help="directory containing best.pt and config.json")
    ap.add_argument("--ckpt", default=None, help="checkpoint path")
    ap.add_argument("--config", default=None, help="config.json path")
    ap.add_argument("--data_dir", required=True, help="dataset data directory for scalers / label map / stat schema")
    ap.add_argument("--label", default=None, help="optional label prefix used to construct flow_id")
    ap.add_argument("--flow_id", default=None, help="optional exact flow_id override (useful when pcap filename was renamed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=5, help="number of top predicted labels to print")
    ap.add_argument("--ga2m_terms", action="store_true", help="print GA2M main/pair terms for the predicted class")
    ap.add_argument("--output_json", default=None, help="optional path to save structured output json")
    ap.add_argument("--stats_max_packets", type=int, default=200)
    ap.add_argument("--seq_max_packets", type=int, default=0, help="0 means all packets, same as build_seq default")
    ap.add_argument("--bytes_pkt_n", type=int, default=5)
    ap.add_argument("--bytes_payload_len", type=int, default=64)
    ap.add_argument("--bytes_start_offset", type=int, default=38)
    ap.add_argument("--bytes_sep", action="store_true", help="force insert SEP between packets during bytes extraction")
    ap.add_argument("--bytes_sanitize_mode", default="none", choices=["none", "zero", "random"])
    ap.add_argument("--preserve_tcp_timestamp", action="store_true")
    ap.add_argument("--timestamp_scale", type=float, default=1.0, help="multiply packet IATs by this factor before inference")
    ap.add_argument("--timestamp_shift", type=float, default=0.0, help="add this many seconds to each affected IAT before inference")
    ap.add_argument("--timestamp_front_n", type=int, default=0, help="only perturb the first N packets' IATs; 0 means all")
    ap.add_argument("--timestamp_jitter_std", type=float, default=0.0, help="add Gaussian noise with this std (seconds) to each affected IAT")
    ap.add_argument("--timestamp_jitter_seed", type=int, default=42, help="random seed for timestamp jitter")

    pre, _ = ap.parse_known_args()
    proc_cfg = os.path.join(ROOT, "data_process", "config.json")
    if os.path.isfile(proc_cfg):
        with open(proc_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        defaults = {
            "stats_max_packets": cfg.get("stats", {}).get("max_packets", 200),
            "seq_max_packets": cfg.get("seq", {}).get("max_packets", 0),
            "bytes_pkt_n": cfg.get("bytes", {}).get("pkt_n", 5),
            "bytes_payload_len": cfg.get("bytes", {}).get("payload_len", 64),
            "bytes_start_offset": cfg.get("bytes", {}).get("start_offset", 38),
            "bytes_sanitize_mode": cfg.get("bytes", {}).get("sanitize_mode", "none"),
            "preserve_tcp_timestamp": cfg.get("bytes", {}).get("preserve_tcp_timestamp", False),
        }
        ap.set_defaults(**defaults)
        if cfg.get("bytes", {}).get("sep"):
            import sys as _sys
            if "--bytes_sep" not in _sys.argv:
                _sys.argv.append("--bytes_sep")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = _load_cfg(args.ckpt_dir, args.config)
    ckpt_path = args.ckpt or (os.path.join(args.ckpt_dir, "best.pt") if args.ckpt_dir else None)
    if not ckpt_path:
        raise ValueError("Need --ckpt_dir or --ckpt")

    use_stat = cfg.get("use_stat", True)
    use_seq = cfg.get("use_seq", True)
    use_bytes = cfg.get("use_bytes", True)
    fusion_mode = cfg.get("fusion_mode", "6+1+1")
    use_bytes, use_bytes_packet, use_bytes_cross = _resolve_bytes_branch_flags(
        fusion_mode,
        bool(cfg.get("use_bytes", False)),
        bool(cfg.get("use_bytes_packet", False)),
        bool(cfg.get("use_bytes_cross", False)),
    )
    bytes_per_packet = bool(cfg.get("bytes_per_packet", False) or fusion_mode in ("6+4+5", "6+4+5+3", "6+4+3"))
    num_packets = int(cfg.get("num_packets", 5))
    num_groups = int(cfg.get("num_groups", 3))

    flow_id = args.flow_id or _infer_flow_id(args.pcap, args.label)
    id2label = _load_label_map(args.data_dir)

    detected_sep = _detect_bytes_sep(args.data_dir, flow_id)
    bytes_sep = bool(args.bytes_sep or detected_sep)

    stat_scaler, seq_scaler = fit_or_load_scalers(
        args.data_dir,
        os.path.join(args.data_dir, "train_ids.txt"),
        fit_stat=use_stat,
        fit_seq=use_seq,
        ckpt_dir=args.ckpt_dir,
    )

    stat_dim, _, _ = _get_stat_info(args.data_dir) if use_stat else (None, None, None)
    concept_mode = cfg.get("concept_mode", "grouped")
    concept_groups, _ = _resolve_concept_groups(args.data_dir, args.ckpt_dir, use_stat, concept_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=_num_classes(args.data_dir),
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

    sample_out: Dict[str, object] = {
        "pcap": os.path.abspath(args.pcap),
        "flow_id": flow_id,
        "fusion_mode": fusion_mode,
        "device": str(device),
        "bytes_sep": bytes_sep,
        "timestamp_scale": float(args.timestamp_scale),
        "timestamp_shift": float(args.timestamp_shift),
        "timestamp_front_n": int(args.timestamp_front_n),
        "timestamp_jitter_std": float(args.timestamp_jitter_std),
        "timestamp_jitter_seed": int(args.timestamp_jitter_seed),
    }

    kwargs = {}
    raw_seq = None
    raw_tokens = None

    with tempfile.TemporaryDirectory(prefix="infer_pcap_") as tmpdir:
        pcap_for_extract = args.pcap
        perturb_packets = None
        touched_iats = 0
        if (
            abs(args.timestamp_scale - 1.0) >= 1e-12 or
            abs(args.timestamp_shift) >= 1e-12 or
            abs(args.timestamp_jitter_std) >= 1e-12
        ):
            perturb_packets = rdpcap(str(args.pcap))
            touched_iats = _apply_timestamp_perturbation(
                perturb_packets,
                scale=args.timestamp_scale,
                shift=args.timestamp_shift,
                front_n=args.timestamp_front_n,
                jitter_std=args.timestamp_jitter_std,
                jitter_seed=args.timestamp_jitter_seed,
            )
            pcap_for_extract = os.path.join(tmpdir, Path(args.pcap).name)
            wrpcap(pcap_for_extract, perturb_packets)
            sample_out["timestamp_perturbed"] = True
            sample_out["timestamp_touched_iats"] = int(touched_iats)
        else:
            sample_out["timestamp_perturbed"] = False

        if use_stat:
            x_stat = _extract_stat_vector(
                pcap_path=pcap_for_extract,
                data_dir=args.data_dir,
                stat_scaler=stat_scaler,
                max_packets=args.stats_max_packets,
                flow_id=flow_id,
            )
            kwargs["x_stat"] = torch.from_numpy(x_stat).unsqueeze(0).to(device)
            sample_out["x_stat_shape"] = list(x_stat.shape)

        if use_seq:
            raw_seq, x_seq, attn_mask = _extract_seq_tensor(
                pcap_path=pcap_for_extract,
                seq_scaler=seq_scaler,
                max_seq_len=int(cfg.get("max_seq_len", 100)),
            )
            kwargs["x_seq"] = torch.from_numpy(x_seq).unsqueeze(0).to(device)
            kwargs["attn_mask"] = torch.from_numpy(attn_mask).unsqueeze(0).to(device)
            sample_out["raw_seq_packets"] = int(raw_seq.shape[0])
            sample_out["x_seq_shape"] = list(x_seq.shape)
            sample_out["raw_seq_stats"] = {
                "pktlen_mean": float(raw_seq[:, 0].mean()),
                "iat_mean": float(raw_seq[:, 1].mean()),
                "iat_std": float(raw_seq[:, 1].std()),
                "payload_mean": float(raw_seq[:, 4].mean()),
                "header_mean": float(raw_seq[:, 5].mean()),
            }

        if use_bytes:
            raw_tokens, x_bytes, bytes_mask = _extract_bytes_tensor(
                pcap_path=pcap_for_extract,
                bytes_len=int(cfg.get("bytes_len", 320)),
                bytes_per_packet=bytes_per_packet,
                bytes_pkt_len=int(cfg.get("bytes_pkt_len", 63)),
                num_packets=num_packets,
                pkt_n=args.bytes_pkt_n,
                payload_len=args.bytes_payload_len,
                start_offset=args.bytes_start_offset,
                insert_sep=bytes_sep,
                sanitize_mode=args.bytes_sanitize_mode,
                preserve_tcp_timestamp=args.preserve_tcp_timestamp,
            )
            kwargs["x_bytes"] = torch.from_numpy(x_bytes).unsqueeze(0).to(device)
            kwargs["bytes_mask"] = torch.from_numpy(bytes_mask).unsqueeze(0).to(device)
            sample_out["raw_token_count"] = int(len(raw_tokens))
            sample_out["x_bytes_shape"] = list(x_bytes.shape)

        with torch.no_grad():
            logits, extras = model(**kwargs)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(probs.argmax())

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

    sample_out.update(
        {
            "pred_label": id2label[pred_idx],
            "pred_conf": float(probs[pred_idx]),
            "top_probs": _format_top_probs(probs, id2label, args.top_k),
            "logits": logits[0].detach().cpu().numpy().tolist(),
        }
    )

    if "c_stat" in extras:
        sample_out["c_stat"] = extras["c_stat"][0].detach().cpu().numpy().tolist()
    if "c_seq" in extras:
        sample_out["c_seq"] = extras["c_seq"][0].detach().cpu().numpy().tolist()
        sample_out["seq_concept_names"] = [f"seq/{n}" for n in ("volume", "timing", "direction", "tcp")]
    if "c_bytes" in extras:
        sample_out["c_bytes"] = extras["c_bytes"][0].detach().cpu().numpy().tolist()
    if "c_final" in extras:
        c_final = extras["c_final"][0].detach().cpu().unsqueeze(0)
        sample_out["c_final"] = c_final[0].numpy().tolist()
        sample_out["concept_names"] = concept_names
        if args.ga2m_terms and isinstance(model.head, GA2MHead):
            target_idx = torch.tensor([pred_idx], device=device)
            exact = ac.ga2m_exact_contrib(model, c_final.to(device), target_idx)
            if exact is not None:
                main_vals, pair_vals, pair_indices = exact
                main_np = main_vals[0].numpy()
                pairs = []
                if pair_vals.numel() > 0:
                    for val, (i, j) in zip(pair_vals[0].numpy(), pair_indices):
                        pairs.append(
                            {
                                "name": f"{concept_names[i]} x {concept_names[j]}",
                                "value": float(val),
                            }
                        )
                    pairs.sort(key=lambda x: abs(x["value"]), reverse=True)
                main_terms = [
                    {"name": concept_names[i], "value": float(main_np[i])}
                    for i in np.argsort(-np.abs(main_np))
                ]
                sample_out["ga2m_main_terms"] = main_terms
                sample_out["ga2m_pair_terms"] = pairs

    lines = []
    lines.append("=" * 88)
    lines.append("Single PCAP Inference")
    lines.append("=" * 88)
    lines.append(f"pcap       : {sample_out['pcap']}")
    lines.append(f"flow_id    : {flow_id}")
    lines.append(f"device     : {device}")
    lines.append(f"fusion     : {fusion_mode}")
    lines.append(f"bytes_sep  : {bytes_sep}")
    lines.append(f"prediction : {sample_out['pred_label']}  conf={sample_out['pred_conf']:.4f}")
    lines.append("top_probs  : " + " | ".join(sample_out["top_probs"]))

    if raw_seq is not None:
        lines.append(
            "seq_raw    : "
            f"packets={raw_seq.shape[0]}  iat_mean={float(raw_seq[:,1].mean()):.6f}  "
            f"pktlen_mean={float(raw_seq[:,0].mean()):.2f}"
        )
    if raw_tokens is not None:
        lines.append(f"bytes_raw  : tokens={len(raw_tokens)}")

    if "c_stat" in sample_out:
        stat_names = [
            "stat/packet_count",
            "stat/packet_length",
            "stat/flow_bytes",
            "stat/packet_time",
            "stat/flag_count",
            "stat/header_window",
        ]
        vals = sample_out["c_stat"]
        lines.append("c_stat     : " + " | ".join(f"{n}={float(v):+.4f}" for n, v in zip(stat_names, vals)))
    if "c_seq" in sample_out:
        vals = sample_out["c_seq"]
        lines.append(
            "c_seq      : " + " | ".join(
                f"{n}={float(v):+.4f}" for n, v in zip(sample_out["seq_concept_names"], vals)
            )
        )
    if "c_bytes" in sample_out:
        vals = sample_out["c_bytes"]
        byte_names = [n for n in concept_names if n.startswith("bytes/")]
        lines.append("c_bytes    : " + " | ".join(f"{n}={float(v):+.4f}" for n, v in zip(byte_names, vals)))

    if args.ga2m_terms and "ga2m_main_terms" in sample_out:
        lines.append("ga2m_main  : " + " | ".join(
            f"{x['name']}({x['value']:+.4f})" for x in sample_out["ga2m_main_terms"][: args.top_k]
        ))
        if sample_out.get("ga2m_pair_terms"):
            lines.append("ga2m_pair  : " + " | ".join(
                f"{x['name']}({x['value']:+.4f})" for x in sample_out["ga2m_pair_terms"][: args.top_k]
            ))

    text = "\n".join(lines)
    print(text, flush=True)

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample_out, f, ensure_ascii=False, indent=2)
        print(f"[save] json -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
