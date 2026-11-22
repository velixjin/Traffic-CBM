#!/usr/bin/env python3
"""
datasets/traffic_dataset.py - Multi-modal traffic dataset.

Reads from data/{name}/ (self-contained after build_manifest):
  stats/stats.parquet   -> x_stat  [n_stat_features]
  seq/seq.parquet       -> x_seq   [max_seq_len, 7]  + attn_mask [max_seq_len]
  bytes/bytes.parquet   -> x_bytes [bytes_len]        + bytes_mask [bytes_len]
  label2id.json
  train_ids.txt / val_ids.txt / test_ids.txt

Usage:
  from datasets.traffic_dataset import TrafficDataset, fit_scalers, collate_fn

  stat_s, seq_s = fit_scalers(data_dir, train_ids_txt)
  ds = TrafficDataset(
      ids_txt  = "data/USTC-TFC2016/train_ids.txt",
      data_dir = "data/USTC-TFC2016",
      use_stat=True, use_seq=True, use_bytes=True,
      max_seq_len=100, bytes_len=320,
      stat_scaler=stat_s, seq_scaler=seq_s,
  )
  loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
"""
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Special tokens — must match build_bytes.py
TOKEN_PAD  = 65536
TOKEN_SEP  = 65537
TOKEN_MASK = 65538
TOKEN_CLS  = 65539

# Non-numeric metadata columns in stats.parquet to exclude from features
_STATS_META_COLS = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp"}

# Continuous feature indices within the 7-dim seq vector (used by seq_scaler)
# [pktlen, iat, direction, tcp_flags, payload_len, header_len, tcp_window]
SEQ_CONT_IDX = [0, 1, 4, 5, 6]  # direction and tcp_flags are discrete, skip scaling


# ---------------------------------------------------------------------------
# FeatureScaler (Welford online algorithm)
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Streaming standardizer using Welford's online algorithm."""

    def __init__(self, num_features: int):
        self.n = 0
        self.mean = np.zeros(num_features, dtype=np.float64)
        self.M2   = np.zeros(num_features, dtype=np.float64)
        self._finalized = False
        self.std  = np.ones(num_features, dtype=np.float64)

    def partial_fit(self, x: np.ndarray) -> None:
        """Update statistics with a 1-D or 2-D array (rows are samples)."""
        x2 = x.reshape(-1, x.shape[-1]).astype(np.float64)
        for row in x2:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.M2   += delta * (row - self.mean)

    def finalize(self) -> None:
        var = self.M2 / max(self.n - 1, 1)
        self.std = np.sqrt(np.maximum(var, 1e-12))
        self._finalized = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._finalized:
            self.finalize()
        return ((x - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# Parquet loaders (load-once into dicts for O(1) __getitem__)
# ---------------------------------------------------------------------------

def _load_ids(ids_txt: str) -> List[str]:
    return [l.strip() for l in Path(ids_txt).read_text().splitlines() if l.strip()]


def _load_stats(parquet_path: str) -> Tuple[List[str], List[int], Dict[str, np.ndarray]]:
    """Returns (all_header, feat_col_indices, {flow_id: float32 array})."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    header = list(df.columns)
    feat_idx = [
        i for i, col in enumerate(header)
        if col != "flow_id" and col not in _STATS_META_COLS
    ]
    feat_cols = [header[i] for i in feat_idx]
    arr = df.iloc[:, feat_idx].to_numpy(dtype=np.float32, na_value=0.0)
    id2stat: Dict[str, np.ndarray] = {
        str(df["flow_id"].iloc[i]): arr[i] for i in range(len(df))
    }
    return header, feat_idx, feat_cols, id2stat


def _load_seq(parquet_path: str) -> Dict[str, np.ndarray]:
    """Returns {flow_id: float32 array of shape (N_pkts, 7)}."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    id2seq: Dict[str, np.ndarray] = {}
    for fid, seq in zip(df["flow_id"], df["seq"]):
        arr = np.stack(list(seq)).astype(np.float32)  # (N_pkts, 7)
        id2seq[str(fid)] = arr
    return id2seq


def _load_bytes(parquet_path: str) -> Dict[str, np.ndarray]:
    """Returns {flow_id: int64 array of token ids}."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    id2bytes: Dict[str, np.ndarray] = {}
    for fid, toks in zip(df["flow_id"], df["tokens"]):
        id2bytes[str(fid)] = np.asarray(toks, dtype=np.int64)
    return id2bytes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """Multi-modal traffic flow dataset.

    Parameters
    ----------
    ids_txt   : path to aligned ids file (data/{name}/train_ids.txt etc.)
    data_dir  : path to data/{name}/ (self-contained directory)
    use_stat  : include statistics modality
    use_seq   : include sequence modality
    use_bytes : include bytes modality
    max_seq_len  : truncate / pad seq to this many packets
    bytes_len    : truncate / pad bytes to this many tokens (used when bytes_per_packet=False)
    bytes_per_packet : if True, split tokens by SEP into num_packets segments, output [B, num_packets, pkt_len]
    bytes_pkt_len    : tokens per packet when bytes_per_packet=True (e.g. 63 for 64-byte payload)
    num_packets      : number of packet segments (default 5)
    stat_scaler  : fitted FeatureScaler for stats (None = no scaling)
    seq_scaler   : fitted FeatureScaler for seq continuous features (None = no scaling)
    """

    def __init__(
        self,
        ids_txt: str,
        data_dir: str,
        use_stat:  bool = True,
        use_seq:   bool = True,
        use_bytes: bool = True,
        max_seq_len: int = 100,
        bytes_len:   int = 320,
        bytes_per_packet: bool = False,
        bytes_pkt_len:    int = 63,
        num_packets:     int = 5,
        stat_scaler: Optional[FeatureScaler] = None,
        seq_scaler:  Optional[FeatureScaler] = None,
    ):
        self.max_seq_len = max_seq_len
        self.bytes_len   = bytes_len
        self.bytes_per_packet = bytes_per_packet
        self.bytes_pkt_len    = bytes_pkt_len
        self.num_packets      = num_packets
        self.stat_scaler = stat_scaler
        self.seq_scaler  = seq_scaler

        data_dir = Path(data_dir)
        ids = _load_ids(ids_txt)

        # ----- load label map -----
        label2id_path = data_dir / "label2id.json"
        with label2id_path.open("r", encoding="utf-8") as f:
            self.label2id: Dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}

        # ----- load modalities -----
        self.id2stat:  Optional[Dict[str, np.ndarray]] = None
        self.id2seq:   Optional[Dict[str, np.ndarray]] = None
        self.id2bytes: Optional[Dict[str, np.ndarray]] = None
        self.stat_feat_cols: Optional[List[str]] = None
        self.stat_dim: Optional[int] = None

        if use_stat:
            stats_path = data_dir / "stats" / "stats.parquet"
            _, _, feat_cols, self.id2stat = _load_stats(str(stats_path))
            self.stat_feat_cols = feat_cols
            self.stat_dim = len(feat_cols)
            print(f"  [stats] loaded {len(self.id2stat)} flows, {self.stat_dim} features")

        if use_seq:
            seq_path = data_dir / "seq" / "seq.parquet"
            self.id2seq = _load_seq(str(seq_path))
            print(f"  [seq]   loaded {len(self.id2seq)} flows")

        if use_bytes:
            bytes_path = data_dir / "bytes" / "bytes.parquet"
            self.id2bytes = _load_bytes(str(bytes_path))
            print(f"  [bytes] loaded {len(self.id2bytes)} flows")

        # ----- filter ids to available intersection -----
        valid = list(ids)
        if self.id2stat  is not None: valid = [f for f in valid if f in self.id2stat]
        if self.id2seq   is not None: valid = [f for f in valid if f in self.id2seq]
        if self.id2bytes is not None: valid = [f for f in valid if f in self.id2bytes]
        if not valid:
            raise RuntimeError("No overlapping flow_ids between ids list and loaded modalities.")
        self.flow_ids = valid
        print(f"  [dataset] {len(self.flow_ids)} flows after intersection")

    def __len__(self) -> int:
        return len(self.flow_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fid = self.flow_ids[idx]
        label_str = fid.split("/", 1)[0]
        y = self.label2id[label_str]
        sample: Dict[str, torch.Tensor] = {
            "label":   torch.tensor(y, dtype=torch.long),
            "flow_id": fid,  # type: ignore[assignment]  (str, not tensor)
        }

        # ----- stats -----
        if self.id2stat is not None:
            x = self.id2stat[fid].copy()
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if self.stat_scaler is not None:
                x = self.stat_scaler.transform(x)
            sample["x_stat"] = torch.from_numpy(x)

        # ----- seq -----
        if self.id2seq is not None:
            arr = self.id2seq[fid]           # (N_pkts, 7), float32
            N = arr.shape[0]
            T = self.max_seq_len

            if self.seq_scaler is not None:
                cont = arr[:, SEQ_CONT_IDX].copy()
                arr = arr.copy()
                arr[:, SEQ_CONT_IDX] = self.seq_scaler.transform(cont)

            if N >= T:
                x_seq   = arr[:T]
                attn    = np.ones(T, dtype=np.float32)
            else:
                pad     = np.zeros((T - N, 7), dtype=np.float32)
                x_seq   = np.concatenate([arr, pad], axis=0)
                attn    = np.array([1.0] * N + [0.0] * (T - N), dtype=np.float32)

            sample["x_seq"]    = torch.from_numpy(x_seq)   # [T, 7]
            sample["attn_mask"] = torch.from_numpy(attn)    # [T]

        # ----- bytes -----
        if self.id2bytes is not None:
            toks = self.id2bytes[fid]
            if self.bytes_per_packet:
                # Split by SEP into segments; if no SEP, split evenly (fallback for old data)
                segs: List[np.ndarray] = []
                if TOKEN_SEP in toks:
                    start = 0
                    for i in range(len(toks)):
                        if toks[i] == TOKEN_SEP:
                            segs.append(np.asarray(toks[start:i], dtype=np.int64))
                            start = i + 1
                    if start < len(toks):
                        segs.append(np.asarray(toks[start:], dtype=np.int64))
                    # build_bytes inserts SEP before each packet → first segment is empty
                    segs = [s for s in segs if len(s) > 0][: self.num_packets]
                else:
                    # No SEP: split evenly into num_packets chunks
                    Lp = self.bytes_pkt_len
                    N = self.num_packets
                    chunk_size = max(1, (len(toks) + N - 1) // N)
                    for k in range(N):
                        s, e = k * chunk_size, min((k + 1) * chunk_size, len(toks))
                        segs.append(np.asarray(toks[s:e], dtype=np.int64))
                # Pad to num_packets segments
                Lp = self.bytes_pkt_len
                N = self.num_packets
                rows = []
                masks = []
                for k in range(N):
                    if k < len(segs):
                        seg = segs[k]
                        if len(seg) >= Lp:
                            row = seg[:Lp]
                            msk = np.ones(Lp, dtype=np.int64)
                        else:
                            row = np.concatenate(
                                [seg, np.full(Lp - len(seg), TOKEN_PAD, dtype=np.int64)]
                            )
                            msk = np.array([1] * len(seg) + [0] * (Lp - len(seg)), dtype=np.int64)
                    else:
                        row = np.full(Lp, TOKEN_PAD, dtype=np.int64)
                        msk = np.zeros(Lp, dtype=np.int64)
                    rows.append(row)
                    masks.append(msk)
                x_bytes_t = torch.tensor(np.stack(rows), dtype=torch.long)       # copy to avoid non-writable
                bytes_mask_t = torch.tensor(np.stack(masks), dtype=torch.long)
                sample["x_bytes"]    = x_bytes_t
                sample["bytes_mask"] = bytes_mask_t
            else:
                L = self.bytes_len
                if len(toks) >= L:
                    x_bytes = toks[:L]
                else:
                    x_bytes = np.concatenate(
                        [toks, np.full(L - len(toks), TOKEN_PAD, dtype=np.int64)]
                    )
                x_bytes_t = torch.tensor(x_bytes, dtype=torch.long)       # copy to avoid non-writable warning
                sample["x_bytes"]    = x_bytes_t                          # [L]
                sample["bytes_mask"] = (x_bytes_t != TOKEN_PAD).long()    # [L]

        return sample


# ---------------------------------------------------------------------------
# collate_fn  (for DataLoader)
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack tensors; pass flow_ids as a list of strings."""
    out: Dict = {"flow_id": [s["flow_id"] for s in batch]}
    keys = [k for k in batch[0].keys() if k != "flow_id"]
    for k in keys:
        out[k] = torch.stack([s[k] for s in batch])
    return out


# ---------------------------------------------------------------------------
# fit_scalers  (fit on train set only)
# ---------------------------------------------------------------------------

# Saved next to best.pt so offline eval uses the same mean/std as training.
FEATURE_SCALERS_FILENAME = "feature_scalers.pkl"


def save_feature_scalers(
    path: str,
    stat_s: Optional[FeatureScaler],
    seq_s: Optional[FeatureScaler],
) -> None:
    """Persist fitted stat/seq FeatureScalers (pickle)."""
    with open(path, "wb") as f:
        pickle.dump({"stat": stat_s, "seq": seq_s}, f)


def load_feature_scalers(path: str) -> Tuple[Optional[FeatureScaler], Optional[FeatureScaler]]:
    """Load scalers saved by save_feature_scalers."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["stat"], d["seq"]


def fit_or_load_scalers(
    data_dir: str,
    train_ids_txt: str,
    fit_stat: bool = True,
    fit_seq: bool = True,
    ckpt_dir: Optional[str] = None,
) -> Tuple[Optional[FeatureScaler], Optional[FeatureScaler]]:
    """Fit on train_ids (same as training), or load from ckpt_dir if feature_scalers.pkl exists.

    Loading avoids drift when train_ids.txt / parquet change after the run: offline eval
    still matches the normalization used for that checkpoint.
    """
    if ckpt_dir:
        p = Path(ckpt_dir) / FEATURE_SCALERS_FILENAME
        if p.is_file():
            stat_s, seq_s = load_feature_scalers(str(p))
            print(f"  [scalers] loaded from {p}", flush=True)
            if not fit_stat:
                stat_s = None
            if not fit_seq:
                seq_s = None
            return stat_s, seq_s
    return fit_scalers(
        data_dir, train_ids_txt, fit_stat=fit_stat, fit_seq=fit_seq,
    )


def fit_scalers(
    data_dir: str,
    train_ids_txt: str,
    fit_stat: bool = True,
    fit_seq:  bool = True,
) -> Tuple[Optional[FeatureScaler], Optional[FeatureScaler]]:
    """Fit FeatureScalers from training data only.

    Returns (stat_scaler, seq_scaler); either may be None if not requested.
    """
    data_dir = Path(data_dir)
    ids = _load_ids(train_ids_txt)
    ids_set = set(ids)

    stat_scaler: Optional[FeatureScaler] = None
    seq_scaler:  Optional[FeatureScaler] = None
    id2stat = None
    id2seq  = None

    if fit_stat:
        _, _, feat_cols, id2stat = _load_stats(str(data_dir / "stats" / "stats.parquet"))
        stat_scaler = FeatureScaler(num_features=len(feat_cols))

    if fit_seq:
        id2seq = _load_seq(str(data_dir / "seq" / "seq.parquet"))
        seq_scaler = FeatureScaler(num_features=len(SEQ_CONT_IDX))

    for fid in ids:
        if id2stat is not None and fid in id2stat:
            x = id2stat[fid].copy()
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            stat_scaler.partial_fit(x)  # type: ignore[union-attr]
        if id2seq is not None and fid in id2seq:
            arr = id2seq[fid]
            seq_scaler.partial_fit(arr[:, SEQ_CONT_IDX])  # type: ignore[union-attr]

    if stat_scaler is not None:
        stat_scaler.finalize()
        print(f"  [stat_scaler] fitted on {stat_scaler.n} samples, {len(feat_cols)} features")
    if seq_scaler is not None:
        seq_scaler.finalize()
        print(f"  [seq_scaler]  fitted on {seq_scaler.n} samples")

    return stat_scaler, seq_scaler
