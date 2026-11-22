#!/usr/bin/env python3
"""
build_seq.py - Extract per-packet sequence features for all packets in each flow.

Output: seq/seq.parquet (single Parquet, columns: flow_id, seq)
        seq/samples/<label>/sample_*.csv (optional, for inspection)
  seq = list of [pktlen, iat, direction, tcp_flags, payload_len, header_len, tcp_window] per packet

Usage:
  python build_seq.py \
    --flows_dir  ./dataset/USTC-TFC2016/flows_filtered \
    --out_dir    ./data/USTC-TFC2016/seq \
    [--workers   8] [--max_packets 100] [--resume]
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

import numpy as np
from scapy.utils import RawPcapReader          # type: ignore
from scapy.layers.inet import IP, TCP, UDP     # type: ignore


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_ids(ids_txt: Path) -> List[str]:
    return [l.strip() for l in ids_txt.read_text().splitlines() if l.strip()]


def _tcp_flags(tcp) -> int:
    if tcp is None:
        return 0
    m = 0
    try:
        f = tcp.flags
        for bit in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20):
            if f & bit:
                m |= bit
    except Exception:
        pass
    return m


def _hdr_win(pkt) -> Tuple[int, int]:
    hdr = win = 0
    try:
        if IP in pkt:
            hdr += int(pkt[IP].ihl) * 4 if pkt[IP].ihl else 20
        if TCP in pkt:
            hdr += int(pkt[TCP].dataofs) * 4 if pkt[TCP].dataofs else 20
            win = int(pkt[TCP].window)
        elif UDP in pkt:
            hdr += 8
    except Exception:
        pass
    return hdr, win


def extract_seq(pcap_path: Path, max_packets: int = 0) -> Optional[np.ndarray]:
    """Return float32 array of shape [N_packets, 7], or None on error. max_packets=0 means all."""
    rows: List[List[float]] = []
    first_src = None
    last_ts: Optional[float] = None
    try:
        rp = RawPcapReader(str(pcap_path))
        for pkt_bytes, meta in rp:
            try:
                pkt = IP(pkt_bytes)
            except Exception:
                try:
                    from scapy.all import Ether
                    pkt = Ether(pkt_bytes)
                except Exception:
                    continue
            if IP not in pkt and TCP not in pkt and UDP not in pkt:
                continue
            if IP in pkt:
                if first_src is None:
                    first_src = pkt[IP].src
                direction = 1 if pkt[IP].src == first_src else -1
                try:
                    ts = float(meta.sec) + float(meta.usec) / 1e6
                except Exception:
                    ts = float(getattr(pkt, "time", 0.0))
            else:
                direction = 1
                ts = 0.0
            pktlen = len(pkt_bytes)
            iat = 0.0 if last_ts is None else max(0.0, ts - last_ts)
            last_ts = ts
            flags = _tcp_flags(pkt[TCP] if TCP in pkt else None)
            hdr, win = _hdr_win(pkt)
            payload = max(0, pktlen - hdr)
            rows.append([float(pktlen), float(iat), float(direction),
                         float(flags), float(payload), float(hdr), float(win)])
            if max_packets > 0 and len(rows) >= max_packets:
                break
        try:
            rp.close()
        except Exception:
            pass
    except Exception:
        return None
    if not rows:
        return None
    return np.array(rows, dtype=np.float32)


def _worker(args_tuple) -> Tuple[str, Optional[np.ndarray], str]:
    """Worker: extract seq for one flow, return (fid, arr, err)."""
    fid, pcap_str, max_packets = args_tuple
    pcap = Path(pcap_str)
    try:
        arr = extract_seq(pcap, max_packets=max_packets)
        return fid, arr, ""
    except Exception as e:
        return fid, None, str(e)


SEQ_COLUMNS = ["pktlen", "iat", "direction", "tcp_flags", "payload_len", "header_len", "tcp_window"]


def _write_sample_csvs(parquet_path: Path, samples_dir: Path, per_class: int) -> None:
    """Write sample flows as CSV for inspection (from Parquet)."""
    import csv
    import pandas as pd
    if not parquet_path.exists() or per_class <= 0:
        return
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "format.txt").write_text(
        "Columns: " + ", ".join(SEQ_COLUMNS) + "\n"
        "  pktlen: packet length (bytes)\n"
        "  iat: inter-arrival time (s)\n"
        "  direction: 1=forward, -1=reverse\n"
        "  tcp_flags: TCP flags bitmask\n"
        "  payload_len: payload bytes\n"
        "  header_len: IP+TCP/UDP header bytes\n"
        "  tcp_window: TCP window size\n",
        encoding="utf-8",
    )
    df = pd.read_parquet(parquet_path)
    if df.empty or "seq" not in df.columns:
        return
    labels = df["flow_id"].str.split("/", n=1, expand=True)[0].unique()
    for label in sorted(labels):
        sub = df[df["flow_id"].str.startswith(str(label) + "/")].head(per_class)
        if sub.empty:
            continue
        out_label = samples_dir / str(label)
        out_label.mkdir(parents=True, exist_ok=True)
        for i, (_, row) in enumerate(sub.iterrows()):
            seq = row["seq"]
            csv_path = out_label / f"sample_{i}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(SEQ_COLUMNS)
                w.writerows(seq)
    print(f"  Samples -> {samples_dir}  ({per_class} per class)", flush=True)


def build_seq(
    flows_dir: Path,
    ids: Optional[List[str]],
    out_dir: Path,
    workers: int = 8,
    max_packets: int = 0,
    sample_per_class: int = 0,
    resume: bool = False,
) -> None:
    import shutil
    if not flows_dir.is_dir():
        raise NotADirectoryError(flows_dir)
    source = flows_dir.resolve()
    destination = out_dir.resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Capture and feature directories must not overlap")
    if not resume:
        if out_dir.exists():
            shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    parquet_path = out_dir / "seq.parquet"

    if ids is None:
        print("  Scanning flows_dir...", flush=True)
        ids = [
            f"{d.name}/{p.stem}"
            for d in sorted(flows_dir.iterdir()) if d.is_dir()
            for p in sorted(d.glob("*.pcap"))
        ]

    existing_ids: set = set()
    if resume and parquet_path.exists():
        import pandas as pd
        try:
            df = pd.read_parquet(parquet_path, columns=["flow_id"])
            existing_ids.update(df["flow_id"].astype(str))
        except Exception:
            pass

    tasks = []
    for fid in ids:
        parts = fid.split("/", 1)
        if len(parts) != 2:
            continue
        label, stem = parts
        pcap = flows_dir / label / f"{stem}.pcap"
        if not pcap.exists():
            continue
        if resume and fid in existing_ids:
            continue
        tasks.append((fid, str(pcap), max_packets))

    total = len(tasks)
    print(f"[build_seq] {total} flows -> {out_dir}  (max_packets={max_packets or 'all'}, resume={resume})")
    ok = skip = 0

    collected: List[Tuple[str, np.ndarray]] = []

    if _HAS_JOBLIB and workers > 1:
        print(f"  Starting {workers} workers (joblib loky)...", flush=True)
        if tqdm:
            results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
                delayed(_worker)(t) for t in tqdm(tasks, unit="flow", desc="seq")
            )
        else:
            results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
                delayed(_worker)(t) for t in tasks
            )
        for i, (fid, arr, err) in enumerate(results, 1):
            if arr is not None:
                collected.append((fid, arr))
                ok += 1
            else:
                skip += 1
                if skip <= 10 and err:
                    print(f"  [WARN] flow failed: {err}", flush=True)
            if not tqdm and (i % 500 == 0 or i == total):
                print(f"  [{i}/{total}]  ok={ok}  skip={skip}", flush=True)
    else:
        print(f"  Sequential processing (joblib not found or workers=1)...", flush=True)
        it = tqdm(enumerate(tasks, 1), total=total, unit="flow", desc="seq") if tqdm \
             else enumerate(tasks, 1)
        for i, t in it:
            fid, arr, err = _worker(t)
            if arr is not None:
                collected.append((fid, arr))
                ok += 1
            else:
                skip += 1
                if skip <= 10 and err:
                    print(f"  [WARN] flow failed: {err}", flush=True)
            if not tqdm and (i % 500 == 0 or i == total):
                print(f"  [{i}/{total}]  ok={ok}  skip={skip}", flush=True)

    # Write single Parquet (like stats)
    if collected:
        import pandas as pd
        df = pd.DataFrame([{"flow_id": fid, "seq": arr.tolist()} for fid, arr in collected])
        if resume and parquet_path.exists():
            old = pd.read_parquet(parquet_path)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates("flow_id", keep="last")
        df.to_parquet(parquet_path, index=False)

    print(f"[DONE] seq -> {parquet_path}  (ok={ok}, skip={skip})")
    if sample_per_class > 0:
        _write_sample_csvs(parquet_path, out_dir / "samples", sample_per_class)


def main() -> None:
    import json as _json

    ap = argparse.ArgumentParser("Extract per-packet sequence features (all packets, .npy per flow)")
    ap.add_argument("--config", default=None, help="path to config.json (reads [seq] section; CLI overrides)")
    ap.add_argument("--flows_dir", required=True, help="flows_filtered/ directory")
    ap.add_argument("--ids", default=None, help="train_ids.txt, val_ids.txt, or test_ids.txt (omit = all flows)")
    ap.add_argument("--out_dir", required=True, help="Output directory (writes seq.parquet + samples/)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_packets", type=int, default=0,
                    help="Max packets per flow (0=all). Use 50-200 for faster run, Dataset truncates at load.")
    ap.add_argument("--sample_per_class", type=int, default=3,
                    help="Write N sample flows per class as CSV in seq/samples/ for inspection (0=skip)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip flows that already exist in Parquet; do not delete seq/ at start")

    pre, _ = ap.parse_known_args()
    if pre.config is not None:
        with open(pre.config, "r", encoding="utf-8") as _f:
            cfg_all = _json.load(_f)
        cfg = cfg_all.get("seq", cfg_all)
        ap.set_defaults(**{k: v for k, v in cfg.items() if v is not None})

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    flows_dir = Path(args.flows_dir)
    ids = load_ids(Path(args.ids)) if args.ids else None
    build_seq(flows_dir, ids, out_dir, workers=args.workers,
              max_packets=args.max_packets, sample_per_class=args.sample_per_class, resume=args.resume)

    cfg_snapshot = {"workers": args.workers, "max_packets": args.max_packets,
                    "sample_per_class": args.sample_per_class}
    with open(out_dir / "config.json", "w", encoding="utf-8") as _f:
        _json.dump(cfg_snapshot, _f, indent=2)


if __name__ == "__main__":
    main()
