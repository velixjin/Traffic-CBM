#!/usr/bin/env python3
"""
build_stats.py - Extract CICFlowMeter statistical features for all packets in each flow.

Output: stats/stats.parquet + stats/samples/<label>/sample_*.csv

Usage:
  python build_stats.py \
    --flows_dir  ./dataset/USTC-TFC2016/flows_filtered \
    --out        ./data/USTC-TFC2016/stats \
    [--workers   8] [--sample_per_class 3]
"""
import os
import sys
import csv
import argparse
import tempfile
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager, redirect_stdout, redirect_stderr

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# cicflowmeter vendor
VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "cicflowmeter"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _suppress_all():
    devnull = open(os.devnull, "w")
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        devnull.close()


def _process_one(args_tuple):
    """Worker: extract CICFlowMeter features for one flow. Returns (fid, header, row) or (fid, None, None)."""
    import os, sys, csv, tempfile
    from contextlib import contextmanager, redirect_stdout, redirect_stderr

    fid, pcap_str, vendor_str, n_packets = args_tuple
    vendor_dir = Path(vendor_str)
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))

    from firstn import save_firstn_to_csv  # type: ignore

    @contextmanager
    def _suppress():
        devnull = open(os.devnull, "w")
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            devnull.close()

    pcap = Path(pcap_str)
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_csv = Path(td) / f"{pcap.stem}.csv"
            with _suppress():
                save_firstn_to_csv(str(pcap), str(tmp_csv), n_packets=n_packets, fields=None, verbose=False)
            if not tmp_csv.exists() or tmp_csv.stat().st_size == 0:
                return fid, None, None
            with tmp_csv.open("r", encoding="utf-8") as f:
                rr = csv.reader(f)
                header = next(rr, None)
                row = next(rr, None)
            if header is None or row is None:
                return fid, None, None
            return fid, header, row
    except Exception:
        return fid, None, None


def load_ids(ids_txt: Path) -> List[str]:
    return [l.strip() for l in ids_txt.read_text().splitlines() if l.strip()]


def _write_samples(stats_dir: Path, df, sample_per_class: int) -> None:
    """Write sample rows per class as CSV for inspection."""
    if sample_per_class <= 0 or df is None or df.empty or "flow_id" not in df.columns:
        return
    samples_dir = stats_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    labels = df["flow_id"].str.split("/", n=1, expand=True)[0].unique()
    for label in sorted(labels):
        sub = df[df["flow_id"].str.startswith(str(label) + "/")].head(sample_per_class)
        if sub.empty:
            continue
        out_label = samples_dir / str(label)
        out_label.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_label / "sample.csv", index=False, encoding="utf-8")
    print(f"  Samples -> {samples_dir}  ({sample_per_class} per class)", flush=True)


def build_stats(
    flows_dir: Path,
    ids: Optional[List[str]],
    out_dir: Path,
    workers: int = 8,
    max_packets: int = 999_999,
    sample_per_class: int = 3,
) -> None:
    import shutil
    if not flows_dir.is_dir():
        raise NotADirectoryError(flows_dir)
    source = flows_dir.resolve()
    destination = out_dir.resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Capture and feature directories must not overlap")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "stats.parquet"

    if ids is None:
        print("  Scanning flows_dir...", flush=True)
        ids = [
            f"{d.name}/{p.stem}"
            for d in sorted(flows_dir.iterdir()) if d.is_dir()
            for p in sorted(d.glob("*.pcap"))
        ]

    vendor_str = str(VENDOR_DIR)
    tasks = []
    for fid in ids:
        parts = fid.split("/", 1)
        if len(parts) != 2:
            continue
        label, stem = parts
        pcap = flows_dir / label / f"{stem}.pcap"
        if not pcap.exists():
            continue
        tasks.append((fid, str(pcap), vendor_str, max_packets))

    total = len(tasks)
    print(f"[build_stats] {total} flows -> {parquet_path}  (max_packets={max_packets})")

    header = None
    rows: List[list] = []
    ok = skip = 0

    if _HAS_JOBLIB and workers > 1:
        print(f"  Starting {workers} workers (joblib loky)...", flush=True)
        if tqdm:
            results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
                delayed(_process_one)(t) for t in tqdm(tasks, unit="flow", desc="stats")
            )
        else:
            results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
                delayed(_process_one)(t) for t in tasks
            )
        for i, (fid, hdr, row) in enumerate(results, 1):
            if hdr is None:
                skip += 1
            else:
                if header is None:
                    header = ["flow_id"] + hdr
                rows.append([fid] + row)
                ok += 1
            if not tqdm and (i % 500 == 0 or i == total):
                print(f"  [{i}/{total}]  ok={ok}  skip={skip}", flush=True)
    else:
        print(f"  Sequential processing (joblib not found or workers=1)...", flush=True)
        it = tqdm(enumerate(tasks, 1), total=total, unit="flow", desc="stats") if tqdm \
             else enumerate(tasks, 1)
        for i, t in it:
            fid, hdr, row = _process_one(t)
            if hdr is None:
                skip += 1
            else:
                if header is None:
                    header = ["flow_id"] + hdr
                rows.append([fid] + row)
                ok += 1
            if not tqdm and (i % 500 == 0 or i == total):
                print(f"  [{i}/{total}]  ok={ok}  skip={skip}", flush=True)

    if header and rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=header)
        df.to_parquet(parquet_path, index=False)
        _write_samples(out_dir, df, sample_per_class)

    print(f"[DONE] stats -> {parquet_path}  (ok={ok}, skip={skip})")


def main() -> None:
    import json as _json

    ap = argparse.ArgumentParser("Extract CICFlowMeter stats features (all packets)")
    ap.add_argument("--config", default=None, help="path to config.json (reads [stats] section; CLI overrides)")
    ap.add_argument("--flows_dir", required=True, help="flows_filtered/ directory")
    ap.add_argument("--ids", default=None, help="train_ids.txt, val_ids.txt, or test_ids.txt (omit = all flows)")
    ap.add_argument("--out", required=True, help="Output stats dir (writes stats.parquet + samples/)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_packets", type=int, default=999_999,
                    help="Max packets per flow for CICFlowMeter (default: all). Use 100-500 for faster run.")
    ap.add_argument("--sample_per_class", type=int, default=3,
                    help="Write N sample flows per class as CSV in samples/ (0=skip)")

    pre, _ = ap.parse_known_args()
    if pre.config is not None:
        with open(pre.config, "r", encoding="utf-8") as _f:
            cfg_all = _json.load(_f)
        cfg = cfg_all.get("stats", cfg_all)
        ap.set_defaults(**{k: v for k, v in cfg.items() if v is not None})

    args = ap.parse_args()

    out_dir = Path(args.out)
    flows_dir = Path(args.flows_dir)
    ids = load_ids(Path(args.ids)) if args.ids else None
    build_stats(flows_dir, ids, out_dir, workers=args.workers,
                max_packets=args.max_packets, sample_per_class=args.sample_per_class)

    cfg_snapshot = {"workers": args.workers, "max_packets": args.max_packets,
                    "sample_per_class": args.sample_per_class}
    with open(out_dir / "config.json", "w", encoding="utf-8") as _f:
        _json.dump(cfg_snapshot, _f, indent=2)


if __name__ == "__main__":
    main()
