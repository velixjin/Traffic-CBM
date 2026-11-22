#!/usr/bin/env python3
"""
build_bytes.py - Extract byte-level bigram token sequences from the first N packets.

Output: bytes/bytes.parquet (single Parquet, columns: flow_id, tokens)
        bytes/samples/<label>/sample.csv (optional, for inspection)

Token vocabulary:
  bigram tokens : 0x0000 - 0xFFFF  (65536 tokens)
  PAD           : 65536
  SEP           : 65537
  MASK          : 65538
  CLS           : 65539

Usage:
  python build_bytes.py \
    --flows_dir  ./dataset/USTC-TFC2016/flows_filtered \
    --out_dir    ./data/USTC-TFC2016/bytes \
    [--pkt_n 5] [--payload_len 64] [--sample_per_class 3] [--workers 8]
  workers=1: sequential (reliable). workers>1: joblib loky (multi-CPU, needs joblib).
"""
import argparse
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


def _progress_parallel(tasks, n_jobs, total):
    """Run joblib Parallel with tqdm progress bar."""
    if not tqdm or total <= 0:
        return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_worker)(t) for t in tasks
        )
    with tqdm(total=total, unit="flow", desc="bytes") as pbar:
        n_completed = [0]  # use list for closure

        class _CallbackParallel(Parallel):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                return result

            def print_progress(self):
                super().print_progress()
                n_completed[0] = self.n_completed_tasks
                pbar.n = n_completed[0]
                pbar.refresh()

        return _CallbackParallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_worker)(t) for t in tasks
        )

from scapy.utils import rdpcap          # type: ignore
from scapy.layers.inet import IP, TCP, UDP  # type: ignore
from scapy.layers.inet6 import IPv6     # type: ignore

TOKEN_PAD  = 65536
TOKEN_SEP  = 65537
TOKEN_MASK = 65538
TOKEN_CLS  = 65539

# ---------------------------------------------------------------------------
# SII sanitization (Strong Identification Information)
# Supports mode="zero" (all zeros) or mode="random" (random bytes).
# Operates on raw Ethernet+IPv4/IPv6+TCP/UDP bytes, no scapy re-parsing.
# Sanitizes: IP ID, IP checksum, IP src/dst, TCP/UDP sport/dport,
#             TCP seq/ack, TCP checksum, TCP Timestamp option (optional).
# ---------------------------------------------------------------------------
import random as _rnd


def _sanitize_tcp_opts(
    b: bytearray,
    start: int,
    end: int,
    mode: str,
    preserve_tcp_timestamp: bool = False,
) -> None:
    """Zero or randomize TCP Timestamp option (kind=8) bytes in-place."""
    i = start
    while i < min(end, len(b)):
        kind = b[i]
        if kind == 0:       # End of Option List
            break
        if kind == 1:       # NOP
            i += 1
            continue
        if i + 1 >= end:
            break
        length = b[i + 1]
        if length < 2 or i + length > end:
            break
        if kind == 8 and length == 10 and not preserve_tcp_timestamp:
            for j in range(i + 2, i + 10):
                b[j] = _rnd.getrandbits(8) if mode == "random" else 0
        i += length


def _sanitize_raw(raw: bytes, mode: str, preserve_tcp_timestamp: bool = False) -> bytes:
    """Return a copy of raw packet bytes with SII fields zeroed/randomized.

    Handles Ethernet + IPv4/IPv6 + TCP/UDP.  TLS gmt_unix_time is not
    modified here because it sits at a variable offset deep in the payload
    and requires TLS layer parsing; use start_offset=38 to push it out of
    the extracted window in most flows.
    """
    if mode == "none":
        return raw
    b = bytearray(raw)
    n = len(b)

    def patch(s: int, e: int) -> None:
        for i in range(s, min(e, n)):
            b[i] = _rnd.getrandbits(8) if mode == "random" else 0

    if n < 14:
        return raw
    ethertype = (b[12] << 8) | b[13]

    if ethertype == 0x0800:         # IPv4
        ip = 14
        if n < ip + 20:
            return raw
        ihl = (b[ip] & 0x0F) * 4
        proto = b[ip + 9]
        patch(ip + 4,  ip + 6)     # IP Identification
        patch(ip + 10, ip + 12)    # IP header checksum
        patch(ip + 12, ip + 20)    # IP src + dst
        l4 = ip + ihl
        if proto == 6 and n >= l4 + 20:     # TCP
            patch(l4,      l4 + 4)          # sport, dport
            patch(l4 + 4,  l4 + 12)         # seq, ack
            patch(l4 + 16, l4 + 18)         # TCP checksum
            doff = (b[l4 + 12] >> 4) * 4
            _sanitize_tcp_opts(b, l4 + 20, l4 + doff, mode, preserve_tcp_timestamp)
        elif proto == 17 and n >= l4 + 8:   # UDP
            patch(l4,      l4 + 4)          # sport, dport
            patch(l4 + 6,  l4 + 8)          # UDP checksum

    elif ethertype == 0x86DD:       # IPv6
        ip = 14
        if n < ip + 40:
            return raw
        next_hdr = b[ip + 6]
        patch(ip + 8, ip + 40)     # IPv6 src + dst
        l4 = ip + 40
        if next_hdr == 6 and n >= l4 + 20:  # TCP
            patch(l4,      l4 + 4)
            patch(l4 + 4,  l4 + 12)
            patch(l4 + 16, l4 + 18)
            doff = (b[l4 + 12] >> 4) * 4
            _sanitize_tcp_opts(b, l4 + 20, l4 + doff, mode, preserve_tcp_timestamp)
        elif next_hdr == 17 and n >= l4 + 8:  # UDP
            patch(l4,      l4 + 4)
            patch(l4 + 6,  l4 + 8)

    return bytes(b)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_ids(ids_txt: Path) -> List[str]:
    return [l.strip() for l in ids_txt.read_text().splitlines() if l.strip()]


def _bigrams(data: bytes) -> List[int]:
    """Convert a byte sequence to bigram token IDs."""
    return [data[i] * 256 + data[i + 1] for i in range(len(data) - 1)]


def _first_src(packets) -> Optional[str]:
    for pkt in packets:
        try:
            if IP in pkt:
                return pkt[IP].src
            if IPv6 in pkt:
                return pkt[IPv6].src
        except Exception:
            pass
    return None


def packets_to_tokens(
    packets,
    pkt_n: int,
    payload_len: int,
    start_offset: int,
    insert_sep: bool,
    sanitize_mode: str = "none",
    preserve_tcp_timestamp: bool = False,
) -> List[int]:
    """Extract bigram tokens from the first pkt_n packets.

    sanitize_mode: "none" | "zero" | "random"
      Sanitizes SII fields (IP src/dst, ports, TCP seq/ack, checksums).
      TCP Timestamp option can be preserved via preserve_tcp_timestamp=True.
    """
    if not packets:
        return []
    src0 = _first_src(packets)
    tokens: List[int] = []
    count = 0
    for pkt in packets:
        has_l3 = (IP in pkt) or (IPv6 in pkt)
        if not has_l3:
            continue
        try:
            raw = bytes(pkt)
        except Exception:
            continue
        # Skip Ethernet-less packets: prepend fake Ethernet header to align offset
        if not hasattr(pkt, "type"):
            try:
                if IP in pkt:
                    src = pkt[IP].src
                else:
                    src = pkt[IPv6].src
                if src == src0:
                    raw = bytes.fromhex("c49a025996f8e46f13e2e3ae0800") + raw
                else:
                    raw = bytes.fromhex("e46f13e2e3aec49a025996f80800") + raw
            except Exception:
                pass
        if sanitize_mode != "none":
            raw = _sanitize_raw(raw, sanitize_mode, preserve_tcp_timestamp=preserve_tcp_timestamp)
        chunk = raw[start_offset: start_offset + payload_len]
        if len(chunk) < 2:
            continue
        if insert_sep:
            tokens.append(TOKEN_SEP)
        tokens.extend(_bigrams(chunk))
        count += 1
        if count >= pkt_n:
            break
    return tokens


def _worker(args_tuple) -> Tuple[str, Optional[List[int]], str]:
    fid, pcap_str, pkt_n, payload_len, start_offset, insert_sep, sanitize_mode, preserve_tcp_timestamp = args_tuple
    try:
        packets = rdpcap(pcap_str)
        # Skip pure-IPv6 flows
        if packets and (IPv6 in packets[0] and IP not in packets[0]):
            return fid, None, "ipv6"
        toks = packets_to_tokens(
            packets,
            pkt_n,
            payload_len,
            start_offset,
            insert_sep,
            sanitize_mode,
            preserve_tcp_timestamp=preserve_tcp_timestamp,
        )
        if not toks:
            return fid, None, "empty"
        return fid, toks, ""
    except Exception as e:
        return fid, None, str(e)


def _token_to_hex(t: int) -> str:
    """Format token as hex: 0-65535 -> 4 digits, 65536+ -> 5 digits (no 0x prefix)."""
    if t <= 0xFFFF:
        return f"{t:04X}"
    return f"{t:05X}"


def _write_samples(parquet_path: Path, samples_dir: Path, per_class: int) -> None:
    """Write sample rows per class: one sample per line, tokens in hex."""
    if not parquet_path.exists() or per_class <= 0:
        return
    import pandas as pd
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "format.txt").write_text(
        "One sample per line: flow_id\\ttoken1 token2 ... (hex, no 0x prefix)\n"
        "  bigram: 0000-FFFF; PAD=10000 SEP=10001 MASK=10002 CLS=10003\n",
        encoding="utf-8",
    )
    df = pd.read_parquet(parquet_path)
    if df.empty or "flow_id" not in df.columns or "tokens" not in df.columns:
        return
    labels = df["flow_id"].str.split("/", n=1, expand=True)[0].unique()
    for label in sorted(labels):
        sub = df[df["flow_id"].str.startswith(str(label) + "/")].head(per_class)
        if sub.empty:
            continue
        out_label = samples_dir / str(label)
        out_label.mkdir(parents=True, exist_ok=True)
        with (out_label / "sample.txt").open("w", encoding="utf-8") as f:
            for _, row in sub.iterrows():
                fid = row["flow_id"]
                toks = row["tokens"]
                hex_str = " ".join(_token_to_hex(int(t)) for t in toks)
                f.write(f"{fid}\t{hex_str}\n")
    print(f"  Samples -> {samples_dir}  ({per_class} per class, hex)", flush=True)


def build_bytes(
    flows_dir: Path,
    ids: Optional[List[str]],
    out_dir: Path,
    pkt_n: int = 5,
    payload_len: int = 64,
    start_offset: int = 14,
    insert_sep: bool = False,
    sample_per_class: int = 0,
    workers: int = 8,
    sanitize_mode: str = "none",
    preserve_tcp_timestamp: bool = False,
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
    ensure_dir(out_dir)
    parquet_path = out_dir / "bytes.parquet"

    if ids is None:
        print("  Scanning flows_dir...", flush=True)
        ids = [
            f"{d.name}/{p.stem}"
            for d in sorted(flows_dir.iterdir()) if d.is_dir()
            for p in sorted(d.glob("*.pcap"))
        ]

    tasks = []
    for fid in ids:
        parts = fid.split("/", 1)
        if len(parts) != 2:
            continue
        label, stem = parts
        pcap = flows_dir / label / f"{stem}.pcap"
        if not pcap.exists():
            continue
        tasks.append((
            fid, str(pcap), pkt_n, payload_len, start_offset, insert_sep,
            sanitize_mode, preserve_tcp_timestamp,
        ))

    smode_str = f", sanitize={sanitize_mode}" if sanitize_mode != "none" else ""
    ts_str = ", keep_tcp_ts" if preserve_tcp_timestamp else ""
    print(f"[build_bytes] {len(tasks)} flows -> {parquet_path}{smode_str}{ts_str}")
    total = len(tasks)
    collected: List[Tuple[str, List[int]]] = []

    if workers <= 1:
        # Sequential (like FlowCAT-v2): reliable, no fork/thread issues
        print(f"  Sequential processing...", flush=True)
        it = (_worker(t) for t in tasks)
        if tqdm:
            it = tqdm(it, total=total, unit="flow", desc="bytes")
        for fid, toks, _ in it:
            if toks is not None:
                collected.append((fid, toks))
    elif _HAS_JOBLIB:
        # joblib loky: robust multiprocessing, avoids fork issues
        print(f"  Starting {workers} workers (joblib loky)...", flush=True)
        results = _progress_parallel(tasks, workers, total)
        collected = [(fid, toks) for fid, toks, _ in results if toks is not None]
    else:
        # Fallback: sequential
        print(f"  joblib not found, sequential processing...", flush=True)
        for t in tasks:
            fid, toks, _ = _worker(t)
            if toks is not None:
                collected.append((fid, toks))

    ok = len(collected)
    skip = total - ok

    if collected:
        import pandas as pd
        df = pd.DataFrame([{"flow_id": fid, "tokens": toks} for fid, toks in collected])
        df.to_parquet(parquet_path, index=False)

    print(f"[DONE] bytes -> {parquet_path}  (ok={ok}, skip={skip})")
    if sample_per_class > 0:
        _write_samples(parquet_path, out_dir / "samples", sample_per_class)


def main() -> None:
    import json as _json
    import shutil as _shutil

    ap = argparse.ArgumentParser("Extract byte bigram token sequences (first N packets, no augmentation)")
    ap.add_argument("--config", default=None, help="path to config.json (reads [bytes] section; CLI overrides)")
    ap.add_argument("--flows_dir", required=True, help="flows_filtered/ directory")
    ap.add_argument("--ids", default=None, help="train_ids.txt, val_ids.txt, or test_ids.txt (omit = all flows)")
    ap.add_argument("--out_dir", required=True, help="Output directory (writes bytes.parquet + samples/)")
    ap.add_argument("--pkt_n", type=int, default=5, help="Packets per flow (default: 5)")
    ap.add_argument("--payload_len", type=int, default=64, help="Bytes per packet (default: 64)")
    ap.add_argument("--start_offset", type=int, default=38,
                    help="Byte offset from the packet start (default: 38)")
    ap.add_argument("--sep", action="store_true", help="Insert SEP token between packets")
    ap.add_argument("--sanitize_mode", default="none",
                    choices=["none", "zero", "random"],
                    help="SII sanitization: none (default) | zero (clear fields) | random (like FlowCAT)")
    ap.add_argument("--preserve_tcp_timestamp", action="store_true",
                    help="keep TCP Timestamp option values while still sanitizing other fields")
    ap.add_argument("--sample_per_class", type=int, default=3,
                    help="Write N sample flows per class as CSV in bytes/samples/ (0=skip)")
    ap.add_argument("--workers", type=int, default=8)

    # load config defaults before final parse
    pre, _ = ap.parse_known_args()
    if pre.config is not None:
        with open(pre.config, "r", encoding="utf-8") as _f:
            cfg_all = _json.load(_f)
        cfg = cfg_all.get("bytes", cfg_all)
        ap.set_defaults(**{k: v for k, v in cfg.items() if v is not None})

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    flows_dir = Path(args.flows_dir)
    ids = load_ids(Path(args.ids)) if args.ids else None

    build_bytes(
        flows_dir=flows_dir,
        ids=ids,
        out_dir=out_dir,
        pkt_n=args.pkt_n,
        payload_len=args.payload_len,
        start_offset=args.start_offset,
        insert_sep=args.sep,
        sanitize_mode=args.sanitize_mode,
        preserve_tcp_timestamp=args.preserve_tcp_timestamp,
        sample_per_class=args.sample_per_class,
        workers=args.workers,
    )

    cfg_snapshot = {
        "pkt_n": args.pkt_n,
        "payload_len": args.payload_len,
        "start_offset": args.start_offset,
        "sep": args.sep,
        "sanitize_mode": args.sanitize_mode,
        "preserve_tcp_timestamp": args.preserve_tcp_timestamp,
        "sample_per_class": args.sample_per_class,
        "workers": args.workers,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as _f:
        _json.dump(cfg_snapshot, _f, indent=2)


if __name__ == "__main__":
    main()
