#!/usr/bin/env python3
"""
Batch mask TLS SNI (host_name) in PCAP/PCAPNG datasets.

Dataset layout:
  DATASET_NAME/<label>/*.pcap[.ng]
  (nested files under each label are included via recursive scan)

Default output:
  DATASET_NAME_no_sni/<label>/...  (same relative paths and names)

Per-pcap logic (length-preserving):
  - Only overwrites SNI host_name bytes; TLS length fields unchanged.
  - Reassembles TCP payload per flow direction before parsing ClientHello.
  - Uses scapy for pcap I/O and packet mutation; TLS parsing is inline.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Single-pcap: TLS SNI parse + mask
# ---------------------------------------------------------------------------


def _u16(b: bytes) -> int:
    return (b[0] << 8) | b[1]


def _u24(b: bytes) -> int:
    return (b[0] << 16) | (b[1] << 8) | b[2]


@dataclass
class SNIHit:
    # Offset is relative to the parsed buffer (TCP stream)
    name_off: int
    name_len: int
    name_bytes: bytes


def _parse_server_name_ext(payload: bytes, s: int, e: int) -> List[SNIHit]:
    # RFC 6066 server_name: list_len(2) + entries...
    if s + 2 > e:
        return []
    list_len = _u16(payload[s : s + 2])
    p = s + 2
    list_end = min(e, p + list_len)

    hits: List[SNIHit] = []
    while p + 3 <= list_end:
        name_type = payload[p]
        name_len = _u16(payload[p + 1 : p + 3])
        name_start = p + 3
        name_end = name_start + name_len
        if name_end > list_end:
            break
        if name_type == 0:  # host_name
            name_bytes = payload[name_start:name_end]
            hits.append(SNIHit(name_off=name_start, name_len=name_len, name_bytes=name_bytes))
        p = name_end
    return hits


def _extract_sni_from_clienthello(payload: bytes, hs_start: int, hs_end: int) -> List[SNIHit]:
    """
    Parse ClientHello body within [hs_start, hs_end) in payload bytes.
    Returns SNI hits with offsets relative to `payload`.
    """
    p = hs_start
    if p + 2 + 32 + 1 > hs_end:
        return []
    p += 2  # client_version
    p += 32  # random

    sess_len = payload[p]
    p += 1 + sess_len
    if p + 2 > hs_end:
        return []
    cs_len = _u16(payload[p : p + 2])
    p += 2 + cs_len
    if p + 1 > hs_end:
        return []
    comp_len = payload[p]
    p += 1 + comp_len

    if p + 2 > hs_end:
        return []
    ext_total_len = _u16(payload[p : p + 2])
    p += 2
    ext_end = min(hs_end, p + ext_total_len)

    hits: List[SNIHit] = []
    while p + 4 <= ext_end:
        ext_type = _u16(payload[p : p + 2])
        ext_len = _u16(payload[p + 2 : p + 4])
        ext_data_start = p + 4
        ext_data_end = ext_data_start + ext_len
        if ext_data_end > ext_end:
            break
        if ext_type == 0x0000:  # server_name
            hits.extend(_parse_server_name_ext(payload, ext_data_start, ext_data_end))
        p = ext_data_end
    return hits


def extract_sni_hits_from_tcp_stream(tcp_stream: bytes) -> List[SNIHit]:
    """
    Best-effort TLS ClientHello parser over a *reassembled* TCP byte stream.
    Returns list of SNIHit with name_off relative to the stream.
    """
    payload = tcp_stream
    hits: List[SNIHit] = []

    # TLS record header: content_type(22), version(0x03,xx), length(2)
    cand_starts: List[int] = []
    # SNI is in ClientHello at the very beginning of a TLS flow.
    # No need to scan 1MB; 64KB is more than enough for handshake records.
    max_scan = min(len(payload) - 5, 64 * 1024)
    for i in range(max_scan):
        if payload[i] != 22:
            continue
        if payload[i + 1] != 3:
            continue
        v_minor = payload[i + 2]
        if v_minor not in (1, 2, 3, 4):
            continue
        rec_len = _u16(payload[i + 3 : i + 5])
        if rec_len <= 0 or rec_len > 1 << 14:
            continue
        if i + 5 + rec_len > len(payload):
            continue
        cand_starts.append(i)
        if len(cand_starts) >= 32:
            break

    for start in cand_starts:
        i = start
        while i + 5 <= len(payload):
            ct = payload[i]
            rec_len = _u16(payload[i + 3 : i + 5])
            rec_start = i + 5
            rec_end = rec_start + rec_len
            if rec_end > len(payload):
                break
            if ct == 22:  # handshake
                h = rec_start
                while h + 4 <= rec_end:
                    hs_type = payload[h]
                    hs_len = _u24(payload[h + 1 : h + 4])
                    hs_start = h + 4
                    hs_end = hs_start + hs_len
                    if hs_end > rec_end:
                        break
                    if hs_type == 1:  # ClientHello
                        hits.extend(_extract_sni_from_clienthello(payload, hs_start, hs_end))
                    h = hs_end
            i = rec_end
        if hits:
            break
    return hits


def _mask_bytes_inplace(buf: bytearray, off: int, ln: int, *, mode: str) -> None:
    if ln <= 0:
        return
    if mode == "zero":
        buf[off : off + ln] = b"\x00" * ln
    elif mode == "random":
        buf[off : off + ln] = secrets.token_bytes(ln)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _tcp_reassemble_segments(pkts: List[object]) -> Tuple[bytearray, List[Tuple[int, int, int, bytes]]]:
    """
    Build a simple TCP byte stream from packets (assumes same direction).
    Returns (stream_bytes, segments) where segments are (rel_start, rel_end, pkt_i, payload_bytes).
    Best-effort; overlaps/gaps are tolerated (unfilled bytes left as 0).
    """
    from scapy.layers.inet import TCP  # type: ignore
    from scapy.packet import Raw  # type: ignore

    segs_seq: List[Tuple[int, int, int, bytes]] = []
    base_seq: Optional[int] = None
    for i, p in enumerate(pkts):
        if not p.haslayer(TCP) or not p.haslayer(Raw):
            continue
        tcp = p.getlayer(TCP)
        raw = p.getlayer(Raw)
        if tcp is None or raw is None:
            continue
        payload = bytes(getattr(raw, "load", b""))
        if not payload:
            continue
        seq = int(getattr(tcp, "seq", 0))
        if base_seq is None or seq < base_seq:
            base_seq = seq
        segs_seq.append((seq, seq + len(payload), i, payload))

    if base_seq is None or not segs_seq:
        return bytearray(), []

    segments: List[Tuple[int, int, int, bytes]] = []
    max_end = 0
    for seq_start, _seq_end, pkt_i, payload in segs_seq:
        rel_start = seq_start - base_seq
        rel_end = rel_start + len(payload)
        if rel_end > max_end:
            max_end = rel_end
        segments.append((rel_start, rel_end, pkt_i, payload))
    segments.sort(key=lambda x: x[0])

    stream = bytearray(max_end)
    filled = bytearray(max_end)
    # Sort by sequence number for consistent reassembly
    for rel_start, rel_end, _pkt_i, payload in sorted(segments, key=lambda x: x[0]):
        for j in range(len(payload)):
            pos = rel_start + j
            if not filled[pos]:
                stream[pos] = payload[j]
                filled[pos] = 1
    return stream, segments


def _patch_sni_hits_back_to_packets(
    hits: List[SNIHit],
    segments: List[Tuple[int, int, int, bytes]],
    pkts: List[object],
    *,
    mode: str,
) -> int:
    """
    Apply stream-relative SNIHit offsets back into per-packet Raw.load bytes.
    Returns number of host_name entries patched.
    """
    from scapy.packet import Raw  # type: ignore

    if not hits or not segments:
        return 0

    pkt_payloads: Dict[int, bytearray] = {}
    for _s, _e, pkt_i, payload in segments:
        if pkt_i not in pkt_payloads:
            pkt_payloads[pkt_i] = bytearray(payload)

    segments_sorted = sorted(segments, key=lambda x: x[0])

    def find_seg(pos: int, start_idx: int = 0) -> Tuple[int, Tuple[int, int, int, bytes]]:
        for k in range(start_idx, len(segments_sorted)):
            s, e, pkt_i, payload = segments_sorted[k]
            if s <= pos < e:
                return k, (s, e, pkt_i, payload)
        raise KeyError(pos)

    patched = 0
    for h in hits:
        pos = h.name_off
        remaining = h.name_len
        seg_idx = 0
        while remaining > 0:
            seg_idx, (s, e, pkt_i, _payload) = find_seg(pos, seg_idx)
            take = min(remaining, e - pos)
            within = pos - s
            _mask_bytes_inplace(pkt_payloads[pkt_i], within, take, mode=mode)
            pos += take
            remaining -= take
        patched += 1

    for pkt_i, ba in pkt_payloads.items():
        p = pkts[pkt_i]
        try:
            if p.haslayer("IP"):  # type: ignore
                del p["IP"].chksum  # type: ignore
            if p.haslayer("TCP"):  # type: ignore
                del p["TCP"].chksum  # type: ignore
        except Exception:
            pass
        p.getlayer(Raw).load = bytes(ba)  # type: ignore

    return patched


def mask_sni_in_pcap(
    in_pcap: str,
    out_pcap: str,
    *,
    mode: str = "zero",
) -> Dict[str, int]:
    """
    Mask SNI(host_name) in TLS ClientHello inside a pcap and write to `out_pcap`.
    Returns simple stats dict.
    """
    from scapy.all import PcapReader, wrpcap  # type: ignore
    from scapy.layers.inet import IP, TCP  # type: ignore
    from scapy.layers.l2 import Ether  # type: ignore
    from scapy.packet import Raw  # type: ignore

    pkts_all: List[object] = []
    # bidirectional canonical key: ((ip,port),(ip,port)) sorted
    bi_flows: Dict[Tuple[Tuple[str, int], Tuple[str, int]], List[int]] = {}

    with PcapReader(in_pcap) as pr:
        for pkt in pr:
            pkts_all.append(pkt)
            p = pkt
            if p.haslayer(Ether):
                p = p.getlayer(Ether).payload  # type: ignore
            ip = p.getlayer(IP)
            if ip is None or not ip.haslayer(TCP) or not p.haslayer(Raw):
                continue
            tcp = p.getlayer(TCP)
            if tcp is None:
                continue
            a = (str(ip.src), int(tcp.sport))
            b = (str(ip.dst), int(tcp.dport))
            key = (a, b) if a <= b else (b, a)
            bi_flows.setdefault(key, []).append(len(pkts_all) - 1)

    masked_flows = 0
    masked_names = 0

    for (_a, _b), idxs in bi_flows.items():
        idxs = sorted(set(idxs))
        if not idxs:
            continue

        flow_pkts_ip: List[object] = []
        for pi in idxs:
            pkt = pkts_all[pi]
            p = pkt
            if p.haslayer(Ether):
                p = p.getlayer(Ether).payload  # type: ignore
            flow_pkts_ip.append(p)

        p0 = flow_pkts_ip[0]
        ip0 = p0.getlayer(IP)
        tcp0 = p0.getlayer(TCP)
        if ip0 is None or tcp0 is None:
            continue
        t0 = (str(ip0.src), int(tcp0.sport), str(ip0.dst), int(tcp0.dport))

        dir1: List[object] = []
        dir2: List[object] = []
        for p in flow_pkts_ip:
            ip = p.getlayer(IP)
            tcp = p.getlayer(TCP)
            if ip is None or tcp is None or not p.haslayer(Raw):
                continue
            t = (str(ip.src), int(tcp.sport), str(ip.dst), int(tcp.dport))
            if t == t0:
                dir1.append(p)
            else:
                dir2.append(p)

        flow_patched = False
        for direction_pkts in (dir1, dir2):
            stream, segs = _tcp_reassemble_segments(direction_pkts)
            if not stream:
                continue
            hits = extract_sni_hits_from_tcp_stream(bytes(stream))
            if not hits:
                continue
            n = _patch_sni_hits_back_to_packets(hits, segs, direction_pkts, mode=mode)
            if n > 0:
                masked_flows += 1
                masked_names += n
                flow_patched = True
                break
        if flow_patched:
            continue

    wrpcap(out_pcap, pkts_all)
    return {
        "pcap_packets": len(pkts_all),
        "flows_with_sni_masked": masked_flows,
        "host_name_entries_masked": masked_names,
    }


# ---------------------------------------------------------------------------
# Dataset batch: walk label dirs, optional parallelism
# ---------------------------------------------------------------------------


@dataclass
class Job:
    in_path: Path
    out_path: Path


def _iter_pcap_jobs(input_root: Path, output_root: Path) -> List[Job]:
    jobs: List[Job] = []
    exts = (".pcap", ".pcapng")
    for label_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        for p in label_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                rel = p.relative_to(input_root)
                jobs.append(Job(in_path=p, out_path=output_root / rel))
    return jobs


def _process_one(job: Job, *, mode: str, overwrite: bool, dry_run: bool) -> Tuple[Job, Optional[dict], Optional[str]]:
    try:
        if job.out_path.exists() and not overwrite:
            return job, None, "skip_exists"
        if dry_run:
            return job, None, None
        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        stats = mask_sni_in_pcap(str(job.in_path), str(job.out_path), mode=mode)
        return job, stats, None
    except Exception as e:
        return job, None, f"error: {type(e).__name__}: {e}"


@dataclass
class MaskSniSummary:
    total_jobs: int
    done: int
    skipped: int
    errors: int
    masked_flows: int
    masked_names: int


def run_mask_sni_dataset(
    input_root: Path,
    output_root: Path,
    *,
    mode: str = "zero",
    workers: int = 1,
    overwrite: bool = True,
    dry_run: bool = False,
) -> MaskSniSummary:
    """
    Mask SNI in all PCAP/PCAPNG under input_root (label-dir layout) into output_root.
    Callable from other scripts or the CLI below.
    """
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input_root is not a directory: {input_root}")
    if output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root):
        raise ValueError("Choose separate, non-overlapping input and output directories.")

    jobs = _iter_pcap_jobs(input_root, output_root)
    if not jobs:
        raise FileNotFoundError(f"No .pcap/.pcapng under label dirs: {input_root}")

    done = skipped = errors = 0
    masked_flows = masked_names = 0

    if workers <= 1:
        for j in jobs:
            job, stats, err = _process_one(j, mode=mode, overwrite=overwrite, dry_run=dry_run)
            if err == "skip_exists":
                skipped += 1
            elif err is not None:
                errors += 1
                print(f"[!] {job.in_path} -> {job.out_path} : {err}")
            else:
                done += 1
                if stats:
                    masked_flows += int(stats.get("flows_with_sni_masked", 0))
                    masked_names += int(stats.get("host_name_entries_masked", 0))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_process_one, j, mode=mode, overwrite=overwrite, dry_run=dry_run)
                for j in jobs
            ]
            for k, fut in enumerate(as_completed(futs), start=1):
                job, stats, err = fut.result()
                if err == "skip_exists":
                    skipped += 1
                elif err is not None:
                    errors += 1
                    print(f"[!] {job.in_path} -> {job.out_path} : {err}")
                else:
                    done += 1
                    if stats:
                        masked_flows += int(stats.get("flows_with_sni_masked", 0))
                        masked_names += int(stats.get("host_name_entries_masked", 0))
                if k % 200 == 0:
                    print(f"[*] progress: {k}/{len(jobs)} (done={done} skip={skipped} err={errors})")

    return MaskSniSummary(
        total_jobs=len(jobs),
        done=done,
        skipped=skipped,
        errors=errors,
        masked_flows=masked_flows,
        masked_names=masked_names,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Mask SNI from a pcap dataset while preserving directory structure.")
    ap.add_argument("--input_root", type=str, required=True, help="Dataset root: DATASET_NAME/<label>/*.pcap")
    ap.add_argument("--output_root", type=str, default=None, help="Output root (default: <input_root>_no_sni)")
    ap.add_argument("--mode", choices=["zero", "random"], default="zero", help="Masking mode for SNI host_name bytes")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (process pool). Default 1 (sequential).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output pcaps")
    ap.add_argument("--dry_run", action="store_true", help="Only print planned work; do not write any files")
    args = ap.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"--input_root is not a directory: {input_root}")

    if args.output_root is None:
        output_root = Path(str(input_root) + "_no_sni")
    else:
        output_root = Path(args.output_root).expanduser().resolve()

    print(f"[*] input_root : {input_root}")
    print(f"[*] output_root: {output_root}")
    print(f"[*] mode={args.mode} workers={args.workers} overwrite={args.overwrite} dry_run={args.dry_run}")

    try:
        summary = run_mask_sni_dataset(
            input_root,
            output_root,
            mode=args.mode,
            workers=args.workers,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    print(f"[*] pcaps found : {summary.total_jobs}")
    print(f"[*] finished. done={summary.done} skip_exists={summary.skipped} errors={summary.errors}")
    if not args.dry_run:
        print(
            f"[*] total masked flows={summary.masked_flows} "
            f"total masked host_name entries={summary.masked_names}"
        )


if __name__ == "__main__":
    os.chdir(os.getcwd())
    main()
