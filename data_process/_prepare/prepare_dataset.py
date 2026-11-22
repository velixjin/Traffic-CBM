#!/usr/bin/env python3
"""
prepare_dataset.py - Universal PCAP flow dataset preparation script.

Pipeline:
  1. [Slice]   raw/<label>/*.pcap  --SplitCap-->  flows_raw/<label>/*.pcap
  2. [Filter]  flows_raw/  ------------>  flows_filtered/<label>/*.pcap
  3. [Split]   flows_filtered/  ------------>  train_ids.txt (80%) + val_ids.txt (10%) + test_ids.txt (10%)
  4. [Meta]            ------------>  label2id.json + label_counts.json

Usage:
  python prepare_dataset.py \\
    --raw   /path/to/raw \\
    --out   ./dataset/USTC-TFC2016 \\
    [--splitcap_exe /path/to/SplitCap.exe]
    [--skip_slice]        # skip SplitCap, use existing flows_raw/
    [--min_packets 3]
    [--min_bytes 2048]
    [--min_class_count 30]
    [--test_ratio 0.1] [--val_ratio 0.1]
    [--seed 42]
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scapy.utils import RawPcapReader  # type: ignore
    _HAS_SCAPY = True
except ImportError:
    _HAS_SCAPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_pcaps(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in (".pcap", ".pcapng") and (p.is_file() or p.is_symlink())
    )


# ---------------------------------------------------------------------------
# Step 1: Slice (SplitCap) directly into flows/
# ---------------------------------------------------------------------------

def _normalize_splitcap_name(name: str) -> str:
    """
    Strip source-file prefix from SplitCap output name and sanitize.
    e.g. 'BitTorrent-pcap-TCP_1-1-0-102_3151_...' -> 'TCP_1-1-0-102_3151_...'
    """
    stem, ext = os.path.splitext(name)
    body = stem
    if "-pcap-" in body:
        body = body.split("-pcap-")[-1]
    else:
        tokens = ("TCP_", "UDP_", "ICMP_", "SCTP_", "GRE_", "ESP_", "IPV6-ICMP_", "ICMPV6_")
        cut_at = -1
        for t in tokens:
            idx = body.find(t)
            if idx != -1:
                cut_at = idx if cut_at == -1 else min(cut_at, idx)
        if cut_at != -1:
            body = body[cut_at:]
    return body.replace(".", "-").replace(" ", "_") + ext


def _unique_target(target: Path) -> Path:
    if not target.exists():
        return target
    stem, ext = target.stem, target.suffix
    h = hashlib.md5(str(target).encode()).hexdigest()[:8]
    return target.with_name(f"{stem}__{h}{ext}")


_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


def _is_pcapng(path: Path) -> bool:
    """Return True if the file starts with pcapng magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _PCAPNG_MAGIC
    except OSError:
        return False


def _convert_pcapng_to_pcap(src: Path, dst: Path) -> None:
    """Convert a pcapng file to legacy pcap format using scapy."""
    from scapy.all import rdpcap, wrpcap
    pkts = rdpcap(str(src))
    wrpcap(str(dst), pkts)


def _run_splitcap(exe: Path, src: Path, tmp_dir: Path) -> List[Path]:
    """Run SplitCap into an isolated tmp_dir and return created pcap files.

    If src is pcapng format (magic 0x0A0D0D0A), it is first converted to
    legacy pcap via scapy, since SplitCap only accepts pcap format.
    """
    ensure_dir(tmp_dir)
    real_src = src.resolve()
    # SplitCap cannot handle pcapng; convert on-the-fly if needed
    if _is_pcapng(real_src):
        converted = tmp_dir / (real_src.stem + "__converted.pcap")
        _convert_pcapng_to_pcap(real_src, converted)
        real_src = converted
    cmd = ["mono", str(exe.resolve()), "-r", str(real_src), "-s", "session", "-o", str(tmp_dir.resolve())]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"SplitCap failed on {src.name}: {msg}") from e
    # Collect all pcap/pcapng files created
    return sorted(
        p for p in tmp_dir.iterdir()
        if p != real_src and p.is_file() and p.suffix.lower() in (".pcap", ".pcapng")
    )


def _slice_one_pcap(args_tuple: Tuple[str, str, str, str]) -> Tuple[str, int, Optional[str]]:
    """
    Worker: slice one source PCAP into its own temp subdir, then move results to dst_dir.
    Each worker has an isolated output directory to avoid race conditions.
    """
    exe_str, src_str, dst_str, label = args_tuple
    exe, src, dst_dir = Path(exe_str), Path(src_str), Path(dst_str)
    # Unique temp subdir per source file
    tmp_dir = dst_dir / f".tmp_{hashlib.md5(src_str.encode()).hexdigest()[:12]}"
    try:
        created = _run_splitcap(exe, src, tmp_dir)
        count = 0
        for fp in created:
            new_name = f"{label}__{_normalize_splitcap_name(fp.name)}"
            tgt = _unique_target(dst_dir / new_name)
            fp.rename(tgt)
            count += 1
        return label, count, None
    except Exception as e:
        return label, 0, str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def step_slice(raw_root: Path, flows_out: Path, exe: Path, workers: int = 8) -> None:
    """Slice raw/<label>/*.pcap into flows_raw/<label>/*.pcap via SplitCap (parallel)."""
    if shutil.which("mono") is None:
        print("[ERR] mono not found. Install with: apt install mono-runtime", file=sys.stderr)
        sys.exit(1)
    if not exe.exists():
        print(f"[ERR] SplitCap.exe not found: {exe}", file=sys.stderr)
        sys.exit(1)
    if not raw_root.exists():
        print(f"[ERR] --raw directory not found: {raw_root}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(flows_out)
    labels = sorted(d.name for d in raw_root.iterdir() if d.is_dir())

    # Clean up any leftover .tmp_* dirs from interrupted previous runs
    for label in labels:
        for d in (flows_out / label).glob(".tmp_*"):
            shutil.rmtree(d, ignore_errors=True)

    # Build task list: one task per source PCAP file
    tasks: List[Tuple[str, str, str, str]] = []
    for label in labels:
        dst_dir = flows_out / label
        ensure_dir(dst_dir)
        for pcap in list_pcaps(raw_root / label):
            tasks.append((str(exe), str(pcap), str(dst_dir), label))

    print(f"  {len(tasks)} source PCAPs across {len(labels)} classes, workers={workers}")

    total = 0
    label_counts: Dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_slice_one_pcap, t): t for t in tasks}
        for fut in as_completed(futures):
            label, count, err = fut.result()
            if err:
                print(f"  [WARN] {err}", file=sys.stderr)
            else:
                total += count
                label_counts[label] = label_counts.get(label, 0) + count

    for label in labels:
        print(f"  [{label}] {label_counts.get(label, 0)} flows")
    print(f"[DONE] slice: {total} flows, {len(labels)} classes -> {flows_out}")


# ---------------------------------------------------------------------------
# Step 2: Filter in-place inside flows/
# ---------------------------------------------------------------------------

def _check_one_flow(args_tuple: Tuple[str, int, int]) -> Tuple[str, bool, str]:
    """Worker: check one pcap file. Returns (path_str, passed, reason)."""
    path_str, min_packets, min_bytes = args_tuple
    fp = Path(path_str)
    size = fp.stat().st_size
    if size < min_bytes:
        return path_str, False, "size"
    if _HAS_SCAPY and min_packets > 0:
        try:
            rp = RawPcapReader(path_str)
            cnt = 0
            for _ in rp:
                cnt += 1
                if cnt >= min_packets:
                    break
            rp.close()
            if cnt < min_packets:
                return path_str, False, "packets"
        except Exception:
            pass
    return path_str, True, ""


def step_filter(flows_raw: Path, flows_out: Path,
                min_packets: int, min_bytes: int,
                min_class_count: int,
                workers: int = 8) -> Dict[str, List[str]]:
    """
    Filter flows_raw/ -> flows/ via hardlinks (parallel).
    flows_raw/ is read-only and kept intact.
    flows/ is rebuilt from scratch each run.
    Writes per-class detail to filter.log in the dataset root.
    Returns {label: [flow_id, ...]} of surviving flows.
    """
    if not flows_raw.exists():
        print(f"[ERR] flows_raw/ not found: {flows_raw}", file=sys.stderr)
        sys.exit(1)

    # Rebuild flows/ from scratch
    if flows_out.exists():
        shutil.rmtree(flows_out)
    ensure_dir(flows_out)

    labels = sorted(d.name for d in flows_raw.iterdir() if d.is_dir() and not d.name.startswith("."))

    # Collect all tasks, track raw count per label
    all_tasks: List[Tuple[str, int, int]] = []
    label_of: Dict[str, str] = {}
    raw_counts: Dict[str, int] = {}
    for label in labels:
        pcaps = list_pcaps(flows_raw / label)
        raw_counts[label] = len(pcaps)
        for fp in pcaps:
            all_tasks.append((str(fp), min_packets, min_bytes))
            label_of[str(fp)] = label

    kept: Dict[str, List[str]] = {}
    del_size: Dict[str, int] = {}
    del_pkts: Dict[str, int] = {}
    removed_flow = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path_str, passed, reason in pool.map(_check_one_flow, all_tasks, chunksize=256):
            fp = Path(path_str)
            label = label_of[path_str]
            if passed:
                dst_dir = flows_out / label
                ensure_dir(dst_dir)
                tgt = dst_dir / fp.name
                try:
                    os.link(str(fp.resolve()), str(tgt))
                except OSError:
                    shutil.copy2(str(fp.resolve()), str(tgt))
                kept.setdefault(label, []).append(f"{label}/{fp.stem}")
            else:
                removed_flow += 1
                if reason == "size":
                    del_size[label] = del_size.get(label, 0) + 1
                else:
                    del_pkts[label] = del_pkts.get(label, 0) + 1

    # Phase 2: drop small classes
    final: Dict[str, List[str]] = {}
    dropped_classes: List[str] = []
    for label in labels:
        ids = kept.get(label, [])
        if len(ids) >= min_class_count:
            final[label] = ids
        else:
            dropped_classes.append(label)
            shutil.rmtree(flows_out / label, ignore_errors=True)

    total = sum(len(v) for v in final.values())

    # Write log
    log_path = flows_out.parent / "filter.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"min_packets={min_packets}  min_bytes={min_bytes}  min_class_count={min_class_count}\n")
        f.write(f"total: {total} kept, {removed_flow} deleted, "
                f"{len(dropped_classes)} classes dropped ({len(labels)} -> {len(final)})\n\n")
        f.write(f"{'class':<20} {'raw':>8} {'kept':>8} {'del_size':>10} {'del_pkts':>10}  status\n")
        f.write("-" * 72 + "\n")
        for label in labels:
            n_kept = len(kept.get(label, []))
            n_size = del_size.get(label, 0)
            n_pkts = del_pkts.get(label, 0)
            if label in dropped_classes:
                status = f"DROPPED (kept={n_kept} < min_class_count={min_class_count})"
            else:
                status = "ok"
            f.write(f"{label:<20} {raw_counts[label]:>8} {n_kept:>8} {n_size:>10} {n_pkts:>10}  {status}\n")

    print(f"[DONE] filter: {total} kept, {removed_flow} deleted, "
          f"{len(dropped_classes)} classes dropped ({len(labels)} -> {len(final)} classes)")
    print(f"  log -> {log_path}")
    return final


# ---------------------------------------------------------------------------
# Step 3: Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(by_label: Dict[str, List[str]],
                      test_ratio: float, val_ratio: float, seed: int
                      ) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_ids, val_ids, test_ids) with per-class stratified sampling (8:1:1 by default)."""
    train_all: List[str] = []
    val_all: List[str] = []
    test_all: List[str] = []

    for label, ids in sorted(by_label.items()):
        n = len(ids)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_test + n_val >= n:
            n_test = max(1, n // 3)
            n_val = max(1, (n - n_test) // 2)
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_test - 1)

        h = int(hashlib.md5(f"{label}|{seed}".encode()).hexdigest()[:8], 16)
        rng = random.Random(h)
        perm = list(ids)
        rng.shuffle(perm)
        test_all.extend(sorted(perm[:n_test]))
        val_all.extend(sorted(perm[n_test : n_test + n_val]))
        train_all.extend(sorted(perm[n_test + n_val :]))

    def by_stem(flow_id: str) -> Tuple[str, str]:
        # Keep the original ordering for unique stems, with a stable tie break.
        return flow_id.split("/", 1)[-1], flow_id

    return sorted(train_all, key=by_stem), sorted(val_all, key=by_stem), sorted(test_all, key=by_stem)


# ---------------------------------------------------------------------------
# Step 4: Write meta files
# ---------------------------------------------------------------------------

def write_ids(path: Path, ids: List[str]) -> None:
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def write_meta(out: Path, by_label: Dict[str, List[str]],
               train_ids: List[str], val_ids: List[str], test_ids: List[str],
               args: argparse.Namespace) -> None:
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    labels_sorted = sorted(by_label.keys())
    label2id = {lab: idx for idx, lab in enumerate(labels_sorted)}
    (out / "label2id.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts = {}
    for lab in labels_sorted:
        ids = by_label[lab]
        counts[lab] = {
            "total": len(ids),
            "train": sum(1 for fid in ids if fid in train_set),
            "val": sum(1 for fid in ids if fid in val_set),
            "test": sum(1 for fid in ids if fid in test_set),
        }
    meta = {
        "params": {
            "min_packets": args.min_packets,
            "min_bytes": args.min_bytes,
            "min_class_count": args.min_class_count,
            "test_ratio": args.test_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "summary": {
            "num_classes": len(labels_sorted),
            "total_flows": sum(v["total"] for v in counts.values()),
            "train_flows": len(train_ids),
            "val_flows": len(val_ids),
            "test_flows": len(test_ids),
        },
        "classes": counts,
    }
    (out / "label_counts.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[DONE] meta: {len(labels_sorted)} classes, "
          f"{meta['summary']['total_flows']} flows "
          f"(train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})")
    print(f"  label2id     -> {out / 'label2id.json'}")
    print(f"  label_counts -> {out / 'label_counts.json'}")
    print(f"  train_ids    -> {out / 'train_ids.txt'}")
    print(f"  val_ids      -> {out / 'val_ids.txt'}")
    print(f"  test_ids     -> {out / 'test_ids.txt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Prepare a per-flow PCAP dataset: slice → filter (in-place) → split → meta",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw", type=str, default=None,
                    help="Raw PCAP root: raw/<label>/*.pcap (required unless --pre_sliced or --skip_slice)")
    ap.add_argument("--out", type=str, required=True,
                    help="Output dataset root (e.g. dataset/USTC-TFC2016)")
    ap.add_argument("--splitcap_exe", type=str,
                    default=os.environ.get("TRAFFIC_CBM_SPLITCAP"),
                    help="Path to SplitCap.exe (invoked via mono)")
    ap.add_argument("--skip_slice", action="store_true",
                    help="Skip SplitCap slicing; reuse existing flows_raw/ in --out")
    ap.add_argument("--pre_sliced", type=str, default=None,
                    help="Already-sliced directory organized as <label>/*.pcap; "
                         "hardlinked into flows/ then filtered+split (no SplitCap needed)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel workers for SplitCap slicing")
    ap.add_argument("--min_packets", type=int, default=3,
                    help="Drop flows with fewer than N packets")
    ap.add_argument("--min_bytes", type=int, default=2048,
                    help="Drop flows with file size smaller than N bytes")
    ap.add_argument("--min_class_count", type=int, default=30,
                    help="Drop classes with fewer than N valid flows")
    ap.add_argument("--test_ratio", type=float, default=0.1,
                    help="Fraction of flows per class for test set (default 0.1, 8:1:1 with val)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of flows per class for validation set (default 0.1, train=1-test-val)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for stratified split")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    out = Path(args.out).expanduser().resolve()
    source_arg = args.pre_sliced or (None if args.skip_slice else args.raw)
    if source_arg:
        source = Path(source_arg).expanduser().resolve()
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise SystemExit("Choose separate, non-overlapping input and output directories.")
    flows_raw = out / "flows_raw"      # permanent raw slices (SplitCap output or pre_sliced copy)
    flows_out = out / "flows_filtered" # filtered hardlinks, rebuilt on every run
    ensure_dir(out)

    # Step 1: Populate flows_raw/
    if args.pre_sliced:
        src = Path(args.pre_sliced).expanduser()
        if not src.is_dir():
            print(f"[ERR] --pre_sliced directory not found: {src}", file=sys.stderr)
            sys.exit(1)
        if flows_raw.exists():
            print(f"[INFO] flows_raw/ already exists, removing before re-link: {flows_raw}")
            shutil.rmtree(flows_raw)
        print("=" * 60)
        print(f"[STEP 1] Linking pre-sliced flows from {src} ...")
        labels = sorted(d.name for d in src.iterdir() if d.is_dir())
        total = 0
        for label in labels:
            dst_dir = flows_raw / label
            ensure_dir(dst_dir)
            for fp in list_pcaps(src / label):
                tgt = dst_dir / (fp.stem + ".pcap")
                if tgt.exists():
                    raise ValueError(f"Duplicate flow stem for class {label}: {fp.stem}")
                if _is_pcapng(fp):
                    _convert_pcapng_to_pcap(fp, tgt)
                else:
                    try:
                        os.link(str(fp.resolve()), str(tgt))
                    except OSError:
                        shutil.copy2(str(fp.resolve()), str(tgt))
                total += 1
        print(f"[DONE] linked {total} flows, {len(labels)} classes -> {flows_raw}")
    elif args.skip_slice:
        if not flows_raw.exists():
            print(f"[ERR] --skip_slice given but flows_raw/ not found: {flows_raw}", file=sys.stderr)
            sys.exit(1)
        print(f"[SKIP] slice (--skip_slice); reusing existing {flows_raw}")
    else:
        if args.raw is None:
            print("[ERR] --raw is required when not using --pre_sliced or --skip_slice", file=sys.stderr)
            sys.exit(1)
        raw = Path(args.raw).expanduser()
        if not raw.is_dir():
            raise FileNotFoundError(f"Raw capture directory does not exist: {raw}")
        if not args.splitcap_exe or not Path(args.splitcap_exe).expanduser().is_file():
            raise FileNotFoundError("Provide --splitcap_exe or set TRAFFIC_CBM_SPLITCAP.")
        if shutil.which("mono") is None:
            raise SystemExit("Mono is required for raw capture slicing.")
        if flows_raw.exists():
            print(f"[INFO] Rebuilding generated flow files: {flows_raw}")
            shutil.rmtree(flows_raw)
        print("=" * 60)
        print("[STEP 1] Slicing raw PCAPs via SplitCap ...")
        step_slice(raw, flows_raw, Path(args.splitcap_exe).expanduser(), workers=args.workers)

    # Step 2: Filter flows_raw/ -> flows/ (hardlinks, flows/ rebuilt each run)
    print("=" * 60)
    print("[STEP 2] Filtering flows ...")
    by_label = step_filter(flows_raw, flows_out, args.min_packets, args.min_bytes,
                           args.min_class_count, workers=args.workers)

    if not by_label:
        print("[ERR] No flows survived filtering.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Stratified split (train:val:test = 8:1:1 by default)
    print("=" * 60)
    print("[STEP 3] Stratified split (train:val:test) ...")
    train_ids, val_ids, test_ids = _stratified_split(
        by_label, args.test_ratio, args.val_ratio, args.seed
    )
    write_ids(out / "train_ids.txt", train_ids)
    write_ids(out / "val_ids.txt", val_ids)
    write_ids(out / "test_ids.txt", test_ids)
    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # Step 4: Meta
    print("=" * 60)
    print("[STEP 4] Writing meta files ...")
    write_meta(out, by_label, train_ids, val_ids, test_ids, args)

    print("=" * 60)
    print(f"[ALL DONE] -> {out}")


if __name__ == "__main__":
    main()
