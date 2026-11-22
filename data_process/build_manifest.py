#!/usr/bin/env python3
"""
build_manifest.py - Build a manifest CSV that records which modalities are available per flow.

Output: data/{name}/manifest.csv
  flow_id, label, split, has_stats, has_seq, has_bytes

  split: "train" | "val" | "test"

Usage:
  python build_manifest.py \
    --dataset_dir  ./dataset/USTC-TFC2016 \
    --data_dir     ./data/USTC-TFC2016 \
    --out          ./data/USTC-TFC2016/manifest.csv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Set


def load_ids(txt: Path, stem_to_fid: dict = None) -> list:
    """Load ids and normalize to label/stem format.

    Three cases:
    1. Already "label/stem"       → keep as-is
    2. USTC style "Label__stem"   → prepend label = split("__")[0]
    3. No "/" and no "__"         → look up in stem_to_fid (built from parquet flow_ids);
                                    covers CipherSpectrum-style stems like
                                    "traffic_2024-01-18_adblockplus.org_..."
    """
    ids = []
    for line in txt.read_text().splitlines():
        fid = line.strip()
        if not fid:
            continue
        if "/" not in fid:
            if "__" in fid:
                label = fid.split("__", 1)[0]
                fid = f"{label}/{fid}"
            elif stem_to_fid and fid in stem_to_fid:
                fid = stem_to_fid[fid]
        ids.append(fid)
    return ids


def load_stats_ids(stats_path: Path) -> Set[str]:
    if not stats_path.exists():
        return set()
    ids: Set[str] = set()
    if stats_path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(stats_path)
        if "flow_id" in df.columns:
            ids = set(df["flow_id"].astype(str))
        return ids
    with stats_path.open("r", encoding="utf-8") as f:
        rr = csv.reader(f)
        header = next(rr, None)
        if header is None:
            return ids
        for row in rr:
            if row:
                ids.add(row[0])
    return ids


def load_seq_ids(seq_dir: Path) -> Set[str]:
    """Load flow_ids from seq/seq.parquet (single file) or seq/data/*.parquet / seq/*.parquet (legacy)."""
    if not seq_dir.exists():
        return set()
    ids: Set[str] = set()
    single = seq_dir / "seq.parquet"
    if single.exists():
        parquets = [single]
    else:
        data_dir = seq_dir / "data"
        parquets = list(data_dir.glob("*.parquet")) if data_dir.exists() else list(seq_dir.glob("*.parquet"))
    if parquets:
        import pandas as pd
        for pf in parquets:
            try:
                df = pd.read_parquet(pf, columns=["flow_id"])
                ids.update(df["flow_id"].astype(str))
            except Exception:
                pass
        return ids
    for label_dir in seq_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for npy in label_dir.glob("*.npy"):
            fid = f"{label_dir.name}/{npy.stem}"
            ids.add(fid)
    return ids


def load_bytes_ids(bytes_path: Path) -> Set[str]:
    """Load flow_ids from bytes/bytes.parquet or bytes.tsv (legacy)."""
    if not bytes_path.exists():
        return set()
    ids: Set[str] = set()
    if bytes_path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(bytes_path, columns=["flow_id"])
        ids = set(df["flow_id"].astype(str))
        return ids
    with bytes_path.open("r", encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            fid = line.split("\t", 1)[0].strip()
            if fid:
                ids.add(fid)
    return ids


def build_manifest(
    dataset_dir: Path,
    data_dir: Path,
    out_csv: Path,
) -> None:
    if out_csv.parent.resolve() == dataset_dir.resolve():
        raise ValueError("Write the manifest and aligned IDs in a separate feature directory.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build stem→full_id index from parquet flow_ids for datasets whose
    # train_ids.txt has stems without label prefix (e.g. CipherSpectrum).
    stats_path_tmp = data_dir / "stats" / "stats.parquet"
    if not stats_path_tmp.exists():
        stats_path_tmp = data_dir / "stats.csv"
    stem_to_fid: dict = {}
    if stats_path_tmp.exists() and stats_path_tmp.suffix == ".parquet":
        try:
            import pandas as pd
            _df = pd.read_parquet(stats_path_tmp, columns=["flow_id"])
            for full_id in _df["flow_id"].astype(str):
                if "/" in full_id:
                    stem_to_fid[full_id.split("/", 1)[1]] = full_id
        except Exception:
            pass

    train_ids = load_ids(dataset_dir / "train_ids.txt", stem_to_fid)
    val_ids   = load_ids(dataset_dir / "val_ids.txt",   stem_to_fid)
    test_ids  = load_ids(dataset_dir / "test_ids.txt",  stem_to_fid)
    ids = train_ids + val_ids + test_ids
    if len(ids) != len(set(ids)):
        raise ValueError("Each flow ID must appear once across the train, validation, and test splits.")
    label2id: Dict[str, int] = json.loads((dataset_dir / "label2id.json").read_text())

    print(f"[build_manifest] train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # Build availability sets (stats.parquet or stats.csv)
    stats_path = data_dir / "stats" / "stats.parquet"
    if not stats_path.exists():
        stats_path = data_dir / "stats.csv"
    stats_ids = load_stats_ids(stats_path)
    bytes_parquet = data_dir / "bytes" / "bytes.parquet"
    bytes_legacy = data_dir / "bytes.tsv"
    bytes_ids = load_bytes_ids(bytes_parquet) if bytes_parquet.exists() else load_bytes_ids(bytes_legacy)
    seq_dir   = data_dir / "seq"
    seq_ids   = load_seq_ids(seq_dir)

    print(f"  stats : {len(stats_ids)}")
    print(f"  bytes : {len(bytes_ids)}")
    print(f"  seq   : {len(seq_ids)}")

    rows = []
    for fid, split in [(fid, "train") for fid in train_ids] + \
                      [(fid, "val")  for fid in val_ids] + \
                      [(fid, "test") for fid in test_ids]:
        parts = fid.split("/", 1)
        if len(parts) != 2:
            continue
        label, stem = parts
        label_id = label2id.get(label, -1)
        has_stats = int(fid in stats_ids)
        has_bytes = int(fid in bytes_ids)
        has_seq   = int(fid in seq_ids)
        rows.append((fid, label, label_id, split, has_stats, has_seq, has_bytes))

    # Write manifest CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["flow_id", "label", "label_id", "split", "has_stats", "has_seq", "has_bytes"])
        wr.writerows(rows)

    # Write aligned IDs: only flows with ALL three modalities available
    aligned_train = [r[0] for r in rows if r[3] == "train" and r[4] and r[5] and r[6]]
    aligned_val   = [r[0] for r in rows if r[3] == "val"   and r[4] and r[5] and r[6]]
    aligned_test  = [r[0] for r in rows if r[3] == "test"  and r[4] and r[5] and r[6]]

    (out_csv.parent / "train_ids.txt").write_text("\n".join(aligned_train) + "\n")
    (out_csv.parent / "val_ids.txt").write_text("\n".join(aligned_val) + "\n")
    (out_csv.parent / "test_ids.txt").write_text("\n".join(aligned_test) + "\n")

    print(f"  all 3 modalities — train: {len(aligned_train)} / {len(train_ids)}"
          f"   val: {len(aligned_val)} / {len(val_ids)}   test: {len(aligned_test)} / {len(test_ids)}")

    # Copy label2id.json so data/{name}/ is self-contained (no dependency on dataset/)
    import shutil
    src = dataset_dir / "label2id.json"
    dst = out_csv.parent / "label2id.json"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied label2id.json -> {dst}")

    print(f"[DONE] manifest    -> {out_csv}")
    print(f"       ids -> {out_csv.parent}/{{train,val,test}}_ids.txt")


def main() -> None:
    ap = argparse.ArgumentParser("Build three-modality availability manifest CSV")
    ap.add_argument("--dataset_dir", required=True,
                    help="dataset/{name}/ (contains train_ids.txt, val_ids.txt, test_ids.txt, label2id.json)")
    ap.add_argument("--data_dir", required=True,
                    help="data/{name}/ (contains stats.csv, bytes.tsv, seq/)")
    ap.add_argument("--out", required=True, help="Output manifest CSV path")
    args = ap.parse_args()

    build_manifest(
        dataset_dir=Path(args.dataset_dir),
        data_dir=Path(args.data_dir),
        out_csv=Path(args.out),
    )


if __name__ == "__main__":
    main()
