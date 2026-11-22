"""Build an unlabelled pretraining cache from the VisQUIC pcaps."""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in ['reproduction', '.', 'data_process', 'data_process/vendor/cicflowmeter']:
    sys.path.insert(0, str(ROOT / extra))
from fast_pcap import read_pcap          # noqa: E402
from firstn import FirstNFlowSession     # noqa: E402
from build_bytes import _sanitize_raw    # noqa: E402

SRC = Path(os.environ.get('TRAFFIC_CBM_VISQUIC_ROOT', ROOT/'dataset'/'VisQUIC'))
OUT = ROOT / 'outputs/reproduction/pretrain_visquic.pt'
# Statistics must live on the same scale as the in-domain caches, so reuse their scaler.
_CACHE = ROOT / 'outputs/reproduction/cache/CipherSpectrum-AES128_corrected_full.pt'
_STAT_SCALER = None


def stat_scaler():
    global _STAT_SCALER
    if _STAT_SCALER is None:
        import torch as _torch
        _STAT_SCALER = _torch.load(str(_CACHE), map_location='cpu', mmap=True)['stat_scaler']
    return _STAT_SCALER
LIMIT = 30000


def worker(path):
    packets = read_pcap(Path(path), 200)
    if not packets:
        return None
    session = FirstNFlowSession(max_packets_per_flow=200, collect_in_memory=True, output_mode='csv',
                                output='/dev/null', verbose=False, min_packets_to_keep=1)
    for p in packets:
        session.process(p)
    session.flush_flows()
    if not session.rows:
        return None
    row = session.rows[0]
    stat = np.nan_to_num(np.asarray([float(row[k]) for k in COLUMNS], dtype=np.float64),
                         nan=0, posinf=0, neginf=0)
    stat = stat_scaler().transform(stat.reshape(1, -1))[0].astype(np.float32)
    seq = np.zeros((100, 7), np.float32)
    base = packets[0].ip.src
    last = None
    for i, p in enumerate(packets[:100]):
        ts = float(p.time)
        iat = 0.0 if last is None else max(0.0, ts - last)
        last = ts
        layer = p.tcp if p.tcp is not None else p.udp
        payload = len(layer.payload) if layer is not None else 0
        header = (p.ip.len - payload) if p.ip.len else 0
        flags = (p.tcp.flags & 0x3f) if p.tcp is not None else 0
        window = p.tcp.window if p.tcp is not None else 0
        seq[i] = [p.frame_len, iat, 1.0 if p.ip.src == base else -1.0, flags, payload, header, window]
    value = np.full((5, 63), 65536, np.int64)
    mask = np.zeros((5, 63), np.int64)
    for i, p in enumerate(packets[:5]):
        raw = _sanitize_raw(b'\0' * 12 + p.raw[12:], 'zero', False)
        chunk = np.frombuffer(raw[38:102], dtype=np.uint8).astype(np.int64)
        if len(chunk) > 1:
            tokens = chunk[:-1] * 256 + chunk[1:]
            value[i, :len(tokens)] = tokens
            mask[i, :len(tokens)] = 1
    steps = min(100, len(packets))
    return dict(stat=stat, seq=seq, attn=(seq.any(1)).astype(np.float32), attn_len=steps,
                x_bytes=value, bytes_mask=mask)


def main():
    global COLUMNS
    COLUMNS = json.loads(_CACHE.with_suffix('.json').read_text())['stat_columns']
    if not SRC.is_dir():
        raise FileNotFoundError(SRC)
    files = sorted(SRC.glob('*.pcap'))[:LIMIT]
    print(f'files={len(files)} columns={len(COLUMNS)}', flush=True)
    start = time.perf_counter()
    context = multiprocessing.get_context('spawn')
    rows = []
    with ProcessPoolExecutor(max_workers=8, mp_context=context) as pool:
        for i, row in enumerate(pool.map(worker, files, chunksize=16)):
            if row is not None:
                rows.append(row)
            if (i + 1) % 5000 == 0:
                print(f'  {i + 1}/{len(files)} kept={len(rows)} {time.perf_counter() - start:.0f}s', flush=True)
    print(f'kept {len(rows)}/{len(files)} in {time.perf_counter() - start:.0f}s', flush=True)

    stat = np.stack([r['stat'] for r in rows])
    seq = np.stack([r['seq'] for r in rows])
    # Match the canonical pipeline: standardise the sequence features with a scaler
    # fitted on the pretraining train split only.
    from sklearn.preprocessing import StandardScaler
    flat = seq.reshape(-1, seq.shape[-1])
    seq_scaler = StandardScaler().fit(flat[:int(len(flat) * 0.95)])
    seq = seq_scaler.transform(flat).reshape(seq.shape).astype(np.float32)
    attn = np.stack([r['attn'] for r in rows]).astype(np.float32)
    x_bytes = np.stack([r['x_bytes'] for r in rows])
    bytes_mask = np.stack([r['bytes_mask'] for r in rows])
    labels = np.zeros(len(rows), np.int64)
    order = np.arange(len(rows))
    rng = np.random.default_rng(42)
    rng.shuffle(order)
    cut = int(len(order) * 0.95)
    def pack(index):
        return dict(x_stat=torch.from_numpy(stat[index]), x_seq=torch.from_numpy(seq[index]),
                    attn_mask=torch.from_numpy(attn[index]), x_bytes=torch.from_numpy(x_bytes[index]),
                    bytes_mask=torch.from_numpy(bytes_mask[index]), label=torch.from_numpy(labels[index]))
    cache = dict(splits=dict(train=pack(order[:cut]), val=pack(order[cut:])),
                 protocol=dict(dataset='VisQUIC', selection='unlabelled QUIC connections, pretraining only',
                               statistics='raw; per-dataset normalisers are refit at fine-tuning time',
                               flows=int(len(rows)), source=str(SRC), seed=42,
                               normalization='standardised sequence features (train-split scaler)'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, OUT)
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)', flush=True)


if __name__ == '__main__':
    main()
