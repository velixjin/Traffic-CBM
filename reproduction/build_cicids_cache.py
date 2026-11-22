"""Extract unlabelled pretraining flows from the CIC-IDS2017 captures.

Streams the pcaps, groups packets into bidirectional flows, and stores the same
statistics / sequence / bytes tensors used by the rest of the pretraining pipeline.
"""
import json
import sys
import time
from pathlib import Path
import os

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in ['reproduction', '.', 'data_process', 'data_process/vendor/cicflowmeter']:
    sys.path.insert(0, str(ROOT / extra))
from fast_pcap import iter_pcap                 # noqa: E402
from firstn import FirstNFlowSession            # noqa: E402
from build_bytes import _sanitize_raw, packets_to_tokens  # noqa: E402

SRC = Path(os.environ.get('TRAFFIC_CBM_CICIDS_ROOT', ROOT/'dataset'/'CIC-IDS2017'))
OUT = ROOT / 'outputs/reproduction/pretrain_cicids.pt'
_CACHE = ROOT / 'outputs/reproduction/cache/CipherSpectrum-AES128_corrected_full.pt'
MAX_FLOWS = 120000
MAX_PACKETS_PER_FLOW = 200
IDLE = 120.0


def flow_key(p):
    a = (p.ip.src, p.tcp.sport if p.tcp is not None else p.udp.sport)
    b = (p.ip.dst, p.tcp.dport if p.tcp is not None else p.udp.dport)
    if a > b:
        a, b = b, a
    return (a, b, p.proto)


def iter_flows(path, seen, budget):
    active = {}
    for p in iter_pcap(path):
        key = flow_key(p)
        ts = float(p.time)
        entry = active.get(key)
        if entry is None or len(entry['packets']) >= MAX_PACKETS_PER_FLOW or ts - entry['last'] > IDLE:
            if entry is not None and entry['packets']:
                yield entry['packets']
                seen += 1
            entry = dict(packets=[], last=ts)
            active[key] = entry
            if seen >= budget:
                return
        entry['packets'].append(p)
        entry['last'] = ts
    for entry in active.values():
        if entry['packets']:
            yield entry['packets']


def features(packets, stat_scaler, seq_scaler):
    session = FirstNFlowSession(max_packets_per_flow=MAX_PACKETS_PER_FLOW, collect_in_memory=True,
                                output_mode='csv', output='/dev/null', verbose=False, min_packets_to_keep=1)
    for p in packets:
        session.process(p)
    session.flush_flows()
    if not session.rows:
        return None
    stat = np.nan_to_num(np.asarray([float(session.rows[0][k]) for k in COLUMNS], dtype=np.float64),
                         nan=0, posinf=0, neginf=0)
    stat = stat_scaler.transform(stat.reshape(1, -1))[0]
    seq = np.zeros((100, 7), np.float32)
    base = packets[0].ip.src
    last = None
    n = min(100, len(packets))
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
    attn = np.zeros(100, np.float32); attn[:n] = 1.0
    value = np.full((5, 63), 65536, np.int64)
    mask = np.zeros((5, 63), np.int64)
    for i, p in enumerate(packets[:5]):
        raw = _sanitize_raw(b'\0' * 12 + p.raw[12:], 'zero', False)
        chunk = np.frombuffer(raw[38:102], dtype=np.uint8).astype(np.int64)
        if len(chunk) > 1:
            tokens = chunk[:-1] * 256 + chunk[1:]
            value[i, :len(tokens)] = tokens
            mask[i, :len(tokens)] = 1
    return dict(stat=stat.astype(np.float32), seq=seq, attn=attn, x_bytes=value, bytes_mask=mask)


def main():
    global COLUMNS
    COLUMNS = json.loads(_CACHE.with_suffix('.json').read_text())['stat_columns']
    if not SRC.is_dir():
        raise FileNotFoundError(SRC)
    stat_scaler = torch.load(str(_CACHE), map_location='cpu', mmap=True)['stat_scaler']
    seq_scaler = torch.load(str(_CACHE), map_location='cpu', mmap=True)['seq_scaler']
    from sklearn.preprocessing import StandardScaler
    files = sorted(SRC.glob('*.classic.pcap'))
    print(f'files={len(files)} budget={MAX_FLOWS}', flush=True)
    rows, seen = [], 0
    start = time.perf_counter()
    for path in files:
        if seen >= MAX_FLOWS:
            break
        t0 = time.perf_counter()
        before = seen
        for packets in iter_flows(path, seen, MAX_FLOWS):
            seen += 1
            row = features(packets, stat_scaler, seq_scaler)
            if row is not None:
                rows.append(row)
            if seen % 10000 == 0:
                print(f'  {path.name} flows={seen} kept={len(rows)} {time.perf_counter() - start:.0f}s', flush=True)
        print(f'[{path.name}] {seen - before} flows in {time.perf_counter() - t0:.0f}s', flush=True)
    print(f'kept {len(rows)} flows in {time.perf_counter() - start:.0f}s', flush=True)

    stat = np.clip(np.stack([r['stat'] for r in rows]), -10.0, 10.0)
    seq = np.stack([r['seq'] for r in rows])
    flat = seq.reshape(-1, seq.shape[-1])
    scaler = StandardScaler().fit(flat)
    seq = scaler.transform(flat).reshape(seq.shape).astype(np.float32)
    attn = np.stack([r['attn'] for r in rows])
    x_bytes = np.stack([r['x_bytes'] for r in rows])
    bytes_mask = np.stack([r['bytes_mask'] for r in rows])
    order = np.random.default_rng(42).permutation(len(rows))
    cut = int(len(order) * 0.95)

    def pack(index):
        return dict(x_stat=torch.from_numpy(stat[index]), x_seq=torch.from_numpy(seq[index]),
                    attn_mask=torch.from_numpy(attn[index]), x_bytes=torch.from_numpy(x_bytes[index]),
                    bytes_mask=torch.from_numpy(bytes_mask[index]),
                    label=torch.zeros(len(index), dtype=torch.long))

    cache = dict(splits=dict(train=pack(order[:cut]), val=pack(order[cut:])),
                 protocol=dict(dataset='CIC-IDS2017', selection='unlabelled bidirectional flows, pretraining only',
                               source=str(SRC), flows=int(len(rows)),
                               normalization='statistics standardised with the AES128 scaler and winsorised at +-10; '
                                             'sequence standardised with a train-split scaler'))
    torch.save(cache, OUT)
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)', flush=True)


if __name__ == '__main__':
    main()
