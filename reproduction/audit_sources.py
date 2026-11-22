"""Read-only provenance/connection/prefix audit and confirmed duplicate verification."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import os
import re
import struct
from scapy.utils import RawPcapReader

ROOT=Path(__file__).resolve().parents[1]
FEATURE_ROOT=Path(os.environ.get('TRAFFIC_CBM_FEATURE_ROOT', ROOT/'data'))
FLOW_ROOT=Path(os.environ.get('TRAFFIC_CBM_FLOW_ROOT', ROOT/'dataset'))


def connection_key(raw,linktype):
    offset={1:14,101:0,228:0,113:16,276:20}.get(linktype)
    if offset is None or len(raw)<offset+20 or raw[offset]>>4!=4:
        return None
    ihl=(raw[offset]&15)*4
    if ihl<20 or len(raw)<offset+ihl+4:
        return None
    proto=raw[offset+9]
    if proto not in [6,17]:
        return None
    a=raw[offset+12:offset+16]+raw[offset+ihl:offset+ihl+2]
    b=raw[offset+16:offset+20]+raw[offset+ihl+2:offset+ihl+4]
    return hashlib.sha256(bytes([proto])+b''.join(sorted([a,b]))).hexdigest()


def inspect(item):
    dataset,split,fid,path=item
    row=dict(dataset=dataset,split=split,flow_id=fid,path=path)
    try:
        h=hashlib.sha256();n=0;timestamp=None;conn=None
        with closing(RawPcapReader(path)) as reader:
            linktype=reader.linktype
            for raw,meta in reader:
                if not n:
                    conn=connection_key(raw,linktype)
                    if hasattr(meta,'sec'):
                        timestamp=float(meta.sec)+float(meta.usec)/(1e9 if reader.nano else 1e6)
                h.update(struct.pack('!I',len(raw)));h.update(raw);n+=1
                if n>=32:
                    break
        basename=Path(path).name
        match=re.match(r'(.+\.pcap)\.(?:TCP|UDP)_',basename)
        row.update(prefix32_sha256=h.hexdigest(),packets_in_prefix=n,connection_hash=conn,
            first_timestamp=timestamp,linktype=linktype,source_capture=match.group(1) if match else None)
    except Exception as error:
        row['error']=str(error)
    return row


def complete_packet_hash(path):
    h=hashlib.sha256();n=0
    with closing(RawPcapReader(path)) as reader:
        for raw,_ in reader:
            h.update(struct.pack('!I',len(raw)));h.update(raw);n+=1
    return h.hexdigest(),n


def main(output,workers):
    output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for data in sorted((FEATURE_ROOT).iterdir()):
        if not (data/'train_ids.txt').exists():
            continue
        for split in ['train','val','test']:
            for fid in (data/f'{split}_ids.txt').read_text().splitlines():
                if fid.strip():
                    jobs.append((data.name,split,fid.strip(),str(FLOW_ROOT/data.name/'flows_filtered'/(fid.strip()+'.pcap'))))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n,row in enumerate(pool.map(inspect,jobs,chunksize=128)):
            rows.append(row)
            if n%20000==0:
                print(f'[sources] {n}/{len(jobs)}',flush=True)
    keys=sorted({k for row in rows for k in row})
    with (output/'flow_provenance.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    good=[r for r in rows if 'error' not in r and r['packets_in_prefix']>=3]
    prefixes=defaultdict(list)
    for row in good:
        prefixes[row['prefix32_sha256']].append(row)
    candidates=[g for g in prefixes.values() if len({(r['dataset'],r['split']) for r in g})>1]
    fullgroups=defaultdict(list)
    for n,group in enumerate(candidates):
        for row in group:
            digest,packets=complete_packet_hash(row['path'])
            fullgroups[digest].append(dict(dataset=row['dataset'],split=row['split'],flow_id=row['flow_id'],packet_count=packets))
        if n%1000==0:
            print(f'[verify duplicate] {n}/{len(candidates)}',flush=True)
    confirmed=[dict(packet_data_sha256=h,flows=g) for h,g in fullgroups.items()
               if len({(r['dataset'],r['split']) for r in g})>1]
    summary=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        take=[r for r in rows if r['dataset']==dataset]
        captures=defaultdict(set);connections=defaultdict(set)
        for row in take:
            if row.get('source_capture'):
                captures[row['source_capture']].add(row['split'])
            if row.get('connection_hash'):
                connections[row['connection_hash']].add(row['split'])
        same_dataset=[g for g in confirmed if len({f['split'] for f in g['flows'] if f['dataset']==dataset})>1]
        test={r['flow_id'] for r in take if r['split']=='test'}
        identical_to_training=set()
        for group in confirmed:
            if any(r['split']=='train' for r in group['flows']):
                identical_to_training.update(r['flow_id'] for r in group['flows'] if r['dataset']==dataset and r['split']=='test')
        summary.append(dict(dataset=dataset,n_flows=len(take),read_errors=sum('error' in r for r in take),
            recovered_capture_groups=len(captures),capture_groups_crossing_splits=sum(len(s)>1 for s in captures.values()),
            connection_groups=len(connections),connection_groups_crossing_splits=sum(len(s)>1 for s in connections.values()),
            confirmed_packet_identical_groups_crossing_splits=len(same_dataset),
            test_flows_identical_to_any_dataset_training_flow=len(identical_to_training),test_flows=len(test)))
    report=dict(summary=summary,confirmed_duplicates=confirmed,comparison=dict(
        capture_groups='Source capture identifiers recovered from flow filenames where available.',
        connection_groups='Matching connection tuples.',
        packet_duplicates='Match the first 32 packets, then compare hashes of all captured packet bytes.'))
    (output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args();main(args.output,args.workers)
