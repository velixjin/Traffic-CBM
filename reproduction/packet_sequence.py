"""Link-type-aware sequence extraction used by the aligned runs."""
import numpy as np
from scapy.all import PcapReader, IP, IPv6, TCP, UDP


def extract_correct(path,limit=100):
    rows=[]
    source=None
    last=None
    with PcapReader(str(path)) as reader:
        for packet in reader:
            net=packet.getlayer(IP) or packet.getlayer(IPv6)
            if net is None:
                continue
            if source is None:
                source=net.src
            timestamp=float(packet.time)
            iat=0 if last is None else max(0,timestamp-last)
            last=timestamp
            transport=packet.getlayer(TCP) or packet.getlayer(UDP)
            # Payload excludes L2, IP and transport headers, but includes transport data.
            payload=len(bytes(transport.payload)) if transport is not None else len(bytes(net.payload))
            header=len(bytes(net))-payload
            flags=(int(packet[TCP].flags)&0x3f) if TCP in packet else 0
            window=int(packet[TCP].window) if TCP in packet else 0
            rows.append([len(bytes(packet)),iat,1 if net.src==source else -1,flags,payload,header,window])
            if limit and len(rows)>=limit:
                break
    return np.asarray(rows,dtype=np.float32).reshape(-1,7)


