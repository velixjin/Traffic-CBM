"""Fast PCAP parsing that avoids scapy dissection.

Produces minimal packet objects exposing exactly the attributes the vendored
cicflowmeter code reads, so the feature definitions stay untouched.
"""
import struct
from decimal import Decimal

_FLAG_CHARS = ((0x01, 'F'), (0x02, 'S'), (0x04, 'R'), (0x08, 'P'),
               (0x10, 'A'), (0x20, 'U'), (0x40, 'E'), (0x80, 'C'))


class _Payload:
    __slots__ = ('n',)

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _IPLayer:
    __slots__ = ('src', 'dst', 'ihl', 'len', 'ttl')

    def __init__(self, src, dst, ihl, total, ttl):
        self.src, self.dst, self.ihl, self.len, self.ttl = src, dst, ihl, total, ttl


class _TransportLayer:
    __slots__ = ('sport', 'dport', 'window', 'payload', 'flags')

    def __init__(self, sport, dport, window, payload_len, flags_byte):
        self.sport, self.dport, self.window = sport, dport, window
        self.payload = _Payload(payload_len)
        self.flags = flags_byte


class MiniPacket:
    __slots__ = ('time', 'proto', 'frame_len', 'ip', 'tcp', 'udp', '_flags_text', 'raw')

    def __init__(self, time, proto, frame_len, ip, tcp, udp, flags_text, raw=b''):
        self.time = time
        self.proto = proto
        self.frame_len = frame_len
        self.ip = ip
        self.tcp = tcp
        self.udp = udp
        self._flags_text = flags_text
        self.raw = raw

    def __contains__(self, key):
        name = key if isinstance(key, str) else getattr(key, '__name__', '')
        if name == 'TCP':
            return self.tcp is not None
        if name == 'UDP':
            return self.udp is not None
        if name in ('IP', 'IPv4'):
            return self.ip is not None
        return False

    def __getitem__(self, key):
        name = key if isinstance(key, str) else getattr(key, '__name__', '')
        if name in ('IP', 'IPv4'):
            return self.ip
        if name == 'TCP':
            return self.tcp
        if name == 'UDP':
            return self.udp
        raise KeyError(key)

    def __len__(self):
        return self.frame_len

    def sprintf(self, fmt):
        if fmt == '%TCP.flags%':
            return self._flags_text
        return ''


def _ipv4(raw, off, frame_len, time, keep_raw=False):
    ihl_words = raw[off] & 0x0F          # scapy stores ihl in 32-bit words
    ihl = ihl_words * 4
    if ihl < 20:
        return None
    proto = raw[off + 9]
    total = struct.unpack_from('>H', raw, off + 2)[0]
    src = '%d.%d.%d.%d' % tuple(raw[off + 12:off + 16])
    dst = '%d.%d.%d.%d' % tuple(raw[off + 16:off + 20])
    ttl = raw[off + 8]
    ip = _IPLayer(src, dst, ihl_words, total, ttl)
    l4 = off + ihl
    tcp = udp = None
    flags_text = ''
    if proto == 6 and len(raw) >= l4 + 20:
        sport, dport = struct.unpack_from('>HH', raw, l4)
        seq, ack = struct.unpack_from('>II', raw, l4 + 4)
        dataofs = (raw[l4 + 12] >> 4) * 4
        flags_byte = raw[l4 + 13]
        window = struct.unpack_from('>H', raw, l4 + 14)[0]
        payload_len = max(0, total - ihl - dataofs)
        tcp = _TransportLayer(sport, dport, window, payload_len, flags_byte)
        flags_text = ''.join(ch for bit, ch in _FLAG_CHARS if flags_byte & bit)
    elif proto == 17 and len(raw) >= l4 + 8:
        sport, dport = struct.unpack_from('>HH', raw, l4)
        udp_len = struct.unpack_from('>H', raw, l4 + 4)[0]
        payload_len = max(0, udp_len - 8)
        udp = _TransportLayer(sport, dport, 0, payload_len, 0)
    else:
        return None
    return MiniPacket(time, proto, frame_len, ip, tcp, udp, flags_text, raw)


def read_pcap(path, limit=200):
    """limit=None reads every IPv4 TCP/UDP packet in the file."""
    """Return IPv4 TCP/UDP packets in capture order (scapy-equivalent subset)."""
    packets = []
    with open(path, 'rb') as stream:
        header = stream.read(24)
        if len(header) < 24:
            return packets
        magic = header[:4]
        if magic == b'\xd4\xc3\xb2\xa1':
            endian, nano = '<', False
        elif magic == b'\xa1\xb2\xc3\xd4':
            endian, nano = '>', False
        elif magic == b'\x4d\x3c\xb2\xa1':
            endian, nano = '<', True
        else:
            endian, nano = '>', True
        linktype = struct.unpack(endian + 'I', header[20:24])[0]
        while limit is None or len(packets) < limit:
            record = stream.read(16)
            if len(record) < 16:
                break
            sec, frac, incl, orig = struct.unpack(endian + 'IIII', record)
            raw = stream.read(incl)
            if len(raw) < incl:
                break
            # scapy stores pcap timestamps as exact Decimals; matching that keeps
            # inter-arrival times bit-identical.
            time = Decimal(sec) + Decimal(frac).scaleb(-9 if nano else -6)
            frame_len = orig if orig else incl
            if linktype == 101:
                off = 0
                if len(raw) >= 1 and (raw[0] >> 4) == 4:
                    packet = _ipv4(raw, off, frame_len, time)
                    if packet is not None:
                        packets.append(packet)
            elif linktype == 113 and len(raw) >= 16:
                if struct.unpack_from('>H', raw, 14)[0] == 0x0800:
                    packet = _ipv4(raw, 16, frame_len, time)
                    if packet is not None:
                        packets.append(packet)
            elif len(raw) >= 14:
                ethertype = struct.unpack_from('>H', raw, 12)[0]
                off = 14
                if ethertype == 0x8100 and len(raw) >= 18:
                    ethertype = struct.unpack_from('>H', raw, 16)[0]
                    off = 18
                if ethertype == 0x0800:
                    packet = _ipv4(raw, off, frame_len, time)
                    if packet is not None:
                        packets.append(packet)
    return packets


def iter_pcap(path, limit=None):
    """Stream IPv4 TCP/UDP packets one at a time (memory-safe for multi-GB captures)."""
    with open(path, 'rb') as stream:
        header = stream.read(24)
        if len(header) < 24:
            return
        magic = header[:4]
        if magic == b'\xd4\xc3\xb2\xa1':
            endian, nano = '<', False
        elif magic == b'\xa1\xb2\xc3\xd4':
            endian, nano = '>', False
        elif magic == b'\x4d\x3c\xb2\xa1':
            endian, nano = '<', True
        else:
            endian, nano = '>', True
        count = 0
        while limit is None or count < limit:
            record = stream.read(16)
            if len(record) < 16:
                break
            sec, frac, incl, orig = struct.unpack(endian + 'IIII', record)
            raw = stream.read(incl)
            if len(raw) < incl:
                break
            time = Decimal(sec) + Decimal(frac).scaleb(-9 if nano else -6)
            frame_len = orig if orig else incl
            if linktype == 101:
                if len(raw) >= 1 and (raw[0] >> 4) == 4:
                    packet = _ipv4(raw, 0, frame_len, time)
                    if packet is not None:
                        count += 1
                        yield packet
            elif linktype == 113 and len(raw) >= 16:
                if struct.unpack_from('>H', raw, 14)[0] == 0x0800:
                    packet = _ipv4(raw, 16, frame_len, time)
                    if packet is not None:
                        count += 1
                        yield packet
            elif len(raw) >= 14:
                ethertype = struct.unpack_from('>H', raw, 12)[0]
                off = 14
                if ethertype == 0x8100 and len(raw) >= 18:
                    ethertype = struct.unpack_from('>H', raw, 16)[0]
                    off = 18
                if ethertype == 0x0800:
                    packet = _ipv4(raw, off, frame_len, time)
                    if packet is not None:
                        count += 1
                        yield packet
