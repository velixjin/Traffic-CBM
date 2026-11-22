from scapy.layers.inet import IP  # type: ignore

from .context import PacketDirection
from .packet_time import PacketTime


class FlowBytes:
    def __init__(self, flow):
        self.flow = flow

    def get_bytes(self) -> int:
        return sum(len(packet) for packet, _ in self.flow.packets)

    def get_rate(self) -> float:
        duration = PacketTime(self.flow).get_duration()
        return 0 if duration == 0 else self.get_bytes() / duration

    def get_bytes_sent(self) -> int:
        return sum(
            len(packet)
            for packet, direction in self.flow.packets
            if direction == PacketDirection.FORWARD
        )

    def get_sent_rate(self) -> float:
        sent = self.get_bytes_sent()
        duration = PacketTime(self.flow).get_duration()
        return -1 if duration == 0 else sent / duration

    def get_bytes_received(self) -> int:
        return sum(
            len(packet)
            for packet, direction in self.flow.packets
            if direction == PacketDirection.REVERSE
        )

    def get_received_rate(self) -> float:
        received = self.get_bytes_received()
        duration = PacketTime(self.flow).get_duration()
        return -1 if duration == 0 else received / duration

    def _header_size(self, packet):
        if IP in packet:
            ihl = packet[IP].ihl
            return 20 if ihl is None else ihl * 4
        return 0

    def get_forward_header_bytes(self) -> int:
        return sum(
            self._header_size(packet)
            for packet, direction in self.flow.packets
            if direction == PacketDirection.FORWARD
        )

    def get_reverse_header_bytes(self) -> int:
        if not self.flow.packets:
            return 0
        return sum(
            self._header_size(packet)
            for packet, direction in self.flow.packets
            if direction == PacketDirection.REVERSE
        )

    def get_min_forward_header_bytes(self) -> int:
        if not self.flow.packets:
            return 0
        vals = [
            self._header_size(packet)
            for packet, direction in self.flow.packets
            if direction == PacketDirection.FORWARD
        ]
        return min(vals) if vals else 0

    def get_forward_rate(self) -> float:
        forward = self.get_forward_header_bytes()
        duration = PacketTime(self.flow).get_duration()
        return forward / duration if duration > 0 else -1

    def get_reverse_rate(self) -> float:
        reverse = self.get_reverse_header_bytes()
        duration = PacketTime(self.flow).get_duration()
        return -1 if duration == 0 else reverse / duration

    def get_header_in_out_ratio(self) -> float:
        reverse_header_bytes = self.get_reverse_header_bytes()
        forward_header_bytes = self.get_forward_header_bytes()
        if reverse_header_bytes != 0:
            return forward_header_bytes / reverse_header_bytes
        return -1

    def get_initial_ttl(self) -> int:
        return [packet["IP"].ttl for packet, _ in self.flow.packets][0]

    def get_bytes_per_bulk(self, direction: PacketDirection) -> float:
        if direction is PacketDirection.FORWARD and self.flow.forward_bulk_count != 0:
            return self.flow.forward_bulk_size / self.flow.forward_bulk_count
        if direction is PacketDirection.REVERSE and self.flow.backward_bulk_count != 0:
            return self.flow.backward_bulk_size / self.flow.backward_bulk_count
        return 0

    def get_packets_per_bulk(self, direction: PacketDirection) -> float:
        if direction is PacketDirection.FORWARD and self.flow.forward_bulk_count != 0:
            return self.flow.forward_bulk_packet_count / self.flow.forward_bulk_count
        if direction is PacketDirection.REVERSE and self.flow.backward_bulk_count != 0:
            return self.flow.backward_bulk_packet_count / self.flow.backward_bulk_count
        return 0

    def get_bulk_rate(self, direction: PacketDirection) -> float:
        if direction is PacketDirection.FORWARD and self.flow.forward_bulk_duration != 0:
            return self.flow.forward_bulk_size / self.flow.forward_bulk_duration
        if direction is PacketDirection.REVERSE and self.flow.backward_bulk_duration != 0:
            return self.flow.backward_bulk_size / self.flow.backward_bulk_duration
        return 0


