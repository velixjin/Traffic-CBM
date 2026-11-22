from .context import PacketDirection
from .packet_time import PacketTime


class PacketCount:
    def __init__(self, flow):
        self.flow = flow

    def get_total(self, packet_direction=None) -> int:
        if packet_direction is not None:
            return len(
                [
                    packet
                    for packet, direction in self.flow.packets
                    if direction == packet_direction
                ]
            )
        return len(self.flow.packets)

    def get_rate(self, packet_direction=None) -> float:
        duration = PacketTime(self.flow).get_duration()
        return 0 if duration == 0 else self.get_total(packet_direction) / duration

    def get_down_up_ratio(self) -> float:
        forward_size = self.get_total(PacketDirection.FORWARD)
        backward_size = self.get_total(PacketDirection.REVERSE)
        return backward_size / forward_size if forward_size > 0 else 0

    @staticmethod
    def get_payload(packet):
        if "TCP" in packet:
            return packet["TCP"].payload
        elif "UDP" in packet:
            return packet["UDP"].payload
        return 0

    def has_payload(self, packet_direction=None) -> int:
        if packet_direction is not None:
            return len(
                [
                    packet
                    for packet, direction in self.flow.packets
                    if direction == packet_direction
                    and len(self.get_payload(packet)) > 0
                ]
            )
        return len(
            [
                packet
                for packet, _ in self.flow.packets
                if len(self.get_payload(packet)) > 0
            ]
        )


