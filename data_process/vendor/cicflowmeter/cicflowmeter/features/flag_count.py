class FlagCount:
    """Extract features related to TCP flags count."""

    def __init__(self, flow):
        self.flow = flow

    def count(self, flag, packet_direction=None) -> bool:
        count = 0
        if packet_direction is not None:
            packets = (
                packet
                for packet, direction in self.flow.packets
                if direction == packet_direction
            )
        else:
            packets = (packet for packet, _ in self.flow.packets)

        for packet in packets:
            if flag[0] in packet.sprintf("%TCP.flags%"):
                count += 1
        return count


