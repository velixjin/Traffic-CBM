import numpy
from scipy import stats as stat  # type: ignore


class PacketLength:
    mean_count = 0
    grand_total = 0

    def __init__(self, flow):
        self.flow = flow

    def get_packet_length(self, packet_direction=None) -> list:
        if packet_direction is not None:
            return [
                len(packet)
                for packet, direction in self.flow.packets
                if direction == packet_direction
            ]
        return [len(packet) for packet, _ in self.flow.packets]

    def get_header_length(self, packet_direction=None) -> list:
        if packet_direction is not None:
            return (
                packet["IP"].ihl * 4
                for packet, direction in self.flow.packets
                if direction == packet_direction
            )
        return (packet["IP"].ihl * 4 for packet, _ in self.flow.packets)

    def get_total_header(self, packet_direction=None) -> int:
        return sum(self.get_header_length(packet_direction))

    def get_min_header(self, packet_direction=None) -> int:
        return min(self.get_header_length(packet_direction))

    def get_max(self, packet_direction=None) -> int:
        try:
            return max(self.get_packet_length(packet_direction))
        except ValueError:
            return 0

    def get_min(self, packet_direction=None) -> int:
        try:
            return min(self.get_packet_length(packet_direction))
        except ValueError:
            return 0

    def get_total(self, packet_direction=None) -> int:
        return sum(self.get_packet_length(packet_direction))

    def get_avg(self, packet_direction=None) -> int:
        count = len(self.get_packet_length(packet_direction))
        return self.get_total(packet_direction) / count if count > 0 else 0

    def first_fifty(self) -> list:
        return self.get_packet_length()[:50]

    def get_var(self, packet_direction=None) -> float:
        var = 0
        if len(self.get_packet_length(packet_direction)) > 0:
            var = numpy.var(self.get_packet_length(packet_direction))
        return var

    def get_std(self, packet_direction=None) -> float:
        return numpy.sqrt(self.get_var(packet_direction))

    def get_mean(self, packet_direction=None) -> float:
        mean = 0
        if len(self.get_packet_length(packet_direction)) > 0:
            mean = numpy.mean(self.get_packet_length(packet_direction))
        return mean

    def get_median(self) -> float:
        return numpy.median(self.get_packet_length())

    def get_mode(self) -> float:
        mode = -1
        if len(self.get_packet_length()) != 0:
            mode = int(stat.mode(self.get_packet_length())[0])
        return mode

    def get_skew(self) -> float:
        mean = self.get_mean()
        median = self.get_median()
        dif = 3 * (mean - median)
        std = self.get_std()
        return (dif / std) if std != 0 else -10

    def get_skew2(self) -> float:
        mean = self.get_mean()
        mode = self.get_mode()
        dif = mean - mode
        std = self.get_std()
        return (dif / std) if std != 0 else -10

    def get_cov(self) -> float:
        return (self.get_std() / self.get_mean()) if self.get_mean() != 0 else -1


