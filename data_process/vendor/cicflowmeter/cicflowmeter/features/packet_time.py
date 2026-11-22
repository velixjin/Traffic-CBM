from datetime import datetime

import numpy
from scipy import stats as stat  # type: ignore


class PacketTime:
    count = 0

    def __init__(self, flow):
        self.flow = flow
        PacketTime.count += 1
        self.packet_times = None

    def _get_packet_times(self):
        if self.packet_times is not None:
            return self.packet_times
        first_packet_time = self.flow.packets[0][0].time
        packet_times = [
            float(packet.time - first_packet_time) for packet, _ in self.flow.packets
        ]
        return packet_times

    def get_packet_iat(self, packet_direction=None):
        if packet_direction is not None:
            packets = [
                packet
                for packet, direction in self.flow.packets
                if direction == packet_direction
            ]
        else:
            packets = [packet for packet, _ in self.flow.packets]
        return [float(packets[i].time - packets[i - 1].time) for i in range(1, len(packets))]

    def relative_time_list(self):
        relative_time_list = []
        packet_times = self._get_packet_times()
        for index, time in enumerate(packet_times):
            if index == 0:
                relative_time_list.append(0)
            elif index < len(packet_times):
                relative_time_list.append(float(time - packet_times[index - 1]))
            elif index < 50:
                relative_time_list.append(0)
            else:
                break
        return relative_time_list

    def get_timestamp(self):
        time = float(self.flow.packets[0][0].time)
        return datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S")

    def get_duration(self):
        return max(self._get_packet_times())

    def get_var(self):
        return numpy.var(self._get_packet_times())

    def get_std(self):
        return numpy.sqrt(self.get_var())

    def get_mean(self):
        mean = 0
        if self._get_packet_times() != 0:
            mean = numpy.mean(self._get_packet_times())
        return mean

    def get_median(self):
        return numpy.median(self._get_packet_times())

    def get_mode(self):
        mode = -1
        if len(self._get_packet_times()) != 0:
            mode = stat.mode(self._get_packet_times())
            mode = float(mode[0])
        return mode

    def get_skew(self):
        mean = self.get_mean()
        median = self.get_median()
        dif = 3 * (mean - median)
        std = self.get_std()
        return (dif / std) if std != 0 else -10

    def get_skew2(self):
        mean = self.get_mean()
        mode = self.get_mode()
        dif = float(mean) - mode
        std = self.get_std()
        return (dif / float(std)) if std != 0 else -10

    def get_cov(self):
        return (self.get_std() / self.get_mean()) if self.get_mean() != 0 else -1


