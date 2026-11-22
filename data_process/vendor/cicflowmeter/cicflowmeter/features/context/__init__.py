from enum import Enum, auto

from scapy.packet import Packet  # type: ignore


class PacketDirection(Enum):
    FORWARD = auto()
    REVERSE = auto()


def get_packet_flow_key(packet: Packet, direction: PacketDirection) -> tuple:
    if "TCP" in packet:
        protocol = "TCP"
    elif "UDP" in packet:
        protocol = "UDP"
    else:
        raise Exception("Only TCP protocols are supported.")

    if direction == PacketDirection.FORWARD:
        dest_ip = packet["IP"].dst
        src_ip = packet["IP"].src
        src_port = packet[protocol].sport
        dest_port = packet[protocol].dport
        return src_ip, dest_ip, src_port, dest_port
    else:
        dest_ip = packet["IP"].src
        src_ip = packet["IP"].dst
        src_port = packet[protocol].dport
        dest_port = packet[protocol].sport
        return src_ip, dest_ip, src_port, dest_port


