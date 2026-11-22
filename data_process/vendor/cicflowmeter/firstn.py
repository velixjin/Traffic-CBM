#!/usr/bin/env python3
"""
Pure Python API for extracting first-N-packet flow features (vendored).

We vendor a minimal subset of cicflowmeter (MIT) to avoid external runtime deps.
Layout:
  this_dir/
    firstn.py
    cicflowmeter/  # minimal package (constants, flow_session, flow, features, writer, utils)
"""
from __future__ import annotations

import os
import sys
from typing import Optional, List

# Make vendored package importable (add directory that CONTAINS the 'cicflowmeter' package)
_PKG_PARENT = os.path.dirname(__file__)  # .../vendor/cicflowmeter
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from scapy.utils import rdpcap  # type: ignore
from scapy.layers.inet import IP, TCP, UDP  # type: ignore
from cicflowmeter.flow_session import FlowSession  # type: ignore
from cicflowmeter.features.context import PacketDirection, get_packet_flow_key  # type: ignore
from cicflowmeter.flow import Flow  # type: ignore
from cicflowmeter.constants import EXPIRED_UPDATE, PACKETS_PER_GC  # type: ignore


class FirstNFlowSession(FlowSession):
    """Flow session that collects only the first N packets per bidirectional flow.
    If output_mode is None, rows are accumulated in-memory under self.rows.
    """

    def __init__(self, max_packets_per_flow: int = 5, collect_in_memory: bool = False,
                 min_packets_to_keep: int = 1, *args, **kwargs):
        self.max_packets_per_flow = max_packets_per_flow
        self.min_packets_to_keep = min_packets_to_keep
        # set flag before super so we can adjust writer after
        self._collect_in_memory = collect_in_memory
        super().__init__(*args, **kwargs)
        # enable in-memory collection if requested
        if self._collect_in_memory:
            self.rows: List[dict] = []
            # ensure no writer is used
            if hasattr(self, 'output_writer'):
                try:
                    del self.output_writer
                except Exception:
                    pass

    def process(self, pkt):
        self.logger.debug(f"Packet {self.packets_count}: {pkt}")
        count = 0
        direction = PacketDirection.FORWARD

        if "TCP" not in pkt and "UDP" not in pkt:
            return None

        try:
            packet_flow_key = get_packet_flow_key(pkt, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return None

        self.packets_count += 1

        if flow is None:
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(pkt, direction)
            flow = self.flows.get((packet_flow_key, count))

        if flow is None:
            direction = PacketDirection.FORWARD
            flow = Flow(pkt, direction)
            packet_flow_key = get_packet_flow_key(pkt, direction)
            self.flows[(packet_flow_key, count)] = flow

        elif len(flow.packets) >= self.max_packets_per_flow:
            self.logger.debug(
                f"Flow has reached max packets ({self.max_packets_per_flow}), skipping packet"
            )
            return None

        elif (pkt.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            expired = EXPIRED_UPDATE
            while (pkt.time - flow.latest_timestamp) > expired:
                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))
                if flow is None:
                    flow = Flow(pkt, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break

        elif hasattr(pkt, "flags") and "F" in pkt.flags:
            flow.add_packet(pkt, direction)
            self.garbage_collect(pkt.time)
            return None

        flow.add_packet(pkt, direction)
        self.logger.debug(f"Added packet to flow, total packets: {len(flow.packets)}")

        if len(flow.packets) >= self.max_packets_per_flow:
            self.logger.debug("Flow reached max packets, collecting immediately")
            self.garbage_collect(pkt.time)
        elif self.packets_count % PACKETS_PER_GC == 0 or flow.duration > 120:
            self.garbage_collect(pkt.time)
        return None

    def garbage_collect(self, latest_time) -> None:
        for k in list(self.flows.keys()):
            flow = self.flows.get(k)
            if not flow or (
                latest_time is not None
                and latest_time - flow.latest_timestamp < EXPIRED_UPDATE
                and flow.duration < 90
            ):
                continue
            # drop short flows (< min_packets_to_keep)
            if len(flow.packets) < self.min_packets_to_keep:
                del self.flows[k]
                continue
            row = flow.get_data(self.fields)
            if self._collect_in_memory:
                self.rows.append(row)
            else:
                self.output_writer.write(row)
            del self.flows[k]
            self.logger.debug(f"Flow Collected! Remain Flows = {len(self.flows)}")

    def flush_flows(self):
        for flow in list(self.flows.values()):
            if len(flow.packets) < self.min_packets_to_keep:
                continue
            row = flow.get_data(self.fields)
            if self._collect_in_memory:
                self.rows.append(row)
            else:
                self.output_writer.write(row)
        self.flows.clear()
        if not self._collect_in_memory:
            del self.output_writer


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_firstn_to_csv(input_file: str, output_csv: str, n_packets: int = 5,
                       fields: Optional[List[str]] = None, verbose: bool = False) -> str:
    """Extract CICFlowMeter features using rdpcap (no AsyncSniffer threads).
    Avoids fork+thread deadlock when used with ProcessPoolExecutor."""
    _ensure_dir(os.path.dirname(output_csv))
    session = FirstNFlowSession(
        max_packets_per_flow=n_packets,
        output_mode="csv",
        output=output_csv,
        fields=fields,
        verbose=verbose,
    )
    try:
        pkts = rdpcap(input_file)
    except Exception:
        return output_csv
    for pkt in pkts:
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            continue
        session.process(pkt)
    session.flush_flows()
    return output_csv


