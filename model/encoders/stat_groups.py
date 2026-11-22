#!/usr/bin/env python3
from typing import List


GROUP_NAMES = [
    "packet_count",
    "packet_length",
    "flow_bytes",
    "packet_time",
    "flag_count",
    "header_window",
]

PACKET_COUNT_KEYS = {
    "tot_fwd_pkts", "tot_bwd_pkts",
    "fwd_pkts_s", "bwd_pkts_s",
    "subflow_fwd_pkts", "subflow_bwd_pkts",
    "fwd_act_data_pkts",
}

PACKET_LENGTH_KEYS = {
    "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "pkt_len_max", "pkt_len_min", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "pkt_size_avg", "fwd_seg_size_avg", "bwd_seg_size_avg",
    "totlen_fwd_pkts", "totlen_bwd_pkts",
}

FLOW_BYTES_KEYS = {
    "flow_byts_s", "flow_pkts_s", "down_up_ratio",
    "fwd_byts_b_avg", "bwd_byts_b_avg",
    "fwd_blk_rate_avg", "bwd_blk_rate_avg",
    "subflow_fwd_byts", "subflow_bwd_byts",
}

PACKET_TIME_KEYS = {
    "flow_duration",
    "flow_iat_mean", "flow_iat_max", "flow_iat_min", "flow_iat_std",
    "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_mean", "fwd_iat_std",
    "bwd_iat_tot", "bwd_iat_max", "bwd_iat_min", "bwd_iat_mean", "bwd_iat_std",
    "active_max", "active_min", "active_mean", "active_std",
    "idle_max", "idle_min", "idle_mean", "idle_std",
}

FLAG_COUNT_KEYS = {
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt",
    "ack_flag_cnt", "urg_flag_cnt", "ece_flag_cnt", "cwr_flag_count",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
}

HEADER_WINDOW_KEYS = {
    "fwd_header_len", "bwd_header_len",
    "init_fwd_win_byts", "init_bwd_win_byts",
}


def build_cic_groups(header: List[str], stat_cols_idx: List[int]) -> List[List[int]]:
    """Map CICFlowMeter columns to six statistical concept groups.

    The groups are returned in this order:
    [packet_count, packet_length, flow_bytes, packet_time, flag_count, header_window]
    """

    # Map column names to positions in the filtered feature vector.
    # Iterate over cached feature columns to preserve their order.
    # The order within a group determines the input order of its linear layer.
    # A changed order can load a checkpoint successfully while misaligning its weights.
    feat_names = [header[idx] for idx in stat_cols_idx]
    name_to_pos = {name: pos for pos, name in enumerate(feat_names)}

    def collect(keys: set) -> List[int]:
        return [name_to_pos[name] for name in feat_names if name in keys]

    g_packet_count = collect(PACKET_COUNT_KEYS)
    g_packet_length = collect(PACKET_LENGTH_KEYS)
    g_flow_bytes = collect(FLOW_BYTES_KEYS)
    g_packet_time = collect(PACKET_TIME_KEYS)
    g_flag_count = collect(FLAG_COUNT_KEYS)
    g_header_window = collect(HEADER_WINDOW_KEYS)

    groups = [g_packet_count, g_packet_length, g_flow_bytes, g_packet_time, g_flag_count, g_header_window]

    return groups

