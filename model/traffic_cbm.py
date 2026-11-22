#!/usr/bin/env python3
"""
traffic_cbm.py - Traffic Concept Bottleneck Model.

Fusion modes
------------
"6+1+1"  (default)
    z = [z_stat(6), z_seq(1), z_bytes(1)] ∈ R^8
    ŷ = head(z)

"6/6/6"
    c_stat, c_seq, c_bytes ∈ R^6  (shared concept space)
    c_fused[k] = Σ_i α[k,i] · c_i[k],   α[k] = softmax(w[k])
    ŷ = head(c_fused)

"6+4"
    c_stat  [B, 6]  — StatConceptEncoder, Level-3 (grouped MLP)
    c_seq   [B, 4]  — SeqConceptEncoder,  Level-3 (4 physically-isolated Transformers)
    c_final = cat(c_stat, c_seq)   [B, 10]
    ŷ = head(c_final)
    Pure concept backbone, no bytes branch. Intended as Stage-1 of two-stage training.

"6+4|b"
    c_stat  [B, 6]  — StatConceptEncoder, Level-3 (grouped MLP)
    c_seq   [B, 4]  — SeqConceptEncoder,  Level-3 (4 physically-isolated Transformers)
    g_bytes [B, 10] — BytesGateEncoder,   Level-2 (conditional gating)
    c_final = cat(c_stat, c_seq) ⊙ sigmoid(g_bytes)   [B, 10]
    ŷ = head(c_final)

"6+4+5"
    c_stat  [B, 6]  — StatConceptEncoder
    c_seq   [B, 4]  — SeqConceptEncoder
    c_bytes [B, 5]  — BytesPacketEncoder (per-packet byte concepts)
    c_final = cat(c_stat, c_seq, c_bytes)   [B, 15]
    ŷ = head(c_final)

"6+4+5+3"
    c_stat  [B, 6]  — StatConceptEncoder
    c_seq   [B, 4]  — SeqConceptEncoder
    c_bytes [B, 8]  — BytesPacketCrossEncoder (5 packet + 3 cross-packet)
    c_final = cat(c_stat, c_seq, c_bytes)   [B, 18]
    ŷ = head(c_final)

"6+4+3"
    c_stat  [B, 6]  — StatConceptEncoder
    c_seq   [B, 4]  — SeqConceptEncoder
    c_bytes [B, 3]  — BytesPacketCrossEncoder (cross_only: 3 cross-packet only)
    c_final = cat(c_stat, c_seq, c_bytes)   [B, 13]
    ŷ = head(c_final)

Usage:
    from model.traffic_cbm import TrafficCBM, build_model
"""
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .encoders.stat_encoder import StatConceptEncoder
from .encoders.sequence_encoder import SequenceEncoder, SeqConceptEncoder
from .encoders.bytes_encoder import BytesEncoder, BytesCNNEncoder, BytesGateEncoder, BytesPacketEncoder, BytesPacketCrossEncoder
try:
    from .vendor.taylor_net import TaylorNet
except Exception:
    TaylorNet = None


# ---------------------------------------------------------------------------
# Concept-level fusion layer  (used only in "6/6/6" mode)
# ---------------------------------------------------------------------------

class ConceptFusionLayer(nn.Module):
    """Per-concept learnable weighted average over N branches.

    For each concept k:
        α[k] = softmax(w[k])          ∈ R^{num_branches}
        c_fused[k] = Σ_i α[k,i] · c_i[k]

    Weights are input-independent (global importance per concept-branch pair).

    Parameters
    ----------
    num_concepts  : K — shared concept dimension across all branches
    num_branches  : number of active encoder branches (2 or 3)
    """

    def __init__(self, num_concepts: int, num_branches: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_branches = num_branches
        # [K, num_branches]; initialise uniformly (logits=0 → equal weights)
        self.attn_logits = nn.Parameter(torch.zeros(num_concepts, num_branches))

    def forward(
        self, branches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        branches : list of [B, K] tensors, one per active branch
        returns  : c_fused [B, K],  alpha [K, num_branches]
        """
        assert len(branches) == self.num_branches
        c_stack = torch.stack(branches, dim=2)          # [B, K, num_branches]
        alpha   = torch.softmax(self.attn_logits, dim=1)  # [K, num_branches]
        c_fused = (c_stack * alpha.unsqueeze(0)).sum(dim=2)  # [B, K]
        return c_fused, alpha


# ---------------------------------------------------------------------------
# GA2M-style head
# ---------------------------------------------------------------------------

class GA2MHead(nn.Module):
    """Generalized additive head with pairwise interactions.

    logits = bias + sum_i f_i(c_i) + sum_{i<j} f_{ij}(c_i, c_j)
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden: int = 16,
        pair_hidden: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.bias = nn.Parameter(torch.zeros(self.num_outputs))

        self.main_effects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden, self.num_outputs),
            )
            for _ in range(self.num_inputs)
        ])

        self.pair_indices = list(combinations(range(self.num_inputs), 2))
        self.pair_effects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, pair_hidden),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(pair_hidden, self.num_outputs),
            )
            for _ in self.pair_indices
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.bias.unsqueeze(0).expand(x.size(0), -1)
        for i, effect in enumerate(self.main_effects):
            logits = logits + effect(x[:, i : i + 1])
        for (i, j), effect in zip(self.pair_indices, self.pair_effects):
            logits = logits + effect(x[:, [i, j]])
        return logits


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TrafficCBM(nn.Module):
    """Multi-modal traffic classifier with concept bottleneck.

    Parameters
    ----------
    stat_encoder  : StatConceptEncoder or None
    seq_encoder   : SequenceEncoder or None
    bytes_encoder : BytesEncoder / BytesCNNEncoder or None
    num_classes   : number of output classes
    head          : "taylor" | "mlp" | "ga2m"
    fusion_mode   : "6+1+1" | "6/6/6"
    head_kwargs   : extra kwargs forwarded to the head
    """

    def __init__(
        self,
        stat_encoder:  Optional[StatConceptEncoder] = None,
        seq_encoder:   Optional[SequenceEncoder]    = None,
        bytes_encoder: Optional[nn.Module]          = None,
        num_classes:   int = 10,
        head:          str = "mlp",
        fusion_mode:   str = "6+1+1",
        head_kwargs:   Optional[Dict] = None,
    ):
        super().__init__()
        assert fusion_mode in ("6+1+1", "6/6/6", "6+4", "6+4|b", "6+4+5", "6+4+5+3", "6+4+3"), \
            f"fusion_mode must be '6+1+1', '6/6/6', '6+4', '6+4|b', '6+4+5', '6+4+5+3', or '6+4+3', got '{fusion_mode}'"
        head_kwargs = head_kwargs or {}

        self.stat_encoder  = stat_encoder
        self.seq_encoder   = seq_encoder
        self.bytes_encoder = bytes_encoder
        self.fusion_mode   = fusion_mode

        def _sum_dims(*items: Tuple[Optional[nn.Module], str, int]):
            total = 0
            for module, attr, fallback in items:
                if module is not None:
                    total += int(getattr(module, attr, fallback))
            return total

        # ---- fused dimension & optional fusion/gate layers ----
        if fusion_mode == "6/6/6":
            assert stat_encoder is not None, \
                "6/6/6 mode requires stat_encoder (defines shared concept dim)"
            num_concepts = stat_encoder.num_concepts
            active = [e for e in (stat_encoder, seq_encoder, bytes_encoder)
                      if e is not None]
            self.concept_fusion = ConceptFusionLayer(num_concepts, len(active))
            fused_dim = num_concepts

        elif fusion_mode == "6+4":
            if seq_encoder is not None:
                assert isinstance(seq_encoder, SeqConceptEncoder), \
                    "6+4: seq_encoder must be SeqConceptEncoder"
            self.concept_fusion = None
            fused_dim = _sum_dims(
                (stat_encoder, "num_concepts", 0),
                (seq_encoder, "num_concepts", 0),
            )

        elif fusion_mode == "6+4|b":
            assert stat_encoder  is not None, "6+4|b requires stat_encoder"
            assert seq_encoder   is not None, "6+4|b requires seq_encoder (SeqConceptEncoder)"
            assert bytes_encoder is not None, "6+4|b requires bytes_encoder (BytesGateEncoder)"
            assert isinstance(seq_encoder,   SeqConceptEncoder), \
                "6+4|b: seq_encoder must be SeqConceptEncoder"
            assert isinstance(bytes_encoder, BytesGateEncoder), \
                "6+4|b: bytes_encoder must be BytesGateEncoder"
            self.concept_fusion = None
            fused_dim = stat_encoder.num_concepts + seq_encoder.num_concepts   # 6 + 4 = 10
            # auxiliary head for bytes gate (helps prevent gate collapse)
            self.bytes_aux_head = nn.Linear(fused_dim, head_kwargs.get("num_classes", 2))

        elif fusion_mode == "6+4+5":
            if seq_encoder is not None:
                assert isinstance(seq_encoder, SeqConceptEncoder), \
                    "6+4+5: seq_encoder must be SeqConceptEncoder"
            if bytes_encoder is not None:
                assert isinstance(bytes_encoder, BytesPacketEncoder), \
                    "6+4+5: bytes_encoder must be BytesPacketEncoder"
            self.concept_fusion = None
            fused_dim = _sum_dims(
                (stat_encoder, "num_concepts", 0),
                (seq_encoder, "num_concepts", 0),
                (bytes_encoder, "out_dim", 0),
            )

        elif fusion_mode == "6+4+5+3":
            if seq_encoder is not None:
                assert isinstance(seq_encoder, SeqConceptEncoder), \
                    "6+4+5+3: seq_encoder must be SeqConceptEncoder"
            if bytes_encoder is not None:
                assert isinstance(bytes_encoder, BytesPacketCrossEncoder), \
                    "6+4+5+3: bytes_encoder must be BytesPacketCrossEncoder"
            self.concept_fusion = None
            fused_dim = _sum_dims(
                (stat_encoder, "num_concepts", 0),
                (seq_encoder, "num_concepts", 0),
                (bytes_encoder, "out_dim", 0),
            )

        elif fusion_mode == "6+4+3":
            if seq_encoder is not None:
                assert isinstance(seq_encoder, SeqConceptEncoder), \
                    "6+4+3: seq_encoder must be SeqConceptEncoder"
            if bytes_encoder is not None:
                assert isinstance(bytes_encoder, BytesPacketCrossEncoder), \
                    "6+4+3: bytes_encoder must be BytesPacketCrossEncoder"
            self.concept_fusion = None
            fused_dim = _sum_dims(
                (stat_encoder, "num_concepts", 0),
                (seq_encoder, "num_concepts", 0),
                (bytes_encoder, "out_dim", 0),
            )

        else:  # "6+1+1"
            self.concept_fusion = None
            fused_dim = 0
            if stat_encoder  is not None:
                fused_dim += int(getattr(stat_encoder,  "num_concepts", 0))
            if seq_encoder   is not None:
                fused_dim += int(getattr(seq_encoder,   "out_dim", 1))
            if bytes_encoder is not None:
                fused_dim += int(getattr(bytes_encoder, "out_dim", 1))

        assert fused_dim > 0, "At least one encoder must be provided."
        self.fused_dim = fused_dim

        # ---- classification head ----
        self._use_taylor = (head == "taylor")
        if self._use_taylor:
            if TaylorNet is None:
                raise ImportError("head='taylor' requires optional dependency 'tensorly'")
            taylor_head_kwargs = dict(head_kwargs)
            taylor_head_kwargs.pop("num_classes", None)
            self.head = TaylorNet(
                num_inputs=fused_dim,
                num_outputs=num_classes,
                encode_concepts=False,
                **taylor_head_kwargs,
            )
        elif head == "ga2m":
            self.head = GA2MHead(
                num_inputs=fused_dim,
                num_outputs=num_classes,
                hidden=int(head_kwargs.get("ga2m_hidden", head_kwargs.get("hidden", 16))),
                pair_hidden=int(head_kwargs.get("ga2m_pair_hidden", 16)),
                dropout=float(head_kwargs.get("dropout", 0.0)),
            )
        else:
            hidden = int(head_kwargs.get("hidden", max(64, fused_dim * 2)))
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden),
                nn.GELU(),
                nn.Dropout(p=float(head_kwargs.get("dropout", 0.1))),
                nn.Linear(hidden, num_classes),
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        x_stat:     Optional[torch.Tensor] = None,
        x_seq:      Optional[torch.Tensor] = None,
        attn_mask:  Optional[torch.Tensor] = None,
        x_bytes:    Optional[torch.Tensor] = None,
        bytes_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        logits  : [B, num_classes]
        extras  : dict with per-branch outputs and fusion info (see below)

        6+1+1 extras keys : z_stat, z_seq, z_bytes, fused
        6/6/6 extras keys : c_stat, c_seq, c_bytes, alpha, c_fused
        6+4   extras keys : c_stat, c_seq, c_final
        6+4|b extras keys : c_stat, c_seq, g_bytes, c_final
        6+4+5 extras keys : c_stat, c_seq, c_bytes, c_final
        """
        z_stat  = self.stat_encoder(x_stat)          if self.stat_encoder  is not None else None
        z_seq   = self.seq_encoder(x_seq, attn_mask) if self.seq_encoder   is not None else None

        if self.fusion_mode == "6/6/6":
            z_bytes = self.bytes_encoder(x_bytes, bytes_mask)
            active  = [z for z in (z_stat, z_seq, z_bytes) if z is not None]
            c_fused, alpha = self.concept_fusion(active)

            logits = self.head(c_fused)[0] if self._use_taylor else self.head(c_fused)

            extras: Dict[str, torch.Tensor] = {"c_fused": c_fused, "alpha": alpha}
            if z_stat  is not None: extras["c_stat"]  = z_stat
            if z_seq   is not None: extras["c_seq"]   = z_seq
            if z_bytes is not None: extras["c_bytes"] = z_bytes

        elif self.fusion_mode == "6+4":
            c_parts = [z for z in (z_stat, z_seq) if z is not None]
            c_final = torch.cat(c_parts, dim=1) if len(c_parts) > 1 else c_parts[0]
            logits  = self.head(c_final)[0] if self._use_taylor else self.head(c_final)
            extras  = {"c_final": c_final}
            if z_stat is not None: extras["c_stat"] = z_stat
            if z_seq  is not None: extras["c_seq"] = z_seq

        elif self.fusion_mode == "6+4|b":
            # c_stat [B,6] + c_seq [B,4] → c_cond [B,10]  (Level-3 named concepts)
            c_cond  = torch.cat([z_stat, z_seq], dim=1)             # [B, 10]
            # BytesGateEncoder conditioned on c_cond → gate logits [B, 10]
            g_bytes = self.bytes_encoder(x_bytes, bytes_mask, c_cond.detach())
            # element-wise gating: each concept modulated by byte-level confidence
            c_final = c_cond * torch.sigmoid(g_bytes)               # [B, 10]

            logits = self.head(c_final)[0] if self._use_taylor else self.head(c_final)

            extras = {
                "c_stat":  z_stat,
                "c_seq":   z_seq,
                "g_bytes": g_bytes,
                "c_final": c_final,
            }

        elif self.fusion_mode == "6+4+5":
            c_bytes = self.bytes_encoder(x_bytes, bytes_mask) if self.bytes_encoder is not None else None
            c_parts = [z for z in (z_stat, z_seq, c_bytes) if z is not None]
            c_final = torch.cat(c_parts, dim=1) if len(c_parts) > 1 else c_parts[0]

            logits = self.head(c_final)[0] if self._use_taylor else self.head(c_final)

            extras = {"c_final": c_final}
            if z_stat  is not None: extras["c_stat"] = z_stat
            if z_seq   is not None: extras["c_seq"] = z_seq
            if c_bytes is not None: extras["c_bytes"] = c_bytes

        elif self.fusion_mode == "6+4+5+3":
            c_bytes = self.bytes_encoder(x_bytes, bytes_mask) if self.bytes_encoder is not None else None
            c_parts = [z for z in (z_stat, z_seq, c_bytes) if z is not None]
            c_final = torch.cat(c_parts, dim=1) if len(c_parts) > 1 else c_parts[0]

            logits = self.head(c_final)[0] if self._use_taylor else self.head(c_final)

            extras = {"c_final": c_final}
            if z_stat  is not None: extras["c_stat"] = z_stat
            if z_seq   is not None: extras["c_seq"] = z_seq
            if c_bytes is not None: extras["c_bytes"] = c_bytes

        elif self.fusion_mode == "6+4+3":
            c_bytes = self.bytes_encoder(x_bytes, bytes_mask) if self.bytes_encoder is not None else None
            c_parts = [z for z in (z_stat, z_seq, c_bytes) if z is not None]
            c_final = torch.cat(c_parts, dim=1) if len(c_parts) > 1 else c_parts[0]

            logits = self.head(c_final)[0] if self._use_taylor else self.head(c_final)

            extras = {"c_final": c_final}
            if z_stat  is not None: extras["c_stat"] = z_stat
            if z_seq   is not None: extras["c_seq"] = z_seq
            if c_bytes is not None: extras["c_bytes"] = c_bytes

        else:  # "6+1+1"
            z_bytes = self.bytes_encoder(x_bytes, bytes_mask) if self.bytes_encoder is not None else None
            z_list  = [z for z in (z_stat, z_seq, z_bytes) if z is not None]
            fused   = torch.cat(z_list, dim=1) if len(z_list) > 1 else z_list[0]

            logits = self.head(fused)[0] if self._use_taylor else self.head(fused)

            extras = {"fused": fused}
            if z_stat  is not None: extras["z_stat"]  = z_stat
            if z_seq   is not None: extras["z_seq"]   = z_seq
            if z_bytes is not None: extras["z_bytes"] = z_bytes

        return logits, extras

    # ------------------------------------------------------------------
    def concept_sparsity_loss(self) -> torch.Tensor:
        """L1 on tabcbm soft masks (0 if using grouped mode or no stat encoder)."""
        if self.stat_encoder is not None:
            return self.stat_encoder.concept_mask_l1()
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(
    num_classes:    int,
    stat_dim:       Optional[int] = None,
    num_concepts:   int = 6,
    concept_mode:   str = "grouped",
    concept_groups: Optional[List[List[int]]] = None,
    max_seq_len:    int = 100,
    bytes_len:      int = 320,
    use_stat:       bool = True,
    use_seq:        bool = True,
    use_bytes:      bool = True,
    use_bytes_packet: Optional[bool] = None,
    use_bytes_cross: Optional[bool] = None,
    head:           str = "mlp",
    fusion_mode:    str = "6+1+1",
    # encoder hyper-params
    stat_hidden:    tuple = (64, 64, 32),
    seq_model_dim:  int = 64,
    seq_num_heads:  int = 4,
    seq_layers:     int = 2,
    bytes_model_dim: int = 64,
    bytes_num_heads: int = 4,
    bytes_layers:   int = 2,
    bytes_out_dim:  int = 1,
    bytes_arch:     str = "transformer",
    seq_pooling:    str = "mean",
    bytes_pooling:  str = "mean",
    dropout:        float = 0.1,
    # 6+4+5 / 6+4+5+3 per-packet bytes
    bytes_per_packet: bool = False,
    bytes_pkt_len:    int = 63,
    num_packets:      int = 5,
    num_cross:        int = 3,   # cross-packet groups for 6+4+5+3
    num_groups:       Optional[int] = None,   # deprecated: alias for num_cross (from config)
    mlp_hidden:       Optional[int] = None,   # MLP head only; if set, overrides default width
) -> TrafficCBM:
    """Convenience factory to build a TrafficCBM from scalar hyper-params.

    In "6/6/6" mode the seq and bytes out_dim are forced to num_concepts
    so all branches share the same concept space.
    """
    assert fusion_mode in ("6+1+1", "6/6/6", "6+4", "6+4|b", "6+4+5", "6+4+5+3", "6+4+3"), \
        f"fusion_mode must be '6+1+1', '6/6/6', '6+4', '6+4|b', '6+4+5', '6+4+5+3', or '6+4+3', got '{fusion_mode}'"

    n_cross = num_groups if num_groups is not None else num_cross

    stat_enc = None
    if use_stat:
        assert stat_dim is not None, "stat_dim required when use_stat=True"
        stat_enc = StatConceptEncoder(
            stat_dim=stat_dim,
            num_concepts=num_concepts,
            concept_mode=concept_mode,
            concept_groups=concept_groups,
            hidden_dims=stat_hidden,
            dropout=dropout,
        )

    seq_enc = None
    if use_seq:
        if fusion_mode in ("6+4", "6+4|b", "6+4+5", "6+4+5+3", "6+4+3"):
            # Level-3: 4 physically-isolated group Transformers
            seq_enc = SeqConceptEncoder(
                seq_len=max_seq_len,
                model_dim=seq_model_dim,
                num_heads=seq_num_heads,
                num_layers=seq_layers,
                dropout=dropout,
                pooling=seq_pooling,
            )
        else:
            # 6+1+1: out_dim=1; 6/6/6: out_dim=num_concepts
            seq_out = num_concepts if fusion_mode == "6/6/6" else 1
            seq_enc = SequenceEncoder(
                seq_dim=7,
                seq_len=max_seq_len,
                model_dim=seq_model_dim,
                num_heads=seq_num_heads,
                num_layers=seq_layers,
                out_dim=seq_out,
                dropout=dropout,
                pooling=seq_pooling,
            )

    bytes_enc = None
    if use_bytes:
        pkt_on = use_bytes_packet
        cross_on = use_bytes_cross
        if fusion_mode == "6+4+5":
            pkt_on = True if pkt_on is None else bool(pkt_on)
            cross_on = False
        elif fusion_mode == "6+4+3":
            pkt_on = False
            cross_on = True if cross_on is None else bool(cross_on)
        elif fusion_mode == "6+4+5+3":
            pkt_on = True if pkt_on is None else bool(pkt_on)
            cross_on = True if cross_on is None else bool(cross_on)
        else:
            pkt_on = bool(pkt_on) if pkt_on is not None else False
            cross_on = bool(cross_on) if cross_on is not None else False

        if fusion_mode == "6+4":
            # pure concept backbone: bytes not used in final prediction
            bytes_enc = None
        elif fusion_mode == "6+4+5":
            # Per-packet byte concepts: 5 packets → 5 scalars
            if pkt_on:
                bytes_enc = BytesPacketEncoder(
                    num_packets=num_packets,
                    bytes_pkt_len=bytes_pkt_len,
                    model_dim=bytes_model_dim,
                    num_heads=bytes_num_heads,
                    num_layers=bytes_layers,
                    dropout=dropout,
                    pooling=bytes_pooling,
                )
        elif fusion_mode == "6+4+5+3":
            # Flexible bytes sub-branches: 5 packet-level and/or 3 cross-packet
            if pkt_on or cross_on:
                if __import__('os').environ.get('CBM_PAIRWISE_CROSS') == '1':
                    from .encoders.bytes_pairwise import BytesPairwiseCrossEncoder
                    encoder = BytesPairwiseCrossEncoder
                else:
                    encoder = BytesPacketCrossEncoder
                bytes_enc = encoder(
                    num_packets=num_packets,
                    num_cross=n_cross,
                    bytes_pkt_len=bytes_pkt_len,
                    cross_only=(not pkt_on and cross_on),
                    use_packet=pkt_on,
                    use_cross=cross_on,
                    model_dim=bytes_model_dim,
                    num_heads=bytes_num_heads,
                    num_layers=bytes_layers,
                    dropout=dropout,
                    pooling=bytes_pooling,
                )
        elif fusion_mode == "6+4+3":
            # Use the three cross-packet scalars.
            if cross_on:
                if __import__('os').environ.get('CBM_PAIRWISE_CROSS') == '1':
                    from .encoders.bytes_pairwise import BytesPairwiseCrossEncoder
                    encoder = BytesPairwiseCrossEncoder
                else:
                    encoder = BytesPacketCrossEncoder
                bytes_enc = encoder(
                    num_packets=num_packets,
                    num_cross=n_cross,
                    bytes_pkt_len=bytes_pkt_len,
                    cross_only=True,
                    use_packet=False,
                    use_cross=True,
                    model_dim=bytes_model_dim,
                    num_heads=bytes_num_heads,
                    num_layers=bytes_layers,
                    dropout=dropout,
                    pooling=bytes_pooling,
                )
        elif fusion_mode == "6+4|b":
            # Level-2: conditional gate, cond_dim = num_concepts_stat + num_concepts_seq
            cond_dim = num_concepts + 4   # stat(6) + seq(4)
            bytes_enc = BytesGateEncoder(
                bytes_len=bytes_len,
                model_dim=bytes_model_dim,
                num_heads=bytes_num_heads,
                num_layers=bytes_layers,
                dropout=dropout,
                pooling=bytes_pooling,
                cond_dim=cond_dim,
            )
        elif bytes_arch == "cnn":
            bytes_out = num_concepts if fusion_mode == "6/6/6" else bytes_out_dim
            bytes_enc = BytesCNNEncoder(
                bytes_len=bytes_len,
                model_dim=bytes_model_dim,
                out_dim=bytes_out,
                dropout=dropout,
            )
        else:
            bytes_out = num_concepts if fusion_mode == "6/6/6" else bytes_out_dim
            bytes_enc = BytesEncoder(
                bytes_len=bytes_len,
                model_dim=bytes_model_dim,
                num_heads=bytes_num_heads,
                num_layers=bytes_layers,
                out_dim=bytes_out,
                dropout=dropout,
                pooling=bytes_pooling,
            )

    head_kwargs: Dict[str, int] = {"num_classes": num_classes}
    if head == "mlp" and mlp_hidden is not None:
        head_kwargs["hidden"] = int(mlp_hidden)

    return TrafficCBM(
        stat_encoder=stat_enc,
        seq_encoder=seq_enc,
        bytes_encoder=bytes_enc,
        num_classes=num_classes,
        head=head,
        fusion_mode=fusion_mode,
        head_kwargs=head_kwargs,
    )
