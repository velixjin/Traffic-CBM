#!/usr/bin/env python3
from typing import Optional

import torch
from torch import nn

# Bigram vocab: 0-65535 (65536) + PAD/SEP/MASK/CLS (4) = 65540
BYTES_VOCAB_SIZE = 65540


# ---------------------------------------------------------------------------
# Conditional gate encoder  (fusion_mode = "6+4|b")
# ---------------------------------------------------------------------------

class BytesGateEncoder(nn.Module):
    """Byte-conditioned encoder for concept-wise gates.

    Use the concatenated statistical and sequence concepts to compute gate logits.

    Architecture
    ------------
    1. Embed the tokens as [B, L, D] and add position embeddings.
    2. TransformerEncoder × num_layers → pool → h_bytes [B, D]
    3. concat(h_bytes, c_cond)         → [B, D + cond_dim]
    4. Linear(D + cond_dim, cond_dim)  → g_bytes [B, cond_dim]

    Forward
    -------
    inputs : x_bytes [B, L], bytes_mask [B, L], c_cond [B, cond_dim]
    output : g_bytes [B, cond_dim]   (gate logits, apply sigmoid in traffic_cbm)

    Usage in TrafficCBM:
        c_cond  = cat(c_stat, c_seq)          [B, cond_dim=10]
        g_bytes = bytes_encoder(x_bytes, mask, c_cond)
        c_final = c_cond * sigmoid(g_bytes)   [B, 10]
    """

    def __init__(
        self,
        bytes_len:  int   = 320,
        vocab_size: int   = BYTES_VOCAB_SIZE,
        model_dim:  int   = 64,
        num_heads:  int   = 4,
        num_layers: int   = 2,
        dropout:    float = 0.1,
        pooling:    str   = "mean",
        cond_dim:   int   = 10,   # = num_concepts_stat + num_concepts_seq = 6+4
    ):
        super().__init__()
        assert pooling in ("mean", "max", "cls")
        self.bytes_len = int(bytes_len)
        self.model_dim = int(model_dim)
        self.pooling   = pooling
        self.cond_dim  = int(cond_dim)
        self.out_dim   = int(cond_dim)   # interface compatibility

        self.byte_embed = nn.Embedding(int(vocab_size), self.model_dim)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
            self.pos_embed = nn.Parameter(torch.zeros(self.bytes_len + 1, self.model_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(self.bytes_len, self.model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=int(num_heads),
            dim_feedforward=self.model_dim * 4, dropout=float(dropout),
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        for attr in ("use_nested_tensor", "enable_nested_tensor"):
            if hasattr(self.encoder, attr):
                try:
                    setattr(self.encoder, attr, False)
                except Exception:
                    pass

        # gate head: conditioned on c_stat + c_seq
        self.gate_head = nn.Sequential(
            nn.Linear(self.model_dim + self.cond_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.model_dim, self.cond_dim),
        )
        # init gate bias to +3 so sigmoid(g) ≈ 0.95 at start (pass-through for Stage-2)
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(3.0)

    def _safe_to_dense(self, h, B, L):
        try:
            h = h.to_padded_tensor(0.0, (B, L, self.model_dim))
        except Exception:
            try:
                from torch.nested import to_padded_tensor as _tp
                h = _tp(h, 0.0, (B, L, self.model_dim))
            except Exception:
                pass
        return h

    def _pool(self, h, attn_mask, B, L):
        if self.pooling == "cls":
            return h[:, 0, :]
        mask = attn_mask.unsqueeze(-1).float()   # [B, L, 1]
        if self.pooling == "max":
            return (h * mask + (1.0 - mask) * (-1e4)).max(dim=1).values
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        x_bytes:    torch.Tensor,
        bytes_mask: Optional[torch.Tensor] = None,
        c_cond:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_bytes   : [B, L] long
        bytes_mask: [B, L] int, 1=valid 0=pad
        c_cond    : [B, cond_dim]  — concat(c_stat, c_seq)
        returns   : g_bytes [B, cond_dim]  gate logits (before sigmoid)
        """
        B, L = x_bytes.shape
        if bytes_mask is None:
            bytes_mask = torch.ones((B, L), device=x_bytes.device, dtype=torch.long)

        x = self.byte_embed(x_bytes)   # [B, L, D]

        if self.pooling == "cls":
            cls       = self.cls_token.expand(B, -1, -1)
            x         = torch.cat([cls, x], dim=1)
            x         = x + self.pos_embed.unsqueeze(0)
            cls_mask  = torch.ones((B, 1), device=x_bytes.device, dtype=bytes_mask.dtype)
            full_mask = torch.cat([cls_mask, bytes_mask], dim=1)
            kpm       = (full_mask == 0).bool()
            h         = self.encoder(x, src_key_padding_mask=kpm)
            h         = self._safe_to_dense(h, B, L + 1)
            pooled    = self._pool(h, full_mask, B, L + 1)
        else:
            x      = x + self.pos_embed.unsqueeze(0)
            kpm    = (bytes_mask == 0).bool()
            h      = self.encoder(x, src_key_padding_mask=kpm)
            h      = self._safe_to_dense(h, B, L)
            pooled = self._pool(h, bytes_mask, B, L)   # [B, D]

        if c_cond is not None:
            gate_in = torch.cat([pooled, c_cond], dim=1)   # [B, D + cond_dim]
        else:
            # fallback: pad with zeros when c_cond not provided
            zeros   = torch.zeros(B, self.cond_dim, device=x_bytes.device,
                                  dtype=pooled.dtype)
            gate_in = torch.cat([pooled, zeros], dim=1)

        return self.gate_head(gate_in)   # [B, cond_dim]


# ---------------------------------------------------------------------------
# Per-packet encoder  (fusion_mode = "6+4+5")
# ---------------------------------------------------------------------------

class BytesPacketEncoder(nn.Module):
    """Encode each packet as one scalar byte concept.

    Inputs: x_bytes and bytes_mask, both [B, num_packets, pkt_len].
    Output: c_bytes [B, num_packets], with one scalar per packet.

    A shared Transformer encodes each packet and projects its representation to a scalar.
    """

    def __init__(
        self,
        num_packets:  int   = 5,
        bytes_pkt_len: int  = 63,
        vocab_size:   int   = BYTES_VOCAB_SIZE,
        model_dim:    int   = 64,
        num_heads:    int   = 4,
        num_layers:   int   = 2,
        dropout:      float = 0.1,
        pooling:      str   = "mean",
    ):
        super().__init__()
        assert pooling in ("mean", "max", "cls")
        self.num_packets   = int(num_packets)
        self.bytes_pkt_len = int(bytes_pkt_len)
        self.model_dim     = int(model_dim)
        self.pooling       = pooling
        self.out_dim       = int(num_packets)

        self.byte_embed = nn.Embedding(int(vocab_size), self.model_dim)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
            self.pos_embed = nn.Parameter(torch.zeros(self.bytes_pkt_len + 1, self.model_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(self.bytes_pkt_len, self.model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=int(num_heads),
            dim_feedforward=self.model_dim * 4, dropout=float(dropout),
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        for attr in ("use_nested_tensor", "enable_nested_tensor"):
            if hasattr(self.encoder, attr):
                try:
                    setattr(self.encoder, attr, False)
                except Exception:
                    pass

        self.proj = nn.Linear(self.model_dim, 1)

    def _safe_to_dense(self, h, B, L):
        try:
            h = h.to_padded_tensor(0.0, (B, L, self.model_dim))
        except Exception:
            try:
                from torch.nested import to_padded_tensor as _tp
                h = _tp(h, 0.0, (B, L, self.model_dim))
            except Exception:
                pass
        return h

    def _pool(self, h, attn_mask, B, L):
        if self.pooling == "cls":
            return h[:, 0, :]
        mask = attn_mask.unsqueeze(-1).float()
        if self.pooling == "max":
            return (h * mask + (1.0 - mask) * (-1e4)).max(dim=1).values
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        x_bytes:    torch.Tensor,
        bytes_mask: Optional[torch.Tensor] = None,
        c_cond:     Optional[torch.Tensor] = None,  # unused, for interface compatibility
    ) -> torch.Tensor:
        """
        x_bytes   : [B, num_packets, pkt_len] long
        bytes_mask: [B, num_packets, pkt_len] int, 1=valid 0=pad
        returns   : c_bytes [B, num_packets]
        """
        B, N, L = x_bytes.shape
        if bytes_mask is None:
            bytes_mask = torch.ones((B, N, L), device=x_bytes.device, dtype=torch.long)

        # Flatten to [B*N, L] for batched encoding
        x_flat = x_bytes.reshape(B * N, L)
        mask_flat = bytes_mask.reshape(B * N, L)

        x = self.byte_embed(x_flat)
        if self.pooling == "cls":
            cls = self.cls_token.expand(B * N, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed.unsqueeze(0)
            cls_mask = torch.ones((B * N, 1), device=x_bytes.device, dtype=bytes_mask.dtype)
            full_mask = torch.cat([cls_mask, mask_flat], dim=1)
            kpm = (full_mask == 0).bool()
            h = self.encoder(x, src_key_padding_mask=kpm)
            h = self._safe_to_dense(h, B * N, L + 1)
            pooled = self._pool(h, full_mask, B * N, L + 1)
        else:
            x = x + self.pos_embed.unsqueeze(0)
            kpm = (mask_flat == 0).bool()
            h = self.encoder(x, src_key_padding_mask=kpm)
            h = self._safe_to_dense(h, B * N, L)
            pooled = self._pool(h, mask_flat, B * N, L)

        # [B*N, D] -> [B*N, 1] -> [B, N]
        c_flat = self.proj(pooled).squeeze(-1)
        c_bytes = c_flat.reshape(B, N)
        # Packet-level mask: zero out invalid packets (flows with < num_packets)
        packet_valid = (bytes_mask.sum(dim=2) > 0).to(c_bytes.dtype)
        return c_bytes * packet_valid


# ---------------------------------------------------------------------------
# Packet-level + cross-packet encoder  (fusion_mode = "6+4+5+3")
# ---------------------------------------------------------------------------

class BytesPacketCrossEncoder(nn.Module):
    """Encode packet-level and cross-packet byte evidence as separate scalars.

    - Packet-level: map the 63 tokens in each packet to one scalar when enabled.
    - Cross-packet: concatenate the same segment from all five packets, then map it to one scalar.
        The default segments are the first, middle, and last 21 tokens.
    Output: c_bytes [B, 5], [B, 3], or [B, 8], depending on which branches are enabled.
    """

    def __init__(
        self,
        num_packets:   int   = 5,
        num_cross:     int   = 3,
        bytes_pkt_len: int   = 63,
        cross_only:    bool  = False,   # Enable only cross-packet concepts.
        use_packet:    Optional[bool] = None,
        use_cross:     Optional[bool] = None,
        vocab_size:    int   = BYTES_VOCAB_SIZE,
        model_dim:     int   = 64,
        num_heads:     int   = 4,
        num_layers:    int   = 2,
        dropout:       float = 0.1,
        pooling:       str   = "mean",
    ):
        super().__init__()
        assert pooling in ("mean", "max", "cls")
        if use_packet is None:
            use_packet = not cross_only
        if use_cross is None:
            use_cross = True
        assert use_packet or use_cross, "BytesPacketCrossEncoder requires at least one active sub-branch"
        if use_cross:
            assert bytes_pkt_len % num_cross == 0
        self.num_packets   = int(num_packets)
        self.num_cross     = int(num_cross)
        self.bytes_pkt_len = int(bytes_pkt_len)
        self.use_packet    = bool(use_packet)
        self.use_cross     = bool(use_cross)
        self.cross_only    = (not self.use_packet) and self.use_cross
        self.packet_only   = self.use_packet and (not self.use_cross)
        self.group_len     = bytes_pkt_len // num_cross   # 21
        self.cross_len     = num_packets * self.group_len  # 105
        self.model_dim     = int(model_dim)
        self.pooling       = pooling
        self.out_dim       = (int(num_packets) if self.use_packet else 0) + (int(num_cross) if self.use_cross else 0)

        self.byte_embed = nn.Embedding(int(vocab_size), self.model_dim)

        # Packet encoder: seq len = bytes_pkt_len (63)
        if self.use_packet:
            if pooling == "cls":
                self.cls_pkt = nn.Parameter(torch.zeros(1, 1, self.model_dim))
                self.pos_pkt = nn.Parameter(torch.zeros(self.bytes_pkt_len + 1, self.model_dim))
            else:
                self.pos_pkt = nn.Parameter(torch.zeros(self.bytes_pkt_len, self.model_dim))
            enc_pkt = nn.TransformerEncoderLayer(
                d_model=self.model_dim, nhead=int(num_heads),
                dim_feedforward=self.model_dim * 4, dropout=float(dropout),
                batch_first=True, activation="gelu",
            )
            self.encoder_pkt = nn.TransformerEncoder(enc_pkt, num_layers=int(num_layers))
        else:
            self.cls_pkt = None
            self.pos_pkt = None
            self.encoder_pkt = None

        # Cross-packet encoder: seq len = cross_len (105)
        if self.use_cross and pooling == "cls":
            self.cls_cross = nn.Parameter(torch.zeros(1, 1, self.model_dim))
            self.pos_cross = nn.Parameter(torch.zeros(self.cross_len + 1, self.model_dim))
        elif self.use_cross:
            self.pos_cross = nn.Parameter(torch.zeros(self.cross_len, self.model_dim))
        else:
            self.cls_cross = None
            self.pos_cross = None
        if self.use_cross:
            enc_cross = nn.TransformerEncoderLayer(
                d_model=self.model_dim, nhead=int(num_heads),
                dim_feedforward=self.model_dim * 4, dropout=float(dropout),
                batch_first=True, activation="gelu",
            )
            self.encoder_cross = nn.TransformerEncoder(enc_cross, num_layers=int(num_layers))
        else:
            self.encoder_cross = None

        for enc in (e for e in (self.encoder_pkt, self.encoder_cross) if e is not None):
            for attr in ("use_nested_tensor", "enable_nested_tensor"):
                if hasattr(enc, attr):
                    try:
                        setattr(enc, attr, False)
                    except Exception:
                        pass

        self.proj = nn.Linear(self.model_dim, 1)

    def _safe_to_dense(self, h, B, L):
        try:
            h = h.to_padded_tensor(0.0, (B, L, self.model_dim))
        except Exception:
            try:
                from torch.nested import to_padded_tensor as _tp
                h = _tp(h, 0.0, (B, L, self.model_dim))
            except Exception:
                pass
        return h

    def _pool(self, h, attn_mask, B, L):
        if self.pooling == "cls":
            return h[:, 0, :]
        mask = attn_mask.unsqueeze(-1).float()
        if self.pooling == "max":
            return (h * mask + (1.0 - mask) * (-1e4)).max(dim=1).values
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        x_bytes:    torch.Tensor,
        bytes_mask: Optional[torch.Tensor] = None,
        c_cond:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_bytes   : [B, num_packets, pkt_len] long
        bytes_mask: [B, num_packets, pkt_len] int, 1=valid 0=pad
        returns   : c_bytes [B, out_dim]
        """
        B, N, L = x_bytes.shape
        if bytes_mask is None:
            bytes_mask = torch.ones((B, N, L), device=x_bytes.device, dtype=torch.long)
        G = self.num_cross
        GL = self.group_len

        c_parts = []

        # ---- 5 packet-level ----
        if self.use_packet:
            x_flat = x_bytes.reshape(B * N, L)
            mask_flat = bytes_mask.reshape(B * N, L)
            x_e = self.byte_embed(x_flat)
            if self.pooling == "cls":
                cls = self.cls_pkt.expand(B * N, -1, -1)
                x_e = torch.cat([cls, x_e], dim=1) + self.pos_pkt.unsqueeze(0)
                m = torch.cat([torch.ones(B*N,1,device=x_bytes.device,dtype=bytes_mask.dtype), mask_flat], dim=1)
            else:
                x_e = x_e + self.pos_pkt.unsqueeze(0)
                m = mask_flat
            kpm = (m == 0).bool()
            h = self.encoder_pkt(x_e, src_key_padding_mask=kpm)
            h = self._safe_to_dense(h, B * N, x_e.shape[1])
            pooled = self._pool(h, m, B * N, x_e.shape[1])
            c_pkt = self.proj(pooled).squeeze(-1).reshape(B, N)
            packet_valid = (bytes_mask.sum(dim=2) > 0).to(c_pkt.dtype)
            c_pkt = c_pkt * packet_valid
            c_parts.append(c_pkt)

        # ---- 3 cross-packet ----
        if self.use_cross:
            x_g = x_bytes.reshape(B, N, G, GL)      # [B, 5, 3, 21]
            mask_g = bytes_mask.reshape(B, N, G, GL)
            c_cross_list = []
            for g in range(G):
                # Concatenate segment g from each packet to form [B, 5 * 21].
                x_c = x_g[:, :, g, :].reshape(B, N * GL)      # [B, 105]
                m_c = mask_g[:, :, g, :].reshape(B, N * GL)
                x_e = self.byte_embed(x_c)
                if self.pooling == "cls":
                    cls = self.cls_cross.expand(B, -1, -1)
                    x_e = torch.cat([cls, x_e], dim=1) + self.pos_cross.unsqueeze(0)
                    m_c = torch.cat([torch.ones(B, 1, device=x_bytes.device, dtype=bytes_mask.dtype), m_c], dim=1)
                else:
                    x_e = x_e + self.pos_cross.unsqueeze(0)
                seq_len = x_e.shape[1]
                kpm = (m_c == 0).bool()
                h = self.encoder_cross(x_e, src_key_padding_mask=kpm)
                h = self._safe_to_dense(h, B, seq_len)
                pooled = self._pool(h, m_c, B, seq_len)
                c_g = self.proj(pooled).squeeze(-1)
                cross_valid = (m_c.sum(dim=1) > 0).to(c_g.dtype)
                c_cross_list.append(c_g * cross_valid)
            c_cross = torch.stack(c_cross_list, dim=1)   # [B, 3]
            c_parts.append(c_cross)

        if len(c_parts) == 1:
            return c_parts[0]
        return torch.cat(c_parts, dim=1)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class _ResConv1dBlock(nn.Module):
    """Pre-activation residual 1D-conv block (BN→ReLU→Conv→BN→ReLU→Dropout→Conv + skip).

    Same padding keeps temporal length unchanged.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


# ---------------------------------------------------------------------------
# CNN encoder  (bytes_arch = "cnn")
# ---------------------------------------------------------------------------

class BytesCNNEncoder(nn.Module):
    """Multiscale 1D convolutional byte encoder.

    Architecture
    ------------
    1. Embedding [B, L] → [B, L, D] → transpose → [B, D, L]
    2. Three parallel Conv1d layers (kernels 3, 5, and 7), each followed by batch normalization and ReLU, produce [B, D, L] tensors. Concatenate them to [B, 3D, L].
    3. 2× ResConv1dBlock (channels=3D, k=3)  → [B, 3D, L]
    4. Masked global average pooling         → [B, 3D]
    5. Linear(3D, out_dim)                   → z_bytes [B, out_dim]

    Returns a single tensor through the same interface as BytesEncoder.
    """

    def __init__(
        self,
        bytes_len: int = 320,
        vocab_size: int = BYTES_VOCAB_SIZE,
        model_dim: int = 64,
        kernel_sizes: tuple = (3, 5, 7),
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bytes_len = int(bytes_len)
        self.out_dim   = int(out_dim)

        self.byte_embed = nn.Embedding(int(vocab_size), int(model_dim))

        self.ms_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, k, padding=k // 2),
                nn.BatchNorm1d(model_dim),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])

        fused_ch = model_dim * len(kernel_sizes)   # 192 with defaults

        self.res_blocks = nn.Sequential(
            _ResConv1dBlock(fused_ch, kernel_size=3, dropout=dropout),
            _ResConv1dBlock(fused_ch, kernel_size=3, dropout=dropout),
        )

        self.proj = nn.Linear(fused_ch, self.out_dim)

    def forward(
        self,
        x_bytes: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_bytes   : [B, L] long
        attn_mask : [B, L] int, 1=valid 0=pad
        returns   : [B, out_dim]
        """
        x = self.byte_embed(x_bytes)   # [B, L, D]
        x = x.transpose(1, 2)          # [B, D, L]

        if attn_mask is not None:
            x = x * attn_mask.unsqueeze(1).float()

        branches = [conv(x) for conv in self.ms_convs]
        h = torch.cat(branches, dim=1)         # [B, 3D, L]
        h = self.res_blocks(h)                 # [B, 3D, L]

        if attn_mask is not None:
            mask   = attn_mask.unsqueeze(1).float()
            h_pool = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        else:
            h_pool = h.mean(dim=2)             # [B, 3D]

        return self.proj(h_pool)               # [B, out_dim]


# ---------------------------------------------------------------------------
# Transformer encoder  (bytes_arch = "transformer", original)
# ---------------------------------------------------------------------------

class BytesEncoder(nn.Module):
    """Encode a bigram token sequence as a vector with out_dim entries.

    Input: x_bytes [B, L] with dtype torch.long and values from 0 to 65539.

    pooling : "mean" | "max" | "cls"
        mean: masked average pooling (default)
        max: masked max pooling
        cls: use the output of a learnable token prepended to the sequence
    """

    def __init__(
        self,
        bytes_len=320,
        vocab_size=BYTES_VOCAB_SIZE,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        out_dim=1,
        pooling="mean",
    ):
        super().__init__()
        assert pooling in ("mean", "max", "cls"), \
            f"pooling must be 'mean'/'max'/'cls', got '{pooling}'"
        self.bytes_len = int(bytes_len)
        self.model_dim = int(model_dim)
        self.out_dim   = int(out_dim)
        self.pooling   = pooling

        self.byte_embed = nn.Embedding(int(vocab_size), self.model_dim)

        # CLS pooling adds a learned token and one position.
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
            self.pos_embed = nn.Parameter(torch.zeros(self.bytes_len + 1, self.model_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(self.bytes_len, self.model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=self.model_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        for attr in ("use_nested_tensor", "enable_nested_tensor"):
            if hasattr(self.encoder, attr):
                try:
                    setattr(self.encoder, attr, False)
                except Exception:
                    pass
        self.proj = nn.Linear(self.model_dim, self.out_dim)

    def _safe_to_dense(self, h, B, seq_len):
        try:
            h = h.to_padded_tensor(0.0, (B, seq_len, self.model_dim))
        except Exception:
            try:
                from torch.nested import to_padded_tensor as _tp
                h = _tp(h, 0.0, (B, seq_len, self.model_dim))
            except Exception:
                pass
        return h

    def forward(self, x_bytes, attn_mask=None):
        # x_bytes: [B, L] long
        B, L = x_bytes.shape
        assert L == self.bytes_len, f"bytes_len mismatch: got {L}, expected {self.bytes_len}"

        if attn_mask is None:
            attn_mask = torch.ones((B, L), device=x_bytes.device, dtype=torch.long)

        x = self.byte_embed(x_bytes)   # [B, L, D]

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)          # [B, 1, D]
            x   = torch.cat([cls, x], dim=1)                # [B, L+1, D]
            x   = x + self.pos_embed.unsqueeze(0)           # [B, L+1, D]
            # The CLS position is always valid.
            cls_mask  = torch.ones((B, 1), device=x_bytes.device, dtype=torch.long)
            full_mask = torch.cat([cls_mask, attn_mask], dim=1)  # [B, L+1]
            kpm = (full_mask == 0).bool()
            h   = self.encoder(x, src_key_padding_mask=kpm)
            h   = self._safe_to_dense(h, B, L + 1)
            pooled = h[:, 0, :]                             # [B, D] CLS representation

        else:
            x = x + self.pos_embed.unsqueeze(0)             # [B, L, D]
            kpm = (attn_mask == 0).bool()
            h   = self.encoder(x, src_key_padding_mask=kpm)
            h   = self._safe_to_dense(h, B, L)
            mask = attn_mask.unsqueeze(-1).float()          # [B, L, 1]

            if self.pooling == "max":
                h_masked = h * mask + (1.0 - mask) * (-1e4)
                pooled   = h_masked.max(dim=1).values       # [B, D]
            else:  # mean
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.proj(pooled)   # [B, out_dim]
