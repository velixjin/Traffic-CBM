#!/usr/bin/env python3
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Internal helper: single-group Transformer for SeqConceptEncoder
# ---------------------------------------------------------------------------

class _GroupTransformerEncoder(nn.Module):
    """One semantic group of seq features → single concept scalar.

    Used exclusively by SeqConceptEncoder to implement Level-3 interpretability
    (physical feature isolation between groups).
    """

    def __init__(self, in_dim, seq_len, model_dim, num_heads, num_layers,
                 dropout, pooling):
        super().__init__()
        assert pooling in ("mean", "max", "cls")
        self.seq_len   = seq_len
        self.model_dim = model_dim
        self.pooling   = pooling

        self.input_proj = nn.Linear(in_dim, model_dim)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            self.pos_embed = nn.Parameter(torch.zeros(seq_len + 1, model_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        for attr in ("use_nested_tensor", "enable_nested_tensor"):
            if hasattr(self.encoder, attr):
                try:
                    setattr(self.encoder, attr, False)
                except Exception:
                    pass
        self.proj = nn.Linear(model_dim, 1)

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

    def forward(self, x_group, attn_mask):
        # x_group  : [B, N, in_dim]
        # attn_mask: [B, N]  1=valid 0=pad
        B, N, _ = x_group.shape
        x = self.input_proj(x_group)   # [B, N, D]

        if self.pooling == "cls":
            cls       = self.cls_token.expand(B, -1, -1)
            x         = torch.cat([cls, x], dim=1)
            x         = x + self.pos_embed.unsqueeze(0)
            cls_mask  = torch.ones((B, 1), device=x_group.device,
                                   dtype=attn_mask.dtype)
            full_mask = torch.cat([cls_mask, attn_mask], dim=1)
            kpm       = (full_mask == 0).bool()
            h         = self.encoder(x, src_key_padding_mask=kpm)
            h         = self._safe_to_dense(h, B, N + 1)
            pooled    = h[:, 0, :]
        else:
            x      = x + self.pos_embed.unsqueeze(0)
            kpm    = (attn_mask == 0).bool()
            h      = self.encoder(x, src_key_padding_mask=kpm)
            h      = self._safe_to_dense(h, B, N)
            mask   = attn_mask.unsqueeze(-1).float()
            if self.pooling == "max":
                pooled = (h * mask + (1.0 - mask) * (-1e4)).max(dim=1).values
            else:
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.proj(pooled)   # [B, 1]


# ---------------------------------------------------------------------------
# Grouped sequence encoder  (fusion_mode = "6+4|b")
# ---------------------------------------------------------------------------

class SeqConceptEncoder(nn.Module):
    """Encode seven packet features in four groups with independent Transformers.

    Feature indices and evidence groups:
      0  pktlen        ┐
      4  payload_len   ├─ volume (concept 0)
      5  header_len    ┘
      1  iat           ── timing (concept 1)
      2  direction     ── direction (concept 2)
      3  tcp_flags     ┐
      6  tcp_window    ┘─ TCP (concept 3)

    Each concept encoder receives only the features assigned to its group.

    Output: [B, 4]. The attention mask follows the SequenceEncoder interface.
    """

    # (group_indices, concept_name)
    GROUPS = [
        ([0, 4, 5], "volume"),
        ([1],        "timing"),
        ([2],        "direction"),
        ([3, 6],     "tcp"),
    ]
    GROUP_NAMES = [name for _, name in GROUPS]

    def __init__(
        self,
        seq_len:    int   = 100,
        model_dim:  int   = 32,
        num_heads:  int   = 2,
        num_layers: int   = 2,
        dropout:    float = 0.1,
        pooling:    str   = "mean",
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.out_dim     = len(self.GROUPS)   # 4
        self.num_concepts = self.out_dim

        self.group_encoders = nn.ModuleList([
            _GroupTransformerEncoder(
                in_dim=len(idx),
                seq_len=seq_len,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
            )
            for idx, _ in self.GROUPS
        ])

    def forward(self, x_seq, attn_mask):
        # x_seq    : [B, N, 7]
        # attn_mask: [B, N]  1=valid 0=pad
        concepts = []
        for (idx, _), enc in zip(self.GROUPS, self.group_encoders):
            x_group = x_seq[:, :, idx]          # [B, N, |group|]
            concepts.append(enc(x_group, attn_mask))   # [B, 1]
        return torch.cat(concepts, dim=1)        # [B, 4]


# ---------------------------------------------------------------------------
# Original Transformer encoder  (fusion_mode "6+1+1" / "6/6/6")
# ---------------------------------------------------------------------------

class SequenceEncoder(nn.Module):
	"""Use a Transformer to map a packet sequence [B, N, D] to out_dim scalars.
	The key padding mask identifies padded positions.

	out_dim=1: a single sequence scalar in the 6+1+1 configuration
	out_dim=K: K sequence concepts in the 6/6/6 configuration

	pooling : "mean" | "max" | "cls"
	    mean: masked average pooling (default)
	    max: masked max pooling
	    cls: use the output of a learnable token prepended to the sequence
	"""

	def __init__(
		self,
		seq_dim=7,
		seq_len=5,
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
		self.seq_len   = seq_len
		self.model_dim = model_dim
		self.out_dim   = int(out_dim)
		self.pooling   = pooling

		self.input_proj = nn.Linear(seq_dim, model_dim)

		# CLS pooling adds a learned token and one position.
		if pooling == "cls":
			self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
			self.pos_embed = nn.Parameter(torch.zeros(seq_len + 1, model_dim))
		else:
			self.pos_embed = nn.Parameter(torch.zeros(seq_len, model_dim))

		enc_layer = nn.TransformerEncoderLayer(
			d_model=model_dim,
			nhead=num_heads,
			dim_feedforward=model_dim * 4,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
		for attr in ("use_nested_tensor", "enable_nested_tensor"):
			if hasattr(self.encoder, attr):
				try:
					setattr(self.encoder, attr, False)
				except Exception:
					pass
		self.proj = nn.Linear(model_dim, self.out_dim)

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

	def forward(self, x_seq, attn_mask):
		# x_seq: [B, N, D_feat], attn_mask: [B, N] (1=valid, 0=pad)
		B, N, _ = x_seq.shape
		assert N == self.seq_len, "Input sequence length differs from the configured sequence length"

		x = self.input_proj(x_seq)   # [B, N, D]

		if self.pooling == "cls":
			cls  = self.cls_token.expand(B, -1, -1)         # [B, 1, D]
			x    = torch.cat([cls, x], dim=1)               # [B, N+1, D]
			x    = x + self.pos_embed.unsqueeze(0)          # [B, N+1, D]
			cls_mask  = torch.ones((B, 1), device=x_seq.device, dtype=attn_mask.dtype)
			full_mask = torch.cat([cls_mask, attn_mask], dim=1)  # [B, N+1]
			kpm = (full_mask == 0).bool()
			h   = self.encoder(x, src_key_padding_mask=kpm)
			h   = self._safe_to_dense(h, B, N + 1)
			pooled = h[:, 0, :]                             # [B, D]

		else:
			x    = x + self.pos_embed.unsqueeze(0)          # [B, N, D]
			kpm  = (attn_mask == 0).bool()
			h    = self.encoder(x, src_key_padding_mask=kpm)
			h    = self._safe_to_dense(h, B, N)
			mask = attn_mask.unsqueeze(-1).float()          # [B, N, 1]

			if self.pooling == "max":
				h_masked = h * mask + (1.0 - mask) * (-1e4)
				pooled   = h_masked.max(dim=1).values       # [B, D]
			else:  # mean
				pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

		return self.proj(pooled)   # [B, out_dim]
