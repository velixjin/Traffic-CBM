#!/usr/bin/env python3
"""
stat_encoder.py - Statistical feature concept encoder.

Two modes (concept_mode):
  "grouped"  (default): Pre-defined concept groups via ConceptNet.
                        Same as FlowCAT-v2. Requires concept_groups.
  "tabcbm"            : Learnable soft mask per concept (TabCBM-style).
                        Automatically discovers groupings. No predefined groups needed.
"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..vendor.concept_net import ConceptNet


class StatConceptEncoder(nn.Module):
    """Map raw statistical features to concept vector z_stat ∈ R^{num_concepts}.

    Parameters
    ----------
    stat_dim        : number of input statistical features (e.g. 76)
    num_concepts    : K — number of output concepts
    concept_mode    : "grouped" | "tabcbm"
    concept_groups  : required when concept_mode="grouped"; list of K lists of feature indices
    hidden_dims     : MLP hidden layer sizes (shared across concepts in grouped mode)
    dropout         : dropout probability
    batchnorm       : use BatchNorm in ConceptNet
    """

    def __init__(
        self,
        stat_dim: int,
        num_concepts: int = 6,
        concept_mode: str = "grouped",
        concept_groups: Optional[List[List[int]]] = None,
        hidden_dims: tuple = (64, 64, 32),
        dropout: float = 0.0,
        batchnorm: bool = True,
    ):
        super().__init__()
        assert concept_mode in ("grouped", "tabcbm"), \
            f"concept_mode must be 'grouped' or 'tabcbm', got '{concept_mode}'"

        self.concept_mode = concept_mode
        self.num_concepts = num_concepts
        self.stat_dim = stat_dim

        if concept_mode == "grouped":
            assert concept_groups is not None and len(concept_groups) == num_concepts, \
                "concept_mode='grouped' requires concept_groups of length num_concepts"
            self.concept_groups = concept_groups
            self.encoder = ConceptNet(
                num_mlps=num_concepts,
                hidden_dims=list(hidden_dims),
                input_layer="linear",
                concept_groups=concept_groups,
                activation=nn.ReLU(),
                dropout=dropout,
                batchnorm=batchnorm,
            )

        else:  # tabcbm
            # Learnable soft mask: each concept has a sigmoid gate over all features.
            # If concept_groups is provided, initialize in-group logits to +2 (~0.88)
            # and out-group logits to -2 (~0.12) as a domain-knowledge warm start.
            # The mask remains fully learnable and can deviate from the initial grouping.
            if concept_groups is not None and len(concept_groups) == num_concepts:
                init = torch.full((num_concepts, stat_dim), -2.0)
                for k, grp in enumerate(concept_groups):
                    for i in grp:
                        if i < stat_dim:
                            init[k, i] = 2.0
                self.concept_mask = nn.Parameter(init)
            else:
                self.concept_mask = nn.Parameter(torch.zeros(num_concepts, stat_dim))

            # Shared MLP backbone applied per concept (using grouped 1×1 Conv trick)
            layers: List[nn.Module] = []
            in_dim = stat_dim
            for h in hidden_dims:
                layers.append(nn.Conv1d(
                    in_dim * num_concepts, h * num_concepts,
                    kernel_size=1, groups=num_concepts,
                ))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(h * num_concepts))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Conv1d(
                in_dim * num_concepts, num_concepts,
                kernel_size=1, groups=num_concepts,
            ))
            self.tabcbm_mlp = nn.Sequential(*layers)

    def forward(self, x_stat: torch.Tensor) -> torch.Tensor:
        """
        x_stat: [B, stat_dim]
        returns: [B, num_concepts]
        """
        if self.concept_mode == "grouped":
            z = self.encoder(x_stat).squeeze(-1)  # [B, K]

        else:  # tabcbm
            # Soft mask: gate each concept's view of the features
            mask = torch.sigmoid(self.concept_mask)          # [K, D]
            # Apply mask: for each concept k, input is x * mask[k]
            # x_stat [B,D] → [B, K*D] → unsqueeze → [B, K*D, 1]
            x_masked = (x_stat.unsqueeze(1) * mask.unsqueeze(0))  # [B, K, D]
            B = x_stat.size(0)
            x_in = x_masked.view(B, self.num_concepts * self.stat_dim, 1)  # [B, K*D, 1]
            z = self.tabcbm_mlp(x_in).squeeze(-1)  # [B, K]

        return z  # [B, num_concepts]

    def concept_mask_l1(self) -> torch.Tensor:
        """L1 sparsity loss on soft masks (tabcbm mode only). Returns 0 in grouped mode."""
        if self.concept_mode == "tabcbm":
            return torch.sigmoid(self.concept_mask).abs().mean()
        return torch.tensor(0.0)
