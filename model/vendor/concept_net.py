from typing import List
import torch
from torch import nn
import torch.nn.functional as F

class ConceptNet(nn.Module):
    def __init__(self, 
                num_mlps, 
                hidden_dims, 
                input_layer='linear', 
                concept_groups=None,
                activation=nn.ReLU(),
                order=1, 
                dropout=0.0, 
                batchnorm=False):
        
        super(ConceptNet, self).__init__()
        # This implementation uses grouped linear projections.
        assert input_layer == "linear"

        self.num_mlps = num_mlps
        self.model_depth = len(hidden_dims) + 1
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = activation

        assert concept_groups is not None and isinstance(concept_groups, list)
        assert len(concept_groups) == num_mlps
        self.concept_groups = concept_groups
        input_dims = [len(_input) for _input in concept_groups]

        layers = []

        # Project each group independently to hidden_dims[0].
        self.input_layers = nn.ModuleList([nn.Linear(input_dims[i], hidden_dims[0]) for i in range(num_mlps)])

        if self.batchnorm is True:
            layers.append(nn.BatchNorm1d(hidden_dims[0] * num_mlps))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(self.activation)
    
        # Grouped 1x1 convolutions implement independent linear layers.
        input_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers.append(nn.Conv1d(in_channels=input_dim * num_mlps,
                                    out_channels=dim * num_mlps,
                                    kernel_size=1,
                                    groups=num_mlps))
            if self.batchnorm is True:
                layers.append(nn.BatchNorm1d(dim * num_mlps))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(self.activation)
            input_dim = dim

        # Output one concept scalar per group.
        layers.append(nn.Conv1d(in_channels=input_dim * num_mlps,
                                out_channels=1 * num_mlps,
                                kernel_size=1,
                                groups=num_mlps))

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Receive one input tensor per concept group.
        Xs = [self.input_layers[i](x[:, self.concept_groups[i]]) for i in range(self.num_mlps)]
        Xs = torch.cat(Xs, dim=1).unsqueeze(-1)
        z = self.hidden_layers(Xs)
        return z


