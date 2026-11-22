from typing import List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sympy
import tensorly as tl
tl.set_backend('pytorch')

from .concept_net import ConceptNet


class Fast_Tucker_Taylor(nn.Module):
    def __init__(self,
                    in_features: int,
                    out_features: int = None,
                    X0=None,
                    order=2,
                    rank=50,
                    initial='Taylor') -> None:
        super().__init__()
        out_features = out_features or in_features
        self.order = order
        self.initial = initial
        if X0 is None:
            self.X0 = 0.0
        else:
            assert type(X0) == float or type(X0) == torch.tensor
            self.X0 = X0

        self.const = nn.Parameter(torch.empty((out_features, )))

        self.Os = [
            nn.Parameter(torch.empty((out_features, rank)))
            for i in range(order)
        ]
        self.Is = [
            nn.Parameter(torch.empty(((i+1), rank, in_features)))
            for i in range(order)
        ]
        self.Gs = [
            nn.Parameter(torch.empty((rank, rank**(i + 1))))
            for i in range(order)
        ]

        self.Os = nn.ParameterList(self.Os)
        self.Is = nn.ParameterList(self.Is)
        self.Gs = nn.ParameterList(self.Gs)

        self.reset_parameter()

    def reset_parameter(self):
        if self.initial != 'Taylor':
            nn.init.zeros_(self.const)
            for O in self.Os:
                nn.init.xavier_uniform_(O)
            for I in self.Is:
                nn.init.xavier_uniform_(I)
            for G in self.Gs:
                nn.init.xavier_uniform_(G)
        else:
            nn.init.zeros_(self.const)
            for i, O in enumerate(self.Os):
                if i == 0:
                    nn.init.normal_(O, mean=0, std=(1 / (O.shape[1])) ** 0.5)
                else:
                    nn.init.normal_(O, mean=0, std=(0 / (O.shape[1])) ** 0.5)

            def _get_base(in_features, current_order):
                res = 1
                for i in range(current_order):
                    res = res / (in_features + 2 * i)
                return res

            for i, I in enumerate(self.Is):
                exponent = 1/2**(i+1)
                base = _get_base(I.shape[1], i+1)
                nn.init.normal_(I, mean=0, std=base ** exponent)

            for G in self.Gs:
                nn.init.normal_(G, mean=0, std=(1 / (G.shape[1])) ** 0.5)

    def reconstruct(self):
        self.to(torch.device('cpu'))
        rank_i, in_features = self.Is[0].shape[1:]
        out_features, rank_o = self.Os[0].shape
        order = len(self.Os)
        params = [self.const.detach()]

        for i in range(order):
            I = self.Is[i]
            O = self.Os[i]
            G = self.Gs[i]
            G = G.reshape([rank_o] + [rank_i for _ in range(i + 1)])
            for j in range(0, i + 1):
                G = G.transpose(-1 - j, -1)
                G = G @ I[j]
                G = G.transpose(-1 - j, -1)
            param = G.transpose(0, -1) @ O.T
            param = param.transpose(0, -1)
            params.append(param.detach())
        return params

    def forward(self, X: torch.tensor):
        flag_reshape = False
        if X.dim() == 1:
            X = X.reshape(1, -1)
            flag_reshape = True

        batch_size, in_features = X.shape
        out_features, rank_o = self.Os[0].shape
        rank_i = self.Is[0].shape[1]
        order = len(self.Os)

        if type(self.X0) == torch.tensor:
            assert X.shape[1] == self.X0.shape[0]
        X = X - self.X0

        Y = torch.zeros((batch_size, out_features)).to(X.device)
        Y = Y + self.const
        for i in range(order):
            I = self.Is[i]
            O = self.Os[i]
            G = self.Gs[i]
            Z = I @ X.T
            if i != 0:
                kron_res = Z[0]
                for j in range(1, i + 1):
                    X3 = Z[j].reshape([rank_i] + [1 for _ in range(j)] + [batch_size])
                    kron_res = (kron_res.unsqueeze(0) * X3)
            else:
                kron_res = Z
            kron_res = kron_res.reshape((-1, batch_size))
            Y = Y + (O @ (G @ kron_res)).T
        if flag_reshape:
            return Y.reshape(-1)
        else:
            return Y


class TaylorNet(nn.Module):
    def __init__(self,
                num_inputs,
                num_outputs,
                concept_groups=None,
                input_layer='linear',
                hidden_dims=[64,64,32],
                X0=None,
                order=2,
                rank=8,
                initial='Xavier',
                concept_dropout=0.0,
                batchnorm=True,
                encode_concepts=True,
                output_penalty=0.0):
        
        super(TaylorNet, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.concept_dropout = concept_dropout
        self.batchnorm = batchnorm
        self.output_penalty = output_penalty
        self.concept_groups = concept_groups

        if encode_concepts:
            self.concept_nns = ConceptNet(num_inputs, 
                                        hidden_dims,
                                        input_layer=input_layer,
                                        concept_groups=concept_groups,
                                        activation=nn.LeakyReLU(),
                                        dropout=self.concept_dropout,
                                        batchnorm=self.batchnorm)
        else:
            self.concept_nns = None
        
        self.taylor_layers = Fast_Tucker_Taylor(num_inputs, 
                                                num_outputs, 
                                                X0=X0, 
                                                order=order, 
                                                rank=rank, 
                                                initial=initial)
    def output_loss(self, output):
        return self.output_penalty * (torch.pow(output, 2).mean(dim=-1)).mean()
  
    def forward(self, x):
        if self.concept_nns:
            z = self.concept_nns(x).squeeze(-1)
        else:
            z = x
        output = self.taylor_layers(z)
        if self.num_outputs == 1:
            output = output.squeeze(-1)

        return output, z


