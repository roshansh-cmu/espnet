import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
import math
import logging


def get_act_fun(act_fun):
    if act_fun == "relu":
        return F.relu
    elif act_fun == "elu":
        return 1 + F.elu
    elif act_fun == "sig":
        return F.sigmoid
    elif act_fun == "swish":
        return F.silu
    elif act_fun == "softmax":
        return F.softmax

class FourierFFTLayer(nn.Module):
    def __init__(self, mode:str='real'):
        super().__init__()
        
        assert mode in ['real', 'abs']
        self.mode = mode


    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        if self.mode == 'real':
            return torch.fft.fft2(hidden_states.float(), dim=(-2,-1)).real
        elif self.mode == 'abs':
            return torch.abs(torch.fft.fft2(hidden_states.float(), dim=(-2,-1)))


class FNetAttention(nn.Module):
    def __init__(
        self,
        # n_head: int,
        n_feat: int,
        ff_inner_dim: int,
        dropout_rate: float = 0.2,
        act_fun: str = "sig",
        mode: str = "real",
        feedforward: bool = True,
    ):
        super().__init__()
        
        # Heads should be 1 for raw FF attention since heads >1 does not gain any new information
        self.feedforward = feedforward

        self.fft = FourierFFTLayer(mode = mode)
        if feedforward:
            self.mixing_layernorm = nn.LayerNorm(n_feat)
            self.ff = nn.Linear(n_feat, ff_inner_dim)
            self.ff2 = nn.Linear(ff_inner_dim, n_feat)
            self.ff_layernorm = nn.LayerNorm(n_feat)
            self.dropout = nn.Dropout(dropout_rate)
            self.activation = get_act_fun(act_fun)


    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        
        if not self.feedforward:
            return self.fft(query)

        x = self.fft(query)
        x = self.mixing_layernorm(x+query)
        x_ff = self.ff(x)
        x_ff = self.ff2(self.activation(x_ff))
        x_ff = self.dropout(x_ff)
        x = self.ff_layernorm(x_ff + x)
        return x
