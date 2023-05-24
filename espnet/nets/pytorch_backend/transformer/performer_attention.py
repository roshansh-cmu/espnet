import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
import math
import logging

from performer_pytorch import FastAttention


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


class PerformerAttention(torch.nn.Module):

    def __init__(
        self,
        size: int,
        n_feat: int,
        act_fun: str,
    ):
        super().__init__()
        self.attn_fn = FastAttention(
            dim_heads = size,
            nb_features = n_feat,
            causal = False,
            kernel_fn = get_act_fun(act_fun)
        )

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        print(query.shape)
        print(key.shape)
        # query = query / math.sqrt(math.sqrt(self.head_dim))
        # key = key / math.sqrt(math.sqrt(self.head_dim)) * attn_mask
        # value = value * attn_mask

        return self.attn_fn(query, key, value)