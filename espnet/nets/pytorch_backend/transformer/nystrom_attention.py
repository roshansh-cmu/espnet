import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
import math
import logging

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


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


class NystromAttention(torch.nn.Module):

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        act_fun: str,
    ):
        super().__init__()


    def forward(        
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):

        pass