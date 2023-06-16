import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
import math
import logging


class AttentionPassthrough(nn.Module):
    def __init__(self):
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
        
        return query
