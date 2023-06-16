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
        n_head: int,
        n_feat: int,
        n_landmarks: int,
        kernel_size: int,
        dropout_rate: float,
        act_fun: str,
    ):
        super().__init__()
        self.head_dim = n_feat
        self.num_head = n_head

        self.num_landmarks = n_landmarks
        self.seq_len = size

        self.use_conv = kernel_size is not None
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (kernel_size, 1), padding = (kernel_size // 2, 0),
                bias = False,
                groups = self.num_head)

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        # This original implementation is more conservative to compute coefficient of Z_0.
        V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'

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
        print(value.shape)
        print(query)
        print(key)
        print(value)
        print(attn_mask.shape)
        print(attn_mask)
        
        attn_mask = attn_mask.permute(0,2,1).int()
        query = query * attn_mask / math.sqrt(math.sqrt(self.head_dim))
        key = key * attn_mask / math.sqrt(math.sqrt(self.head_dim))

        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(query, key.transpose(-1, -2)) - 1e9 * (1 - attn_mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, value)
        else:
            query_landmarks = query.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            key_landmarks = key.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(query, key_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(query_landmarks, key_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(query_landmarks, key.transpose(-1, -2)) - 1e9 * (1 - attn_mask[:, None, None, :]), dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, value))
 
        if self.use_conv:
            X += self.conv(value * attn_mask[:, None, :, None])

        return X