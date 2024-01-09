"""

Implements Flash Attention 

""" 
import logging
import torch
from torch import nn 
from flash_attn import flash_attn_func


class FlashAttention(torch.nn.Module):
    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout_rate = dropout_rate
        # self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, pos_emb=None, mask=None):
        # query B x L x D
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        orig = None 
        if q.dtype not in [torch.bfloat16]:
            orig = q.dtype
            q, k, v = [x.to(torch.bfloat16) for x in [q, k, v]]
        
        outputs = flash_attn_func(q,k,v,dropout_p=self.dropout_rate)

        if orig is not None:
            outputs = outputs.to(orig)

        return outputs.view(n_batch, -1, self.h * self.d_k)


# 