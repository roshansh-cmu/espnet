#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright NLPLab (https://github.com/OpenNLPLab/cosFormer/blob/main)
# Apache 2.0 (https://github.com/OpenNLPLab/cosFormer/blob/main/LICENSE)

"""Cosine former self-attention layer definition."""


import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
import math


class CosformerAttention(nn.Module):
    """
    Cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        cos_reweight=False,
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = (
            self.get_act_fun(act_fun)
            if act_fun != "softmax"
            else torch.nn.Softmax(dim=-1)
        )
        self.act_fun_type = act_fun
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal
        ##
        self.reweight = cos_reweight

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

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
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = (
            self.act_fun(q.view(-1, bsz, n_head, head_dim))
            if self.act_fun_type != "elu"
            else 1 + self.act_fun(q.view(-1, bsz, n_head, head_dim))
        )
        k = (
            self.act_fun(k.view(-1, bsz, n_head, head_dim))
            if self.act_fun_type != "elu"
            else 1 + self.act_fun(q.view(-1, bsz, n_head, head_dim))
        )

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        if self.reweight:
            q_ = torch.cat(
                [
                    q * torch.sin(weight_index[:, :tgt_len, :] / m),
                    q * torch.cos(weight_index[:, :tgt_len, :] / m),
                ],
                dim=-1,
            )
            # (N * h, S, 2 * d)
            k_ = torch.cat(
                [
                    k * torch.sin(weight_index[:, :src_len, :] / m),
                    k * torch.cos(weight_index[:, :src_len, :] / m),
                ],
                dim=-1,
            )
        else:
            q_ = q
            # (N * h, S, 2 * d)
            k_ = k

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)

    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query

        n_head = self.n_head
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // n_head

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [
                q * torch.sin(weight_index[:, :tgt_len, :] / m),
                q * torch.cos(weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [
                k * torch.sin(weight_index[:, :src_len, :] / m),
                k * torch.cos(weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask == float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output


class NormAttention(nn.Module):
    """
    Norm attention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate=0.0,
        act_fun="relu",
        kdim=None,
        vdim=None,
        causal=False,
        has_outproj=True,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.linear_k = nn.Linear(self.kdim, n_feat)
        self.linear_v = nn.Linear(self.vdim, n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    # def get_index(self, seq_len):
    #     index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

    #     return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "sig":
            return F.sigmoid
        elif act_fun == "swish":
            return F.silu

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
        if key == None:
            key = query
        if value == None:
            value = query
        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head
        query = query.view(tgt_len, bsz, embed_dim)
        key = key.view(src_len, bsz, embed_dim)
        value = value.view(src_len, bsz, embed_dim)

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv  # / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        else:
            # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            # z_ = 1 / torch.clamp_min(
            #     torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            # )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)

    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
                query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension.
                key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
                E is the embedding dimension.
                attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query

        n_head = self.n_head
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // n_head

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        q_ = q
        k_ = k
        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask == float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output


# def test(batch=2, tgt_len=10, src_len=20, embed_dim=128, n_head=8, N=100, causal=False):
#     model = CosformerAttention(embed_dim=embed_dim, n_head=n_head, causal=causal)
#     diff = 0
#     if causal:
#         mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float("-inf"))
#     else:
#         mask = None
#     for i in range(N):
#         query = torch.rand(tgt_len, batch, embed_dim)
#         key = torch.rand(src_len, batch, embed_dim)
#         value = torch.rand(src_len, batch, embed_dim)
#         left_res = model.left_product(query, key, value, mask)
#         right_res = model(query, key, value)
#         diff += torch.norm(left_res - right_res)
#     diff /= N

#     if causal:
#         print("Test result for causal model:")
#     else:
#         print("Test result for bidirectional model:")
#     print(f"The error of left multiplication and right multiplication is {diff}")


# def main():
#     test(tgt_len=10, src_len=20, causal=False)
#     test(tgt_len=10, src_len=10, causal=True)


# if __name__ == "__main__":
#     main()
