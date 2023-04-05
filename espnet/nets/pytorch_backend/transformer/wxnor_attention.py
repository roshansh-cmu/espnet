#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Roshan Sharma (Carnegie Mellon University)
# Apache 2.0

"""X-NOR self-attention layer definition."""


import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
import math
import logging
from espnet.nets.pytorch_backend.transformer.xnor_attention import XNorAttention


class WeightedXNorCosAttention(XNorAttention):
    """
    XNorcos attention
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
        super().__init__(n_head, n_feat)
        self.weights = torch.nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        self.register_parameter("xnor_weights", self.weights)

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)

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
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # n_batch_pos = pos_emb.size(0)
        # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        q_ = torch.cat(
            [
                q_ * torch.sin(weight_index[:, :tgt_len, :] / m),
                q_ * torch.cos(weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        # (N * h, S, 2 * d)
        comp_q_ = torch.cat(
            [
                comp_q_ * torch.sin(weight_index[:, :tgt_len, :] / m),
                comp_q_ * torch.cos(weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        k_ = torch.cat(
            [
                k_ * torch.sin(weight_index[:, :src_len, :] / m),
                k_ * torch.cos(weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )
        comp_k_ = torch.cat(
            [
                comp_k_ * torch.sin(weight_index[:, :src_len, :] / m),
                comp_k_ * torch.cos(weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
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
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output = (F.relu(self.weights[0]) * attn_output) + (
                F.relu(self.weights[1]) * attn_output2
            )
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class WeightedXNorWeightedCosAttention(XNorAttention):
    """
    XNorcos attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
        self.weights = torch.nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        self.alpha = torch.nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)

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
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # n_batch_pos = pos_emb.size(0)
        # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        q_ = torch.cat(
            [
                q_ * torch.sin(self.alpha[0] * weight_index[:, :tgt_len, :] / m),
                q_ * torch.cos(self.alpha[0] * weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        # (N * h, S, 2 * d)
        comp_q_ = torch.cat(
            [
                comp_q_ * torch.sin(self.alpha[1] * weight_index[:, :tgt_len, :] / m),
                comp_q_ * torch.cos(self.alpha[1] * weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        k_ = torch.cat(
            [
                k_ * torch.sin(self.alpha[0] * weight_index[:, :src_len, :] / m),
                k_ * torch.cos(self.alpha[0] * weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )
        comp_k_ = torch.cat(
            [
                comp_k_ * torch.sin(self.alpha[1] * weight_index[:, :src_len, :] / m),
                comp_k_ * torch.cos(self.alpha[1] * weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
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
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output = (F.relu(self.weights[0]) * attn_output) + (
                F.relu(self.weights[1]) * attn_output2
            )
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class WeightedXNorAttention(XNorAttention):
    """
    Weighted XNorm attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)

        self.weights = torch.nn.Parameter(torch.FloatTensor([1.0, 0.5]))
        self.register_parameter("xnor_weights", self.weights)

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
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
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
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output = (F.relu(self.weights[0]) * attn_output) + (
                F.relu(self.weights[1]) * attn_output2
            )  # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
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


class GatedXNorAttention(XNorAttention):
    """
    Gated XNorm attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)

        size = n_feat // n_head
        self.gater = torch.nn.Linear(2 * size, 2)
        # self.weights = torch.nn.Parameter(torch.FloatTensor([1.0, 0.5]))
        # self.register_parameter("xnor_weights",self.weights)

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
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum("nld,nlm->ndm", k_, v)
        comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z_ = 1 / torch.clamp_min(
            torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
        )
        comp_z_ = 1 / torch.clamp_min(
            torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
        )
        # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

        # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
        attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

        # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
        concatenated_output = torch.cat((attn_output, attn_output2), dim=-1)

        gating_weights = torch.sigmoid(self.gater(concatenated_output))
        wt_1 = gating_weights[:, :, 0].unsqueeze(-1).repeat(1, 1, attn_output.shape[-1])
        wt_2 = gating_weights[:, :, 1].unsqueeze(-1).repeat(1, 1, attn_output.shape[-1])
        # logging.warning(f"gating_weights: {gating_weights.shape} {attn_output.shape} {attn_output2.shape} {wt_2.shape} {wt_1.shape}")

        attn_output = wt_1 * attn_output + wt_2 * attn_output2

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class GatedXNorNoDenomAttention(XNorAttention):
    """
    Gated XNorm No Denom attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)

        size = n_feat // n_head
        self.gater = torch.nn.Linear(2 * size, 2)
        # self.weights = torch.nn.Parameter(torch.FloatTensor([1.0, 0.5]))
        # self.register_parameter("xnor_weights",self.weights)

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
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum("nld,nlm->ndm", k_, v)
        comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z = 1 / torch.clamp_min(
            torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
        )
        # comp_z_ = 1 / torch.clamp_min(
        #   torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
        # )
        # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
        z_ = torch.ones_like(z)
        comp_z_ = torch.ones_like(z)

        attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
        if torch.isnan(attn_output).any():
            logging.warning(f"z_ has NAN")

        # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
        attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

        # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
        concatenated_output = torch.cat((attn_output, attn_output2), dim=-1)

        gating_weights = torch.sigmoid(self.gater(concatenated_output))
        wt_1 = gating_weights[:, :, 0].unsqueeze(-1).repeat(1, 1, attn_output.shape[-1])
        wt_2 = gating_weights[:, :, 1].unsqueeze(-1).repeat(1, 1, attn_output.shape[-1])
        # logging.warning(f"gating_weights: {gating_weights.shape} {attn_output.shape} {attn_output2.shape} {wt_2.shape} {wt_1.shape}")

        attn_output = wt_1 * attn_output + wt_2 * attn_output2

        ## Normalize weights
        denom = wt_1 + wt_2
        denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
        # attn_output = attn_output / denom

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class TermGatedXNorAttention(XNorAttention):
    """
    Term Gated XNorm attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
        size = n_feat // n_head
        self.gater = torch.nn.Linear(2 * size, 2)

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

        q = torch.nn.functional.softmax(q.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum("nld,nlm->ndm", k_, v)
        comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z_ = 1 / torch.clamp_min(
            torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
        )
        comp_z_ = 1 / torch.clamp_min(
            torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
        )
        # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

        # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
        attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

        # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
        concatenated_output = torch.cat((attn_output, attn_output2), dim=-1)
        concatenated_output = torch.mean(concatenated_output, dim=1)
        gating_weights = (
            F.sigmoid(self.gater(concatenated_output))
            .unsqueeze(1)
            .repeat(1, attn_output.shape[1], 1)
        )

        wt_1 = gating_weights[:, :, 0].unsqueeze(-1).repeat(1, 1, attn_output.shape[-1])
        wt_2 = gating_weights[:, :, 1].unsqueeze(-1).repeat(1, 1, attn_output.shape[-1])
        # logging.warning(f"gating_weights: {gating_weights.shape} {attn_output.shape} {attn_output2.shape}")
        attn_output = wt_1 * attn_output + wt_2 * attn_output2

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class WeightedXNorSigmoidAttention(XNorAttention):
    """
    Weighted XNorm attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)

        self.weights = torch.nn.Parameter(torch.FloatTensor([1.0, 0.5]))
        self.register_parameter("xnor_weights", self.weights)

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
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.sigmoid(q.view(tgt_len, bsz, n_head, head_dim))
        comp_q = 1 - q
        k = torch.nn.functional.sigmoid(k.view(tgt_len, bsz, n_head, head_dim))
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
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
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output = (F.relu(self.weights[0]) * attn_output) + (
                F.relu(self.weights[1]) * attn_output2
            )  # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class MultiWeightedXNorAttention(XNorAttention):
    """
    Weighted XNorm attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(n_feat // n_head)) for _ in range(2)]
        )
        self.register_parameter("xnor_weights_1", self.weights[0])
        self.register_parameter("xnor_weights_2", self.weights[1])

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
        # q = self.act_fun(q)
        # k = self.act_fun(k)

        q = torch.nn.functional.softmax(q.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(tgt_len, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
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
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output = (self.weights[0].unsqueeze(0).unsqueeze(1) * attn_output) + (
                self.weights[1].unsqueeze(0).unsqueeze(1) * attn_output2
            )

            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class WeightedXNorRopeAttention(XNorAttention):
    """
    XNorm attention
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
        zero_triu=False,
    ):
        super().__init__(n_head, n_feat)
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
        # ROPE Encoding
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            2048, n_feat // n_head
        )
        self.weights = torch.nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

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

        sinusoidal_pos = self.embed_positions(
            query.permute(1, 0, 2).view(bsz, -1, n_head, head_dim).shape[:-1]
        )[None, None, :, :]

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_q = 1 - q
        k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
        comp_k = 1 - k

        q, k = self.apply_rotary_position_embeddings(
            sinusoidal_pos, q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3)
        )
        comp_q, comp_k = self.apply_rotary_position_embeddings(
            sinusoidal_pos, comp_q.permute(1, 2, 0, 3), comp_k.permute(1, 2, 0, 3)
        )
        q, k = q.permute(2, 0, 1, 3), k.permute(2, 0, 1, 3)
        comp_q, comp_k = comp_q.permute(2, 0, 1, 3), comp_k.permute(2, 0, 1, 3)

        # multihead reshape
        # (N * h, L, d)
        q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
        comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nlm", k_, v)
            comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
            qkv += comp_qkv
            # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
            # k_cum = torch.cumsum(k_, dim=1)
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
            comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            comp_z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
            )
            # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

            # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
            attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

            # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
            attn_output = (F.relu(self.weights[0]) * attn_output) + (
                F.relu(self.weights[1]) * attn_output2
            )
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)
