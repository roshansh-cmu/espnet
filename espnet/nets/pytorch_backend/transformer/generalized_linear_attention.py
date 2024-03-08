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



# @staticmethod
# def apply_rotary_position_embeddings(
#     sinusoidal_pos, query_layer, key_layer, value_layer=None
# ):
#     # https://kexue.fm/archives/8265
#     # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#     # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#     sin, cos = sinusoidal_pos.chunk(2, dim=-1)
#     # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#     sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
#     # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#     cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
#     # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
#     rotate_half_query_layer = torch.stack(
#         [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
#     ).reshape_as(query_layer)
#     query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
#     # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
#     rotate_half_key_layer = torch.stack(
#         [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
#     ).reshape_as(key_layer)
#     key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
#     if value_layer is not None:
#         # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
#         rotate_half_value_layer = torch.stack(
#             [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
#         ).reshape_as(value_layer)
#         value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
#         return query_layer, key_layer, value_layer
#     return query_layer, key_layer




# class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
#     """This module produces sinusoidal positional embeddings of any length."""

#     def __init__(
#         self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
#     ) -> None:
#         super().__init__(num_positions, embedding_dim)
#         self.weight = self._init_weight(self.weight)

#     @staticmethod
#     def _init_weight(out: nn.Parameter) -> nn.Parameter:
#         """
#         Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
#         the 2nd half of the vector. [dim // 2:]
#         """
#         n_pos, dim = out.shape
#         position_enc = np.array(
#             [
#                 [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
#                 for pos in range(n_pos)
#             ]
#         )
#         out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
#         sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
#         out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
#         out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
#         out.detach_()
#         return out

#     @torch.no_grad()
#     def forward(
#         self, input_ids_shape: torch.Size, past_key_values_length: int = 0
#     ) -> torch.Tensor:
#         """`input_ids_shape` is expected to be [bsz x seqlen]."""
#         bsz, seq_len = input_ids_shape[:2]
#         positions = torch.arange(
#             past_key_values_length,
#             past_key_values_length + seq_len,
#             dtype=torch.long,
#             device=self.weight.device,
#         )
#         return super().forward(positions)

# class RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     @property
#     def sin_cached(self):
#         # logger.warning_once(
#         #     "The sin_cached attribute will be removed in 4.40. Bear in mind that its contents changed in v4.38. Use "
#         #     "the forward method of RoPE from now on instead."
#         # )
#         return self._sin_cached

#     @property
#     def cos_cached(self):
#         # logger.warning_once(
#         #     "The cos_cached attribute will be removed in 4.40. Bear in mind that its contents changed in v4.38. Use "
#         #     "the forward method of RoPE from now on instead."
#         # )
#         return self._cos_cached

#     def forward(self, x, position_ids, seq_len=None):
#         # if seq_len is not None:
#         #     logger.warning_once("The `seq_len` argument is deprecated and unused. It will be removed in v4.40.")

#         # x: [bs, num_attention_heads, seq_len, head_size]
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()
#         freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         cos = emb.cos().to(dtype=x.dtype)
#         sin = emb.sin().to(dtype=x.dtype)
#         # backwards compatibility
#         self._cos_cached = cos
#         self._sin_cached = sin
#         return cos, sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = torch.arange(seq_len).float()
        # [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = torch.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return cos, sin
        # return (
        #     cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
        #     sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        # )

        
# class LinearScalingRotaryEmbedding(RotaryEmbedding):
#     """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
#         self.scaling_factor = scaling_factor
#         super().__init__(dim, max_position_embeddings*scaling_factor, base, device)

#     def forward(self, x, position_ids, seq_len=None):
#         # difference to the original RoPE: a scaling factor is aplied to the position ids
#         position_ids = position_ids.float() / self.scaling_factor
#         cos, sin = super().forward(x, position_ids, seq_len)
#         return cos, sin

class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings*scaling_factor, base, device)


    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = torch.arange(seq_len).float()
        t = t / self.scaling_factor
        # [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = torch.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]
        
        
        
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def forward(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids, seq_len)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     print("cos")
#     print(cos.shape)
    
#     cos = cos.squeeze(0)  # [seq_len, dim]
#     sin = sin.squeeze(0)  # [seq_len, dim]
#     print("position_ids")
#     print(position_ids.shape)
#     print("cos")
#     print(cos.shape)

#     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
#     k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
#     return q_embed.to(q.dtype), k_embed.to(k.dtype)

#     # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
#     # sin = sin[position_ids].unsqueeze(unsqueeze_dim)


#     # q_embed = (q * cos) + (rotate_half(q) * sin)
#     # k_embed = (k * cos) + (rotate_half(k) * sin)
#     # return q_embed, k_embed


# class RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     @property
#     def sin_cached(self):
#         return self._sin_cached

#     @property
#     def cos_cached(self):
#         return self._cos_cached

#     def forward(self, x, position_ids, seq_len=None):

#         # x: [bs, num_attention_heads, seq_len, head_size]
        
#         print("RotaryEmbedding")
#         print(x.shape)
#         print(position_ids.shape)
           
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, x.size(2))
#         position_ids_expanded = position_ids[:, None, :].float().cuda()
#         print(inv_freq_expanded.shape)
        
#         print(position_ids_expanded.shape)
        
#         freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         cos = emb.cos().to(dtype=x.dtype)
#         sin = emb.sin().to(dtype=x.dtype)
#         # backwards compatibility
#         self._cos_cached = cos
#         self._sin_cached = sin
#         return cos, sin
#     # def forward(self, x, position_ids, seq_len=None):
#     #     # x: [bs, num_attention_heads, seq_len, head_size]
#     #     print(x.shape)
#     #     print(position_ids.shape)
           
#     #     inv_freq_expanded = self.inv_freq[None, :, None].expand(-1, -1, position_ids.shape[1])
#     #     print(inv_freq_expanded.shape)
        
#     #     position_ids_expanded = position_ids[:, None, :].float().cuda()
#     #     print(position_ids_expanded.shape)
        
#     #     # freqs = torch.einsum('bij,bkj->bik', inv_freq_expanded, position_ids_expanded)
#     #     # emb = torch.cat((freqs, freqs), dim=-1)
#     #     # cos = emb.cos().to(dtype=x.dtype)
#     #     # sin = emb.sin().to(dtype=x.dtype)
        
#     #     freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
#     #     emb = torch.cat((freqs, freqs), dim=-1)
#     #     cos = emb.cos().to(dtype=x.dtype)
#     #     sin = emb.sin().to(dtype=x.dtype)
#     #     # backwards compatibility
#     #     self._cos_cached = cos
#     #     self._sin_cached = sin
#     #     return cos, sin
        


def apply_rotary_pos_emb(q, k, cos, sin):

    # if position_ids is None:
    #     # Note: Only for LlamaForCausalLMPipe model pretraining
    #     cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    #     sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    # else:
    #     cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
    #     sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
    #     cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    #     sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    
    # Note: Only for LlamaForCausalLMPipe model pretraining
    cos = cos[:, : q.shape[1], :, :].cuda()  # [bs, seq_len, 1, dim]
    sin = sin[:, : q.shape[1], :, :] .cuda() # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
        

        

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


class GeneralizedXNorAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.2,
        act_fun: str = "softmax",
        kdim: int = None,
        vdim: int = None,
        has_outproj: bool = True,
        cos_reweight: bool = False,
        learn_weight: bool = False,
        learn_gate: bool = False,
        add_rope: bool = False,
        add_pi: bool = False,
        max_length: int = 8192,
        base_length:int = 2048,
        formulation: str = "generalized",
        nterms: int = 2,
    ):
        super().__init__()

        assert formulation in ["xnor", "generalized"]

        self.n_feat = n_feat
        self.kdim = kdim if kdim is not None else n_feat
        self.vdim = vdim if kdim is not None else n_feat
        self.n_head = n_head
        self.has_outproj = has_outproj
        size = n_feat // n_head

        self.act_fun = get_act_fun(act_fun)

        # q, k, v projection
        self.linear_v = nn.Linear(self.vdim, n_feat)

        if formulation == "xnor":
            self.linear_k = torch.nn.ModuleList([nn.Linear(self.kdim, n_feat)])
            self.linear_q = torch.nn.ModuleList([nn.Linear(n_feat, n_feat)])
        elif formulation == "generalized":
            self.linear_k = torch.nn.ModuleList(
                [nn.Linear(self.kdim, n_feat) for _ in range(nterms)]
            )
            self.linear_q = torch.nn.ModuleList(
                [nn.Linear(n_feat, n_feat) for _ in range(nterms)]
            )

        # outprojection
        self.linear_out = nn.Linear(n_feat, n_feat)
        # dropout rate
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.formulation = formulation

        self.cos_reweight = cos_reweight
        self.learn_weight = learn_weight
        self.add_rope = add_rope
        self.add_pi = add_pi
        self.max_length = max_length
        self.head_dim = n_feat // n_head
        self.rope_theta = 1000000

        nterms = 2 if formulation == "xnor" else nterms

        if self.learn_weight:
            self.weights = torch.nn.Parameter(torch.FloatTensor([1.0] * nterms))
            self.register_parameter("xnor_weights", self.weights)
        
        if self.add_rope:
            if add_pi:
            
                scaling_factor = max_length // base_length
                self.rotary_emb = LinearScalingRotaryEmbedding(
                        n_feat // n_head,
                        max_position_embeddings=self.max_length,
                        scaling_factor=scaling_factor,
                        base=self.rope_theta,
                    )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_length,
                    base=self.rope_theta,
                )


        self.learn_gate = learn_gate

        if self.learn_gate:
            self.gater = torch.nn.Linear(nterms * size, nterms)
        

        assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)

    def insert_cosine_posenc(self, queries, keys, src_len, tgt_len):
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(queries[-1])
        queries = [
            torch.cat(
                [
                    q_ * torch.sin(weight_index[:, :tgt_len, :] / m),
                    q_ * torch.cos(weight_index[:, :tgt_len, :] / m),
                ],
                dim=-1,
            )
            for q_ in queries
        ]
        keys = [
            torch.cat(
                [
                    k_ * torch.sin(weight_index[:, :src_len, :] / m),
                    k_ * torch.cos(weight_index[:, :src_len, :] / m),
                ],
                dim=-1,
            )
            for k_ in keys
        ]

        return queries, keys

    def add_rope_posenc(self, queries, keys,values, src_len, tgt_len):
        """
        value: (N,S,h,d)

        query: ## (L, N, h, d) -> (N,h,L,d)
        """

        bsz,src_len,n_head,head_dim = values.shape
        values = values.permute(0,2,1,3)

        # attention_mask = 
        # position_ids = attention_mask.long().cumsum(-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, 1)
        #FIXME: Change this to correct shapes 
        # position_ids = torch.arange(queries[0].shape[1]).unsqueeze(0).unsqueeze(0)
     
        # [seq_len, dim]
        
        # position_ids = torch.arange(0, src_len).expand((bsz, src_len))
      
        # # position_ids = torch.arange(0, src_len).unsqueeze(-1).expand(-1, head_dim).unsqueeze(0)
        # print("position_ids.shape")
        # print(position_ids.shape)
        


   
        cos, sin = self.rotary_emb(values, src_len)
        # batch size , number of heads, sequence length, head dimensions 
        # print(cos.shape)
        
        # print("#"*4)
        # outputs = [apply_rotary_pos_emb(query_state, key_state, cos, sin) for (query_state,key_state) in zip(queries,keys)]

        outputs = [apply_rotary_pos_emb(query_state.permute(1,2,0,3), key_state.permute(1,2,0,3), cos, sin) for (query_state,key_state) in zip(queries,keys)]
        
        query_states, key_states = [x[0].permute(2,0,1,3) for x in outputs],[x[1].permute(2,0,1,3) for x in outputs]
        # query_states, key_states = [x[0] for x in outputs],[x[1]  for x in outputs]

        return query_states, key_states


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

        n_head = self.n_head
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // n_head

        value = self.linear_v(self.dropout(value)).view(
            bsz, src_len, n_head, head_dim
        )  ## (S, N, h, d)

        queries = [
            self.linear_q[i](self.dropout(query))
            .view(bsz, tgt_len, n_head, head_dim)
            .permute(1, 0, 2, 3)
            for i in range(len(self.linear_q))
        ]  ## (L, N, h, d)

        keys = [
            self.linear_k[i](self.dropout(key))
            .view(bsz, src_len, n_head, head_dim)
            .permute(1, 0, 2, 3)
            for i in range(len(self.linear_k))
        ]  ## (S, N, h, d)

        if self.act_fun == F.softmax:
            queries = [self.act_fun(query, dim=-1) for query in queries]
            keys = [self.act_fun(key, dim=0) for key in keys]
        else:
            queries = [self.act_fun(query) for query in queries]
            keys = [self.act_fun(key) for key in keys]

        if self.formulation == "xnor":
            queries.append(1 - queries[-1])
            keys.append(1 - keys[-1])

        if self.add_rope:
            queries,keys = self.add_rope_posenc(queries,keys,value,src_len,tgt_len)

        queries = [
            query.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
            for query in queries
        ]
        keys = [
            key.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
            for key in keys
        ]
        v = value.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)


        # Cosine Reweighting

        if self.cos_reweight:
            queries, keys = self.insert_cosine_posenc(queries, keys, src_len, tgt_len)
        
        

        ## TODO: Add Causal Version if needed
        kv_terms = [torch.einsum("nld,nlm->ndm", k_, v) for k_ in keys]
        if self.act_fun != F.softmax:
            z_terms = [
                1
                / torch.clamp_min(
                    torch.einsum("nld,nd->nl", self.dropout(q_), torch.sum(k_, axis=1)),
                    eps,
                )
                for q_, k_ in zip(queries, keys)
            ]
            output_terms = [
                torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
                for q_, kv_, z_ in zip(queries, kv_terms, z_terms)
            ]
        else:
            output_terms = [
                torch.einsum("nld,ndm->nlm", q_, kv_)
                for q_, kv_ in zip(queries, kv_terms)
            ]

        if self.learn_weight:
            normalized_term_weights = [
                float(self.weights[i] / (1e-16 + sum(self.weights)))
                for i in range(len(self.weights))
            ]
            output_terms = [
                normalized_term_weights[i] * output_term
                for i, output_term in enumerate(output_terms)
            ]
        elif self.learn_gate:
            concatenated_output = torch.cat(output_terms, dim=-1)
            gating_weights = torch.sigmoid(self.gater(concatenated_output))
            term_weights = [
                gating_weights[:, :, i]
                .unsqueeze(-1)
                .repeat(1, 1, output_terms[0].shape[-1])
                for i in range(gating_weights.shape[-1])
            ]
            normalized_term_weights = [
                term_weights[i] / (sum(term_weights) + 1e-16)
                for i in range(len(term_weights))
            ]
            output_terms = [
                normalized_term_weights[i] * output_term
                for i, output_term in enumerate(output_terms)
            ]
        attn_output = sum(output_terms)

        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.linear_out(attn_output)

        return attn_output.view(bsz, tgt_len, self.n_feat)


# class XNorCosAttention(nn.Module):
#     """
#     XNorcos attention
#     """

#     def __init__(
#         self,
#         n_head,
#         n_feat,
#         dropout_rate=0.2,
#         act_fun="relu",
#         kdim=None,
#         vdim=None,
#         causal=False,
#         has_outproj=True,
#         zero_triu=False,
#     ):
#         super().__init__()
#         self.n_feat = n_feat
#         self.kdim = kdim if kdim is not None else n_feat
#         self.vdim = vdim if kdim is not None else n_feat
#         self.n_head = n_head
#         self.has_outproj = has_outproj
#         self.act_fun = self.get_act_fun(act_fun)
#         # q, k, v projection
#         self.linear_k = nn.Linear(self.kdim, n_feat)
#         self.linear_v = nn.Linear(self.vdim, n_feat)
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         # outprojection
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         # dropout rate
#         self.dropout_rate = dropout_rate
#         # causal
#         self.causal = causal
#         self.dropout = torch.nn.Dropout(dropout_rate)

#         assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

#     def get_index(self, seq_len):
#         index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

#         return nn.Parameter(index, requires_grad=False)

#     def get_act_fun(self, act_fun):
#         if act_fun == "relu":
#             return F.relu
#         elif act_fun == "elu":
#             return 1 + F.elu
#         elif act_fun == "sig":
#             return F.sigmoid
#         elif act_fun == "swish":
#             return F.silu

#     def forward(
#         self,
#         query: Tensor,
#         key: Optional[Tensor] = None,
#         value: Optional[Tensor] = None,
#         pos_emb: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         eps: Optional[float] = 1e-6,
#     ):
#         """Input shape: Sequence x Batch x Embedding
#         Args:
#                 query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
#                 where the mask prevents the attention from looking forward in time (default: None).
#         """
#         if key == None:
#             key = query
#         if value == None:
#             value = query
#         n_head = self.n_head
#         bsz, tgt_len, embed_dim = query.size()
#         src_len = key.size(1)
#         head_dim = embed_dim // n_head
#         query = query.permute(1, 0, 2)  # query.view(tgt_len, bsz, embed_dim)
#         key = key.permute(1, 0, 2)  # key.view(src_len, bsz, embed_dim)
#         value = value.permute(1, 0, 2)  # value.view(src_len, bsz, embed_dim)

#         # get q, k, v
#         # (L, N, E)
#         q = self.linear_q(query)
#         # (S, N, E)
#         k = self.linear_k(key)
#         # (S, N, E)
#         v = self.linear_v(value)

#         q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
#         comp_q = 1 - q
#         k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
#         comp_k = 1 - k

#         # multihead reshape
#         # (N * h, L, d)
#         q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         m = max(src_len, tgt_len)
#         # get index and send to cuda
#         weight_index = self.get_index(m).to(q)
#         q_ = torch.cat(
#             [
#                 q_ * torch.sin(weight_index[:, :tgt_len, :] / m),
#                 q_ * torch.cos(weight_index[:, :tgt_len, :] / m),
#             ],
#             dim=-1,
#         )
#         # (N * h, S, 2 * d)
#         comp_q_ = torch.cat(
#             [
#                 comp_q_ * torch.sin(weight_index[:, :tgt_len, :] / m),
#                 comp_q_ * torch.cos(weight_index[:, :tgt_len, :] / m),
#             ],
#             dim=-1,
#         )
#         k_ = torch.cat(
#             [
#                 k_ * torch.sin(weight_index[:, :src_len, :] / m),
#                 k_ * torch.cos(weight_index[:, :src_len, :] / m),
#             ],
#             dim=-1,
#         )
#         comp_k_ = torch.cat(
#             [
#                 comp_k_ * torch.sin(weight_index[:, :src_len, :] / m),
#                 comp_k_ * torch.cos(weight_index[:, :src_len, :] / m),
#             ],
#             dim=-1,
#         )

#         # (N * h, S, d)
#         v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         if self.causal:
#             ## Need to improve speed!
#             # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->nlm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
#             # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
#             kv_cum = torch.cumsum(kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
#             comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
#             # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
#             # k_cum = torch.cumsum(k_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
#             # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
#             # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
#             attn_output = qkv + comp_qkv  # / denom.unsqueeze(-1)
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         else:
#             # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->ndm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
#             # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
#             z_ = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", self.dropout(q_), torch.sum(k_, axis=1)), eps
#             )
#             comp_z_ = 1 / torch.clamp_min(
#                 torch.einsum(
#                     "nld,nd->nl", self.dropout(comp_q_), torch.sum(comp_k_, axis=1)
#                 ),
#                 eps,
#             )
#             # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
#             attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

#             # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
#             attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

#             # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
#             attn_output += attn_output2
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         # L, N, E
#         if self.has_outproj:
#             attn_output = self.linear_out(attn_output)

#         return attn_output.view(bsz, tgt_len, self.n_feat)


# class SoftmaxAttention(nn.Module):
#     """
#     Softmax attention
#     """

#     def __init__(
#         self,
#         n_head,
#         n_feat,
#         dropout_rate=0.0,
#         act_fun="relu",
#         kdim=None,
#         vdim=None,
#         causal=False,
#         has_outproj=True,
#         zero_triu=False,
#     ):
#         super().__init__()
#         self.n_feat = n_feat
#         self.kdim = kdim if kdim is not None else n_feat
#         self.vdim = vdim if kdim is not None else n_feat
#         self.n_head = n_head
#         self.has_outproj = has_outproj
#         self.act_fun = self.get_act_fun(act_fun)
#         # q, k, v projection
#         self.linear_k = nn.Linear(self.kdim, n_feat)
#         self.linear_v = nn.Linear(self.vdim, n_feat)
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         # outprojection
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         # dropout rate
#         self.dropout_rate = dropout_rate
#         # causal
#         self.causal = causal

#         assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

#     def get_act_fun(self, act_fun):
#         if act_fun == "relu":
#             return F.relu
#         elif act_fun == "elu":
#             return 1 + F.elu
#         elif act_fun == "sig":
#             return F.sigmoid
#         elif act_fun == "swish":
#             return F.silu

#     def forward(
#         self,
#         query: Tensor,
#         key: Optional[Tensor] = None,
#         value: Optional[Tensor] = None,
#         pos_emb: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         eps: Optional[float] = 1e-6,
#     ):
#         """Input shape: Sequence x Batch x Embedding
#         Args:
#                 query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
#                 where the mask prevents the attention from looking forward in time (default: None).
#         """
#         if key == None:
#             key = query
#         if value == None:
#             value = query
#         n_head = self.n_head
#         bsz, tgt_len, embed_dim = query.size()
#         src_len = key.size(1)
#         head_dim = embed_dim // n_head
#         query = query.view(tgt_len, bsz, embed_dim)
#         key = key.view(src_len, bsz, embed_dim)
#         value = value.view(src_len, bsz, embed_dim)

#         # get q, k, v
#         # (L, N, E)
#         q = self.linear_q(query)
#         # (S, N, E)
#         k = self.linear_k(key)
#         # (S, N, E)
#         v = self.linear_v(value)

#         q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
#         k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)

#         # multihead reshape
#         # (N * h, L, d)
#         q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         if self.causal:
#             ## Need to improve speed!
#             # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->nlm", k_, v)
#             # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
#             kv_cum = torch.cumsum(kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
#             # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
#             # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
#             attn_output = qkv  # / denom.unsqueeze(-1)
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         else:
#             # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->ndm", k_, v)
#             # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
#             z_ = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
#             )
#             # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
#             attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         # L, N, E
#         if self.has_outproj:
#             attn_output = self.linear_out(attn_output)

#         return attn_output.view(bsz, tgt_len, self.n_feat)


# class XNorAttention(nn.Module):
#     """
#     XNorm attention
#     """

#     def __init__(
#         self,
#         n_head,
#         n_feat,
#         dropout_rate=0.0,
#         act_fun="relu",
#         kdim=None,
#         vdim=None,
#         causal=False,
#         has_outproj=True,
#         zero_triu=False,
#     ):
#         super().__init__()
#         self.n_feat = n_feat
#         self.kdim = kdim if kdim is not None else n_feat
#         self.vdim = vdim if kdim is not None else n_feat
#         self.n_head = n_head
#         self.has_outproj = has_outproj
#         self.act_fun = self.get_act_fun(act_fun)
#         # q, k, v projection
#         self.linear_k = nn.Linear(self.kdim, n_feat)
#         self.linear_v = nn.Linear(self.vdim, n_feat)
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         # outprojection
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         # dropout rate
#         self.dropout_rate = dropout_rate
#         # causal
#         self.causal = causal

#         assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

#     def get_act_fun(self, act_fun):
#         if act_fun == "relu":
#             return F.relu
#         elif act_fun == "elu":
#             return 1 + F.elu
#         elif act_fun == "sig":
#             return F.sigmoid
#         elif act_fun == "swish":
#             return F.silu

#     @staticmethod
#     def apply_rotary_position_embeddings(
#         sinusoidal_pos, query_layer, key_layer, value_layer=None
#     ):
#         # https://kexue.fm/archives/8265
#         # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         sin, cos = sinusoidal_pos.chunk(2, dim=-1)
#         # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
#         # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
#         # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
#         rotate_half_query_layer = torch.stack(
#             [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
#         ).reshape_as(query_layer)
#         query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
#         # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
#         rotate_half_key_layer = torch.stack(
#             [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
#         ).reshape_as(key_layer)
#         key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
#         if value_layer is not None:
#             # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
#             rotate_half_value_layer = torch.stack(
#                 [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
#             ).reshape_as(value_layer)
#             value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
#             return query_layer, key_layer, value_layer
#         return query_layer, key_layer

#     def forward(
#         self,
#         query: Tensor,
#         key: Optional[Tensor] = None,
#         value: Optional[Tensor] = None,
#         pos_emb: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         eps: Optional[float] = 1e-6,
#     ):
#         """Input shape: Sequence x Batch x Embedding
#         Args:
#                 query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
#                 where the mask prevents the attention from looking forward in time (default: None).
#         """
#         if key == None:
#             key = query
#         if value == None:
#             value = query
#         n_head = self.n_head
#         bsz, tgt_len, embed_dim = query.size()
#         src_len = key.size(1)
#         head_dim = embed_dim // n_head
#         query = query.view(tgt_len, bsz, embed_dim)
#         key = key.view(src_len, bsz, embed_dim)
#         value = value.view(src_len, bsz, embed_dim)

#         # get q, k, v
#         # (L, N, E)
#         q = self.linear_q(query)
#         # (S, N, E)
#         k = self.linear_k(key)
#         # (S, N, E)
#         v = self.linear_v(value)

#         # activation

#         q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
#         comp_q = 1 - q
#         k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
#         comp_k = 1 - k

#         # multihead reshape
#         # (N * h, L, d)
#         q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # n_batch_pos = pos_emb.size(0)
#         # p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
#         # p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

#         # (N * h, S, d)
#         k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         if self.causal:
#             ## Need to improve speed!
#             # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->nlm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
#             # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
#             kv_cum = torch.cumsum(kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
#             comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
#             qkv += comp_qkv
#             # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
#             # k_cum = torch.cumsum(k_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
#             # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
#             # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
#             attn_output = qkv  # / denom.unsqueeze(-1)
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         else:
#             # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->ndm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
#             # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
#             z_ = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
#             )
#             comp_z_ = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
#             )
#             # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
#             attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

#             # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
#             attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

#             # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
#             attn_output += attn_output2
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         # L, N, E
#         if self.has_outproj:
#             attn_output = self.linear_out(attn_output)

#         return attn_output.view(bsz, tgt_len, self.n_feat)


# class ModifiedXNorAttention(XNorAttention):
#     """
#     XNorm attention
#     """

#     def __init__(
#         self,
#         n_head,
#         n_feat,
#         dropout_rate=0.0,
#         act_fun="relu",
#         kdim=None,
#         vdim=None,
#         causal=False,
#         has_outproj=True,
#         zero_triu=False,
#     ):
#         super().__init__(n_head, n_feat)
#         self.n_feat = n_feat
#         self.kdim = kdim if kdim is not None else n_feat
#         self.vdim = vdim if kdim is not None else n_feat
#         self.n_head = n_head
#         self.has_outproj = has_outproj
#         self.act_fun = self.get_act_fun(act_fun)
#         # q, k, v projection
#         self.linear_k = nn.Linear(self.kdim, n_feat)
#         self.linear_v = nn.Linear(self.vdim, n_feat)
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         # outprojection
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         # dropout rate
#         self.dropout_rate = dropout_rate
#         # causal
#         self.causal = causal

#         assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

#     def get_act_fun(self, act_fun):
#         if act_fun == "relu":
#             return F.relu
#         elif act_fun == "elu":
#             return 1 + F.elu
#         elif act_fun == "sig":
#             return F.sigmoid
#         elif act_fun == "swish":
#             return F.silu

#     def forward(
#         self,
#         query: Tensor,
#         key: Optional[Tensor] = None,
#         value: Optional[Tensor] = None,
#         pos_emb: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         eps: Optional[float] = 1e-6,
#     ):
#         """Input shape: Sequence x Batch x Embedding
#         Args:
#                 query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
#                 where the mask prevents the attention from looking forward in time (default: None).
#         """
#         if key == None:
#             key = query
#         if value == None:
#             value = query
#         n_head = self.n_head
#         bsz, tgt_len, embed_dim = query.size()
#         src_len = key.size(1)
#         head_dim = embed_dim // n_head
#         query = query.view(tgt_len, bsz, embed_dim)
#         key = key.view(src_len, bsz, embed_dim)
#         value = value.view(src_len, bsz, embed_dim)

#         # get q, k, v
#         # (L, N, E)
#         q = self.linear_q(query)
#         # (S, N, E)
#         k = self.linear_k(key)
#         # (S, N, E)
#         v = self.linear_v(value)

#         # activation
#         # q = self.act_fun(q)
#         # k = self.act_fun(k)

#         q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
#         comp_q = 1 - q
#         k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=0)
#         comp_k = 1 - k

#         # multihead reshape
#         # (N * h, L, d)
#         q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         if self.causal:
#             ## Need to improve speed!
#             # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->nlm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
#             # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
#             kv_cum = torch.cumsum(kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
#             comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
#             qkv += comp_qkv
#             # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
#             # k_cum = torch.cumsum(k_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
#             # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
#             # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
#             attn_output = qkv  # / denom.unsqueeze(-1)
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         else:
#             # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->ndm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
#             # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
#             z_ = torch.clamp_min(
#                 torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
#             ) + torch.clamp_min(
#                 torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
#             )
#             # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
#             attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_ + comp_kv_, z_)

#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         # L, N, E
#         if self.has_outproj:
#             attn_output = self.linear_out(attn_output)

#         return attn_output.view(bsz, tgt_len, self.n_feat)


# class XNorRelPosAttention(XNorAttention):
#     """
#     XNorm attention
#     """

#     def __init__(
#         self,
#         n_head,
#         n_feat,
#         dropout_rate=0.0,
#         act_fun="relu",
#         kdim=None,
#         vdim=None,
#         causal=False,
#         has_outproj=True,
#         zero_triu=False,
#     ):
#         super().__init__(n_head, n_feat)
#         self.n_feat = n_feat
#         self.kdim = kdim if kdim is not None else n_feat
#         self.vdim = vdim if kdim is not None else n_feat
#         self.n_head = n_head
#         self.has_outproj = has_outproj
#         self.act_fun = self.get_act_fun(act_fun)
#         # q, k, v projection
#         self.linear_k = nn.Linear(self.kdim, n_feat)
#         self.linear_v = nn.Linear(self.vdim, n_feat)
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         # outprojection
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         # dropout rate
#         self.dropout_rate = dropout_rate
#         # causal
#         self.causal = causal

#         assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"
#         self.zero_triu = zero_triu
#         # linear transformation for positional encoding
#         self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
#         # these two learnable bias are used in matrix c and matrix d
#         # as described in https://arxiv.org/abs/1901.02860 Section 3.3
#         self.pos_bias_u = nn.Parameter(torch.Tensor(n_head, n_feat // n_head))
#         self.pos_bias_v = nn.Parameter(torch.Tensor(n_head, n_feat // n_head))
#         torch.nn.init.xavier_uniform_(self.pos_bias_u)
#         torch.nn.init.xavier_uniform_(self.pos_bias_v)

#     def rel_shift(self, x):
#         """Compute relative positional encoding.

#         Args:
#             x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
#             time1 means the length of query vector.

#         Returns:
#             torch.Tensor: Output tensor.

#         """
#         zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
#         x_padded = torch.cat([zero_pad, x], dim=-1)

#         x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
#         x = x_padded[:, :, 1:].view_as(x)[
#             :, :, :, : x.size(-1) // 2 + 1
#         ]  # only keep the positions from 0 to time2

#         if self.zero_triu:
#             ones = torch.ones((x.size(2), x.size(3)), device=x.device)
#             x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

#         return x

#     def forward(
#         self,
#         query: Tensor,
#         key: Optional[Tensor] = None,
#         value: Optional[Tensor] = None,
#         pos_emb: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         eps: Optional[float] = 1e-6,
#     ):
#         """Input shape: Sequence x Batch x Embedding
#         Args:
#                 query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
#                 where the mask prevents the attention from looking forward in time (default: None).
#         """
#         if key == None:
#             key = query
#         if value == None:
#             value = query
#         n_head = self.n_head
#         bsz, tgt_len, embed_dim = query.size()
#         nbatch_pos, _, embed_dim = pos_emb.size()
#         src_len = key.size(1)
#         head_dim = embed_dim // n_head
#         query = query.view(tgt_len, bsz, embed_dim)
#         key = key.view(src_len, bsz, embed_dim)
#         value = value.view(src_len, bsz, embed_dim)
#         # logging.info(f"POS EMB Shape {pos_emb.shape} QUERY {query.shape}")
#         pos_emb = pos_emb.view(-1, n_batch_pos, embed_dim)

#         # get q, k, v
#         # (L, N, E)
#         q = self.linear_q(query)
#         # (S, N, E)
#         k = self.linear_k(key)
#         # (S, N, E)
#         v = self.linear_v(value)

#         # activation
#         # q = self.act_fun(q)
#         # k = self.act_fun(k)

#         ## REL POS EMB
#         n_batch_pos = pos_emb.size(0)

#         p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, head_dim)
#         p_ = torch.nn.functional(p, dim=-1)
#         comp_p_ = 1 - p
#         p_ = p_.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)
#         comp_p_ = comp_p_.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

#         q_reshaped_ = q.transpose(0, 1).view(
#             bsz, tgt_len, self.n_head, embed_dim // self.n_head
#         )  ## (batch,seqlen,n_head,emb_dim)
#         q_with_bias_u = (q_reshaped_ + self.pos_bias_u).transpose(1, 2)
#         q_with_bias_v = (q_reshaped_ + self.pos_bias_v).transpose(1, 2)

#         q_comp_with_bias_u = (1 - q_reshaped_ + self.pos_bias_u).transpose(1, 2)
#         q_comp_with_bias_v = (1 - q_reshaped_ + self.pos_bias_v).transpose(1, 2)
#         # (batch, head, time1, d_k)
#         q_with_bias_u_ = torch.nn.functional.softmax(q_with_bias_u, dim=-1).view(
#             bsz * self.n_head, tgt_len, embed_dim // self.n_head
#         )
#         # (batch, head, time1, d_k)
#         q_with_bias_v_ = torch.nn.functional.softmax(q_with_bias_v, dim=-1).view(
#             bsz * self.n_head, tgt_len, embed_dim // self.n_head
#         )

#         q_comp_with_bias_u_ = torch.nn.functional.softmax(
#             q_comp_with_bias_u, dim=-1
#         ).view(bsz * self.n_head, tgt_len, embed_dim // self.n_head)
#         # (batch, head, time1, d_k)
#         q_comp_with_bias_v_ = torch.nn.functional.softmax(
#             q_comp_with_bias_v, dim=-1
#         ).view(bsz * self.n_head, tgt_len, embed_dim // self.n_head)

#         # # (batch, head, time1, 2*time1-1)
#         # matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
#         # matrix_bd = self.rel_shift(matrix_bd)

#         k = torch.nn.functional.softmax(
#             k.contiguous().view(bsz, src_len, self.n_head, embed_dim // self.n_head),
#             dim=-1,
#         )
#         comp_k = 1 - k

#         # multihead reshape
#         # (N * h, L, d)
#         q_with_bias_u_ = (
#             q_with_bias_u_.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         )
#         q_comp_with_bias_u_ = (
#             q_comp_with_bias_u_.contiguous()
#             .view(-1, bsz * n_head, head_dim)
#             .transpose(0, 1)
#         )
#         q_comp_with_bias_v_ = (
#             q_comp_with_bias_v_.contiguous()
#             .view(-1, bsz * n_head, head_dim)
#             .transpose(0, 1)
#         )
#         q_with_bias_v_ = (
#             q_with_bias_v_.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         )

#         # (N * h, S, d)
#         k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         if self.causal:
#             ## Need to improve speed!
#             # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->nlm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
#             # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
#             kv_cum = torch.cumsum(kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
#             comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
#             qkv += comp_qkv
#             # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
#             # k_cum = torch.cumsum(k_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
#             # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
#             # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
#             attn_output = qkv  # / denom.unsqueeze(-1)
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         else:
#             # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->ndm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)

#             # (N * h, S, d) (N * h, 2S-1, d) -> (N * h, 2 * d, d)
#             rv_ = torch.einsum("nld,nlm->ndm", k_, p_)
#             comp_rv_ = torch.einsum("nld,nlm->ndm", comp_k_, comp_p_)
#             logging.info(
#                 f"P {p_.shape} K {k_.shape} V {v.shape} QU {q_with_bias_u_.shape} QV {q_comp_with_bias_v_}"
#             )
#             # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
#             z_1_ = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", q_with_bias_u_, torch.sum(k_, axis=1)), eps
#             )
#             comp_z_1_ = 1 / torch.clamp_min(
#                 torch.einsum(
#                     "nld,nd->nl", q_comp_with_bias_u_, torch.sum(comp_k_, axis=1)
#                 ),
#                 eps,
#             )
#             z_2 = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", q_with_bias_v_, torch.sum(p_, axis=1)), eps
#             )
#             comp_z_2 = 1 / torch.clamp_min(
#                 torch.einsum(
#                     "nld,nd->nl", q_comp_with_bias_v_, torch.sum(comp_p_, axis=1)
#                 ),
#                 eps,
#             )
#             # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
#             attn_output1 = torch.einsum("nld,ndm,nl->nlm", q_with_bias_u_, kv_, z_1_)
#             comp_attn_output1 = torch.einsum(
#                 "nld,ndm,nl->nlm", q_comp_with_bias_u_, comp_kv_, comp_z_1_
#             )
#             attn_output2 = self.rel_shift(
#                 torch.einsum("nld,ndm,nl->nlm", q_with_bias_v_, rv_, z_2)
#             )
#             comp_attn_output2 = self.rel_shift(
#                 torch.einsum("nld,ndm,nl->nlm", q_comp_with_bias_v_, comp_rv_, comp_z_2)
#             )

#             # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
#             attn_output = (
#                 attn_output1 + attn_output2 + comp_attn_output1 + comp_attn_output2
#             )
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         # L, N, E
#         if self.has_outproj:
#             attn_output = self.linear_out(attn_output)

#         return attn_output.view(bsz, tgt_len, self.n_feat)

#     def left_product(
#         self,
#         query: Tensor,
#         key: Optional[Tensor] = None,
#         value: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         eps: Optional[float] = 1e-6,
#     ):
#         """Input shape: Sequence x Batch x Embedding
#         Args:
#                 query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
#                 where the mask prevents the attention from looking forward in time (default: None).
#         """
#         # test for the correctness of the program
#         if key == None:
#             key = query
#         if value == None:
#             value = query

#         n_head = self.n_head
#         tgt_len, bsz, embed_dim = query.size()
#         src_len = key.size(0)
#         head_dim = embed_dim // n_head

#         # get q, k, v
#         # (L, N, E)
#         q = self.linear_q(query)
#         # (S, N, E)
#         k = self.linear_k(key)
#         # (S, N, E)
#         v = self.linear_v(value)

#         # activation
#         q = self.act_fun(q)
#         k = self.act_fun(k)

#         # multihead reshape
#         # (N * h, L, d)
#         q = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         # (N * h, S, d)
#         k = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         # (N * h, S, d)
#         v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         q_ = q
#         k_ = k
#         # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
#         weights = torch.bmm(q_, k_.transpose(1, 2))
#         # mask
#         if self.causal:
#             weights = weights.masked_fill(attn_mask == float("-inf"), 0)
#         # (N * h, L, S) -> (N * h, L, S)
#         denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
#         # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
#         attn_weights = weights / denom
#         # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
#         attn_output = torch.bmm(attn_weights, v)
#         # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#         # L, N, E
#         if self.has_outproj:
#             attn_output = self.linear_out(attn_output)

#         return attn_output





# class XNorRopeAttention(XNorAttention):
#     """
#     XNor Rope attention
#     """

#     def __init__(
#         self,
#         n_head,
#         n_feat,
#         dropout_rate=0.0,
#         act_fun="relu",
#         kdim=None,
#         vdim=None,
#         causal=False,
#         has_outproj=True,
#         zero_triu=False,
#     ):
#         super().__init__(n_head, n_feat)
#         self.n_feat = n_feat
#         self.kdim = kdim if kdim is not None else n_feat
#         self.vdim = vdim if kdim is not None else n_feat
#         self.n_head = n_head
#         self.has_outproj = has_outproj
#         self.act_fun = self.get_act_fun(act_fun)
#         # q, k, v projection
#         self.linear_k = nn.Linear(self.kdim, n_feat)
#         self.linear_v = nn.Linear(self.vdim, n_feat)
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         # outprojection
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         # dropout rate
#         self.dropout_rate = dropout_rate
#         # causal
#         self.causal = causal
#         # ROPE Encoding
#         self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
#             8192, n_feat // n_head
#         )

#         assert n_feat % self.n_head == 0, "embed_dim must be divisible by n_head"

#     def forward(
#         self,
#         query: Tensor,
#         key: Optional[Tensor] = None,
#         value: Optional[Tensor] = None,
#         pos_emb: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         eps: Optional[float] = 1e-6,
#     ):
#         """Input shape: Sequence x Batch x Embedding
#         Args:
#                 query (Tensor): `(N,L, E)` where L is the target sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 key (Tensor): `(N,S, E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 value (Tensor): `(N,S,E)` where S is the source sequence length, N is the batch size,
#                 E is the embedding dimension.
#                 attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
#                 where the mask prevents the attention from looking forward in time (default: None).
#         """
#         if key == None:
#             key = query
#         if value == None:
#             value = query
#         n_head = self.n_head
#         bsz, tgt_len, embed_dim = query.size()
#         src_len = key.size(1)
#         head_dim = embed_dim // n_head
#         query = query.view(tgt_len, bsz, embed_dim)
#         key = key.view(src_len, bsz, embed_dim)
#         value = value.view(src_len, bsz, embed_dim)

#         sinusoidal_pos = self.embed_positions(
#             query.permute(1, 0, 2).view(bsz, -1, n_head, head_dim).shape[:-1]
#         )[None, None, :, :]
#         # logging.info(f"POSENC {sinusoidal_pos.shape}")
#         ## ROPE: Needs (bs,num_heads,seq_len,per_head_dim)

#         # get q, k, v
#         # (L, N, E)
#         q = self.linear_q(query)
#         # (S, N, E)
#         k = self.linear_k(key)
#         # (S, N, E)
#         v = self.linear_v(value)

#         q = torch.nn.functional.softmax(q.view(-1, bsz, n_head, head_dim), dim=-1)
#         comp_q = 1 - q
#         k = torch.nn.functional.softmax(k.view(-1, bsz, n_head, head_dim), dim=-1)
#         comp_k = 1 - k
#         q, k = self.apply_rotary_position_embeddings(
#             sinusoidal_pos, q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3)
#         )
#         comp_q, comp_k = self.apply_rotary_position_embeddings(
#             sinusoidal_pos, comp_q.permute(1, 2, 0, 3), comp_k.permute(1, 2, 0, 3)
#         )
#         q, k = q.permute(2, 0, 1, 3), k.permute(2, 0, 1, 3)
#         comp_q, comp_k = comp_q.permute(2, 0, 1, 3), comp_k.permute(2, 0, 1, 3)

#         # multihead reshape
#         # (N * h, L, d)
#         q_ = q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_q_ = comp_q.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         k_ = k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)
#         comp_k_ = comp_k.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         # (N * h, S, d)
#         v = v.contiguous().view(-1, bsz * n_head, head_dim).transpose(0, 1)

#         if self.causal:
#             ## Need to improve speed!
#             # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->nlm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->nldm", comp_k_, v)
#             # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
#             kv_cum = torch.cumsum(kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
#             comp_kv_cum = torch.cumsum(comp_kv_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
#             comp_qkv = torch.einsum("nld,nldm->nlm", comp_q_, comp_kv_cum)
#             qkv += comp_qkv
#             # (N * h, L, 2 * d) -> (N * h,  L, 2 * d)
#             # k_cum = torch.cumsum(k_, dim=1)
#             # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
#             # denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
#             # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
#             attn_output = qkv  # / denom.unsqueeze(-1)
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         else:
#             # (N * h, S, 2 * d) (N * h, S, d) -> (N * h, 2 * d, d)
#             kv_ = torch.einsum("nld,nlm->ndm", k_, v)
#             comp_kv_ = torch.einsum("nld,nlm->ndm", comp_k_, v)
#             # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
#             z_ = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
#             )
#             comp_z_ = 1 / torch.clamp_min(
#                 torch.einsum("nld,nd->nl", comp_q_, torch.sum(comp_k_, axis=1)), eps
#             )
#             # (N * h, L, 2 * d) (N * h, 2 * d, d) (N * h, L) -> (N * h, L, d)
#             attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)

#             # attn_output = torch.einsum("nld,ndm->nlm", q_, kv_)
#             attn_output2 = torch.einsum("nld,ndm,nl->nlm", comp_q_, comp_kv_, comp_z_)

#             # comp_qkv = torch.einsum("nld,ndm->nlm", comp_q_, comp_kv_)
#             attn_output += attn_output2
#             # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
#             attn_output = (
#                 attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
#             )
#         # L, N, E
#         if self.has_outproj:
#             attn_output = self.linear_out(attn_output)

#         return attn_output.view(bsz, tgt_len, self.n_feat)