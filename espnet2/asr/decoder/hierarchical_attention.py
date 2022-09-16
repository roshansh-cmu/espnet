"""
This file implements hierarchical attention for combination of audio and visual features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(name):
    """Returns a callable activation function from torch."""
    if name in (None, "linear"):
        return lambda x: x
    elif name in ("sigmoid", "tanh"):
        return getattr(torch, name)
    else:
        return getattr(F, name)


class HierarchicalAttention(nn.Module):
    """Hierarchical attention over multiple modalities."""

    def __init__(
        self, ctx_dims, hid_dim, mid_dim, att_activ="tanh", att_feed_factor=0.0
    ):
        super().__init__()

        self.activ = get_activation_fn(att_activ)
        self.ctx_dims = ctx_dims
        self.hid_dim = hid_dim
        self.mid_dim = mid_dim
        self.att_feed_factor = att_feed_factor

        self.ctx_projs = nn.ModuleList(
            [nn.Linear(dim, mid_dim, bias=False) for dim in self.ctx_dims]
        )
        self.dec_proj = nn.Linear(hid_dim, mid_dim, bias=True)
        self.mlp = nn.Linear(self.mid_dim, 1, bias=False)
        self.attn = None

    def forward(self, contexts, hid):
        dec_state_proj = self.dec_proj(hid)
        ctx_projected = torch.cat(
            [p(ctx).unsqueeze(0) for p, ctx in zip(self.ctx_projs, contexts)], dim=0
        )
        energies = self.mlp(self.activ(dec_state_proj + ctx_projected))
        att_dist = nn.functional.softmax(energies, dim=0)
        if torch.rand(1) < self.att_feed_factor:
            att_dist = torch.ones_like(att_dist)
            att_dist[0, ::] *= 0.6
            att_dist[1, ::] *= 0.4
        ctxs_cat = torch.cat([c.unsqueeze(0) for c in contexts])
        joint_context = (att_dist * ctxs_cat).sum(0)
        self.attn = att_dist

        return joint_context
