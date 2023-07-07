import torch
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class FNetSelfAttention(torch.nn.Module):
    def __init__(
        self,
        size,
        dropout_rate=0,
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.norm1 = LayerNorm(size)

    def forward(self, query, key, value, pos_emb=None, mask=None):
        # query B x L x D
        outputs = torch.fft.fftn(query, dim=(1, 2)).real
        outputs = self.norm1(outputs)
        return self.dropout(outputs)
