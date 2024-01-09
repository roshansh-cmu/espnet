




from typeguard import check_argument_types
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from typing import Any, List, Sequence, Tuple

from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.context_decoder_layer import ContextualDecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.asr.decoder.transformer_decoder import BaseTransformerDecoder
import torch 


class BlockHANTransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        last_hierarchical_attn: bool = False,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        if last_hierarchical_attn:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ) if lnum > 0 else ContextualDecoderLayer(
                    size=attention_dim,
                    self_attn=MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    src_attn=MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    feed_forward=PositionwiseFeedForward(
                        attention_dim, linear_units, dropout_rate
                    ),
                    dropout_rate=dropout_rate,
                    normalize_before=normalize_before,
                    concat_after=concat_after,
                    context_attn=MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    use_context_hier_attn=True,
                    ),
                layer_drop_rate,
            )
        else:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: ContextualDecoderLayer(
                    size=attention_dim,
                    self_attn=MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    src_attn=MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    feed_forward=PositionwiseFeedForward(
                        attention_dim, linear_units, dropout_rate
                    ),
                    dropout_rate=dropout_rate,
                    normalize_before=normalize_before,
                    concat_after=concat_after,
                    use_context_hier_attn=True,
                    context_attn=MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    ),
                layer_drop_rate,
            )
    
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        context: List[torch.Tensor]= None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        if context is not None:
            context,context_lens = context
            context_mask = (~make_pad_mask(context_lens, maxlen=context.size(1)))[:, None, :].to(context.device)
        else:
            context_mask = None
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )
        # logging.warning(f"memory_mask.shape: {memory_mask.shape} Memory shape: {memory.shape} Target shape: {tgt.shape}")

        x = self.embed(tgt)
        for decoder in self.decoders:
            x, tgt_mask, memory, memory_mask,_ = decoder(
                x, tgt_mask, memory, memory_mask,context=context,context_mask=context_mask
            )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None and self.compute_output_projection:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
        context: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        if context is not None:
            context,context_lens = context
            context_mask = (~make_pad_mask(context_lens, maxlen=context.size(1)))[:, None, :].to(context.device)
        else:
            context_mask = None
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask,_ = decoder(
                x, tgt_mask, memory, None, cache=c,context=context,context_mask=context_mask
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def score(self, ys, state, x,context=None,context_mask=None):
        """Score."""

        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state,context=context
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor,context: torch.Tensor=None,context_mask: torch.Tensor=None
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state,context=context)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

        