# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import torch
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder

from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class BlockConformerEncoder(ConformerEncoder):
    """Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        prompt_after_cnn: bool = False,
    ):
        assert check_argument_types()
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=macaron_style,
            rel_pos_type=rel_pos_type,
            pos_enc_layer_type=pos_enc_layer_type,
            selfattention_layer_type=selfattention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_cnn_module,
            zero_triu=zero_triu,
            cnn_module_kernel=cnn_module_kernel,
            padding_idx=padding_idx,
            interctc_layer_idx=interctc_layer_idx,
            interctc_use_conditioning=interctc_use_conditioning,
            stochastic_depth_rate=stochastic_depth_rate,
            layer_drop_rate=layer_drop_rate,
            max_pos_emb_len=max_pos_emb_len,
        )
        self._output_size = output_size

        self.prompt_after_cnn = prompt_after_cnn
        if prompt_after_cnn:
            self.init_prompt = torch.nn.Parameter(torch.randn(10, output_size))
            self.register_parameter("init_prompt", self.init_prompt)
            self.prompt_generator = MultiHeadedAttention(
                n_head=1, n_feat=output_size, dropout_rate=0.1
            )
            self.prev_prompts = [self.init_prompt]

    def reset_prompt(self):
        self.prev_prompts = [self.init_prompt]

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        enc_context: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            if self.prompt_after_cnn:
                prev_prompt = (
                    self.prev_prompts[-1].repeat(xs_pad.shape[0], 1, 1)
                    if self.prev_prompts[-1].ndim == 2
                    else self.prev_prompts[-1]
                )
                masks = (
                    ~make_pad_mask(ilens + 4 * prev_prompt.shape[1])[:, None, :]
                ).to(xs_pad.device)
            else:
                prev_prompt = None

            # logging.warning(f"Prev Prompt Shape : {prev_prompt.shape}")
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        if self.prompt_after_cnn:
            mean_semantic_embedding = (
                enc_context
                if enc_context is not None
                else torch.zeros((xs_pad[0].shape[0], 1, self._output_size)).to(
                    xs_pad[0].device
                )
            )
            # mean_semantic_embedding = torch.mean(enc_context,dim=1).unsqueeze(1) if enc_context is not None else torch.zeros((xs_pad[0].shape[0],1,self._output_size)).to(xs_pad[0].device)

            ## Prev Semantic Embedding is Query and prev Prompt is key and value
            # logging.warning(f"Before Mean Semantic Embedding Shape : {mean_semantic_embedding.shape} Prev Prompt Shape : {prev_prompt.shape}")
            current_prompt = self.prompt_generator(
                prev_prompt, mean_semantic_embedding, mean_semantic_embedding, mask=None
            )
            # logging.warning(f" After Mean Semantic Embedding Shape : {mean_semantic_embedding.shape} Prev Prompt Shape : {prev_prompt.shape} Current Prompt Shape : {current_prompt.shape} {xs_pad[0].shape} {xs_pad[1].shape} {masks.shape}")
            ## Concatenate prompt and input
            xs_pad = (torch.cat([xs_pad[0], current_prompt], dim=1), xs_pad[1])
            self.prev_prompts.append(current_prompt.detach())

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None

        if self.prompt_after_cnn:
            xs_pad = xs_pad[:, 10:, :]
            olens = olens - 10
        return xs_pad, olens, None
