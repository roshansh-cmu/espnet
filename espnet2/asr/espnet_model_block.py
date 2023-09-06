import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet2.asr.ctc import CTC

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import pad_list, th_accuracy, make_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)


if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetBlockASRModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        enc_context: str = None,
        dec_context: str = None,
        block_length: int = 2,
        encoder_prompt: bool = False,
        detach_context: bool = False,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        )
        if dec_context == "add":
            self.decoder.compute_output_projection = False
        self.enc_context_method = enc_context
        self.dec_context_method = dec_context
        self.enc_context = []
        self.dec_context = []
        self.block_id = 0
        self.block_length = block_length
        self.detach_context = detach_context
        self.encoder_prompt = encoder_prompt
        self.gating_factor = None
        self.topic_vector = None  # Placeholder for topic vector

        if (
            self.enc_context_method == "attention"
            or self.enc_context_method == "attentionv2"
        ):
            self.block_attention = MultiHeadedAttention(
                n_head=4, n_feat=self.encoder._output_size, dropout_rate=0.2
            )
        if (
            self.enc_context_method == "attention"
            or self.enc_context_method == "attentionv3"
        ):
            self.gating_factor = torch.nn.parameter.Parameter(torch.tensor(0.5))

        if self.enc_context_method == "attentiongated":
            self.prev_block_attention = MultiHeadedAttention(
                n_head=2, n_feat=self.encoder._output_size, dropout_rate=0.2
            )
            self.curr_block_attention = MultiHeadedAttention(
                n_head=2, n_feat=self.encoder._output_size, dropout_rate=0.2
            )
            self.gater = torch.nn.Sequential(
                torch.nn.Linear(self.encoder._output_size * 2, 1, bias=True),
                torch.nn.Sigmoid(),
            )

    def reset_batch(self):
        self.enc_context = []
        self.dec_context = []
        self.block_id = 0
        if self.encoder_prompt:
            self.encoder.reset_prompt()

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        block_id: int = 0,
        final_block: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        if block_id == 0:
            self.reset_batch()
        self.block_id = block_id
        self.final_block = final_block

        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        if self.encoder_prompt:
            # logging.warning(f"Block ID {self.block_id} ENC Shape{self.enc_context[-1][0].shape if self.block_id !=0 else None }")
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats,
                feats_lengths,
                enc_context=self.enc_context[-1][0] if self.block_id != 0 else None,
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        encoder_out, encoder_out_lens = self.combine_encoder_context(
            encoder_out, encoder_out_lens
        )

        return encoder_out, encoder_out_lens

    def combine_encoder_context(self, encoder_out, encoder_out_lens):
        ## Concatenation of prev encoder output - DONE
        ## Addition of prev block encoder output - DONE
        ## Gated Addition of attended context vector - DONE

        ## We want to only keep context from past block_length blocks
        if (
            len(self.enc_context) == self.block_length
            and self.enc_context_method != "attention"
        ):
            if not self.training:
                self.enc_context[0][0].detach().cpu().numpy()

            self.enc_context = self.enc_context[1:]

        if self.enc_context_method == "attention":
            if len(self.enc_context) >= 1 and self.block_id > 0:
                hs, hlens = self.enc_context[-1][0], self.enc_context[-1][1]
                memory_mask = (~make_pad_mask(hlens, maxlen=hs.size(1)))[:, None, :].to(
                    hs.device
                )

                attended_output = self.block_attention(
                    hs, encoder_out, encoder_out, memory_mask
                )
                output = encoder_out + self.gating_factor * attended_output
            else:
                output = encoder_out
            self.enc_context = [[encoder_out.detach(), encoder_out_lens]]
            encoder_out = output

        # elif self.enc_context_method == "attentionv2":
        #     if len(self.enc_context) >= 1 and self.block_id > 0:
        #         hs, hlens = self.enc_context[-1][0], self.enc_context[-1][1]
        #         memory_mask = (~make_pad_mask(hlens, maxlen=hs.size(1)))[:, None, :].to(
        #             hs.device
        #         )
        #         try:
        #             attended_output = self.block_attention(
        #                 hs, encoder_out, encoder_out, memory_mask
        #             )
        #             output = hs + self.gating_factor * attended_output
        #             outlens = torch.ones_like(hlens) * hs.size(1)
        #         except:
        #             logging.info(
        #                 f"HS shape {hs.shape} Encoder out shape {encoder_out.shape}"
        #             )
        #             output = hs
        #             outlens = hlens
        #     else:
        #         output = encoder_out
        #         outlens = encoder_out_lens
        #     self.enc_context = [[encoder_out.detach(), outlens]]
        #     encoder_out = output

        elif self.enc_context_method == "attentionv3":
            if len(self.enc_context) >= 1 and self.block_id > 0:
                hs, hlens = self.enc_context[-1][0], self.enc_context[-1][1]
                memory_mask = (~make_pad_mask(hlens, maxlen=hs.size(1)))[:, None, :].to(
                    hs.device
                )
                output = (
                    1 - self.gating_factor
                ) * hs + self.gating_factor * encoder_out
                outlens = torch.ones_like(hlens) * hs.size(1)
            else:
                output = encoder_out
                outlens = encoder_out_lens

            self.enc_context = [[encoder_out.detach(), outlens]]
            encoder_out = output
            encoder_out_lens = outlens

        elif self.enc_context_method == "attentiongated":
            if self.block_id > 0 and len(self.enc_context) >= 1:
                prev_sem_emb, prevlens = (
                    self.enc_context[-1][0],
                    self.enc_context[-1][1],
                )
                topic_mask = None

                current_topic_similarity = self.curr_block_attention(
                    encoder_out, self.topic_vector, self.topic_vector, topic_mask
                )
                prev_topic_similarity = self.prev_block_attention(
                    prev_sem_emb, self.topic_vector, self.topic_vector, topic_mask
                )
                combined = torch.cat(
                    [
                        torch.mean(current_topic_similarity, dim=1),
                        torch.mean(prev_topic_similarity, dim=1),
                    ],
                    dim=-1,
                )
                self.gating_factor = self.gater(combined).unsqueeze(-1)
                output = (
                    (1 - self.gating_factor) * prev_topic_similarity
                    + self.gating_factor * current_topic_similarity
                )

                outlens = torch.ones_like(prevlens) * output.size(1)
            else:
                output = encoder_out
                outlens = encoder_out_lens

            self.enc_context = [[encoder_out.detach(), encoder_out_lens]]
            encoder_out = output
            encoder_out_lens = outlens

        elif self.enc_context_method == "concat":
            self.enc_context.append([encoder_out.detach(), encoder_out_lens])
            if len(self.enc_context) > 0 and self.block_id > 0:
                # logging.info(
                #     f"Block {self.block_id} Encoder Context Length {len(self.enc_context)} | Shapes {[x[0].shape for x in self.enc_context]} | Maxlens {[x[1].max() for x in self.enc_context]}"
                # )
                context = [x for x in self.enc_context[0][0].clone()]
                context_lens = self.enc_context[0][1].clone()
                prev_ctx = self.enc_context[1:] if len(self.enc_context) > 1 else []
                for prev_context, prev_lens in prev_ctx + [
                    [encoder_out, encoder_out_lens]
                ]:
                    for k, x in enumerate(context):
                        context[k] = torch.cat(
                            [
                                x[: context_lens[k], :],
                                prev_context[k][: prev_lens[k], :],
                            ],
                            dim=0,
                        )
                        context_lens[k] += prev_lens[k]
                encoder_out = pad_list(context, pad_value=0.0)
                encoder_out_lens = context_lens
            # logging.info(
            #     f"Block {self.block_id} Max Encoder Length {max(encoder_out_lens)} Enc shape {encoder_out.shape}"
            # )

        elif self.enc_context_method == "concatv2":
            if len(self.enc_context) >= 1 and self.block_id > 0:
                context, context_lens = self.enc_context[0]
                context = [x for x in context]
                prev_ctx = self.enc_context[1:] if len(self.enc_context) > 1 else []
                for prev_context, prev_lens in prev_ctx + [
                    [encoder_out, encoder_out_lens]
                ]:
                    for k, x in enumerate(context):
                        context[k] = torch.cat(
                            [
                                x[: context_lens[k], :],
                                prev_context[k][: prev_lens[k], :],
                            ],
                            dim=0,
                        )
                        context_lens[k] += prev_lens[k]
                output = pad_list(context, pad_value=0.0)
                output_lens = context_lens
            else:
                output = encoder_out
                output_lens = encoder_out_lens

            # logging.warning(f"Block {self.block_id} Max Encoder Length {max(encoder_out_lens)} Enc shape {encoder_out.shape}")
            self.enc_context.append([encoder_out.detach(), encoder_out_lens])
            encoder_out = output
            encoder_out_lens = output_lens

        elif self.enc_context_method == "add":
            if len(self.enc_context) >= 1 and self.block_id > 0:
                output = (
                    torch.sum(torch.stack(self.enc_context, dim=0), dim=0) + encoder_out
                )
            else:
                output = encoder_out
            self.enc_context.append(encoder_out.detach(), encoder_out_lens)
            encoder_out = output

        elif self.enc_context_method == "han":
            self.enc_context.append([encoder_out.detach(), encoder_out_lens])

        return encoder_out, encoder_out_lens

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        block_id: int = 0,
        final_block: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, block_id=block_id, final_block=final_block
        )
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            if self.final_block:
                stats["final_acc"] = acc_att
            stats[f"block_{self.block_id}_acc"] = acc_att
            stats[f"block_{self.block_id}_att_loss"] = loss_att.detach()
            if self.enc_context_method == "attentiongated" and self.block_id > 0:
                stats[f"block_{self.block_id}_gate_wt"] = torch.mean(
                    self.gating_factor.detach()
                )
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()
        stats[f"loss_block_{self.block_id}"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        return_hs: bool = False,
    ):
        return_hs = True if self.enc_context_method == "attentiongated" else False
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 2. Decoder
        if self.block_id > 0 and self.dec_context_method == "han":
            ## Concatenate prompt to input
            # logging.warning(f"Encoder Context Length {len(self.enc_context)}")
            decoder_out, _ = self.decoder(
                encoder_out,
                encoder_out_lens,
                ys_in_pad,
                ys_in_lens,
                context=self.enc_context[-2],
            )
        else:
            if return_hs:
                decoder_out, _, self.topic_vector = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, return_hs=True
                )
                self.topic_vector = self.topic_vector.detach()
            else:
                decoder_out, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
                )

        ## 3. Process Decoder Context
        # if self.dec_context_method == "add":
        #     prev_context = 0
        #     if self.block_id > self.block_length:
        #         prev_context = self.dec_context[-(self.block_length - 1) :]
        #         context = torch.sum(torch.stack(prev_context, dim=0), dim=0)
        #     else:
        #         context = decoder_out
        #     decoder_out = self.decoder.output_layer(context)
        #     if self.dec_context is None:
        #         self.dec_context = [decoder_out.detach()]
        #     else:
        #         self.dec_context.append(decoder_out.detach())

        # 4. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # 5. Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att
