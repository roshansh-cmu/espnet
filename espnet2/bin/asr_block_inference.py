#!/usr/bin/env python3
import argparse
import logging
import sys
from distutils.version import LooseVersion
from pathlib import Path
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
from typeguard import check_argument_types, check_return_type
from espnet.nets.beam_search_han import BeamSearchHAN

from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer import (
    ExtendedHypothesis as ExtTransHypothesis,
)
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.block_wise_asr import ASRTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.beam_search_timesync import BeamSearchTimeSync
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args

try:
    from transformers import AutoModelForSeq2SeqLM
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        transducer_conf: dict = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        enh_s2t_task: bool = False,
        quantize_asr_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        hugging_face_decoder: bool = False,
        hugging_face_decoder_max_length: int = 256,
        time_sync: bool = False,
        multi_asr: bool = False,
    ):
        assert check_argument_types()

        task = ASRTask if not enh_s2t_task else EnhS2TTask

        if quantize_asr_model or quantize_lm:
            if quantize_dtype == "float16" and torch.__version__ < LooseVersion(
                "1.5.0"
            ):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with "
                    "torch version < 1.5.0. Switch to qint8 dtype instead."
                )

        quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype = getattr(torch, quantize_dtype)

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = task.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        if enh_s2t_task:
            asr_model.inherite_attributes(
                inherite_s2t_attrs=[
                    "ctc",
                    "decoder",
                    "eos",
                    "joint_network",
                    "sos",
                    "token_list",
                    "use_transducer_decoder",
                ]
            )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        if quantize_asr_model:
            logging.info("Use quantized asr model for decoding.")

            asr_model = torch.quantization.quantize_dynamic(
                asr_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
            )

        decoder = asr_model.decoder

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )

            if quantize_lm:
                logging.info("Use quantized lm for decoding.")

                lm = torch.quantization.quantize_dynamic(
                    lm, qconfig_spec=quantize_modules, dtype=quantize_dtype
                )

            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        if asr_model.use_transducer_decoder:
            beam_search_transducer = BeamSearchTransducer(
                decoder=asr_model.decoder,
                joint_network=asr_model.joint_network,
                beam_size=beam_size,
                lm=scorers["lm"] if "lm" in scorers else None,
                lm_weight=lm_weight,
                token_list=token_list,
                **transducer_conf,
            )
            beam_search = None
            hugging_face_model = None
            hugging_face_linear_in = None
        elif (
            decoder.__class__.__name__ == "HuggingFaceTransformersDecoder"
            and hugging_face_decoder
        ):
            if not is_transformers_available:
                raise ImportError(
                    "`transformers` is not available."
                    " Please install it via `pip install transformers`"
                    " or `cd /path/to/espnet/tools && . ./activate_python.sh"
                    " && ./installers/install_transformers.sh`."
                )

            hugging_face_model = AutoModelForSeq2SeqLM.from_pretrained(
                decoder.model_name_or_path
            )

            hugging_face_model.lm_head.load_state_dict(decoder.lm_head.state_dict())

            if hasattr(hugging_face_model, "model"):
                hugging_face_model.model.decoder.load_state_dict(
                    decoder.decoder.state_dict()
                )
                del hugging_face_model.model.encoder
            else:
                hugging_face_model.decoder.load_state_dict(decoder.decoder.state_dict())
                del hugging_face_model.encoder

            del asr_model.decoder.lm_head
            del asr_model.decoder.decoder

            hugging_face_linear_in = decoder.linear_in
            hugging_face_model.to(device=device).eval()

            beam_search = None
            beam_search_transducer = None
        else:
            beam_search_transducer = None
            hugging_face_model = None
            hugging_face_linear_in = None

            weights = dict(
                decoder=1.0 - ctc_weight,
                ctc=ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
            )

            if time_sync:
                if not hasattr(asr_model, "ctc"):
                    raise NotImplementedError(
                        "BeamSearchTimeSync without CTC is not supported."
                    )
                if batch_size != 1:
                    raise NotImplementedError(
                        "BeamSearchTimeSync with batching is not yet supported."
                    )
                logging.info("BeamSearchTimeSync implementation is selected.")

                scorers["ctc"] = asr_model.ctc
                beam_search = BeamSearchTimeSync(
                    beam_size=beam_size,
                    weights=weights,
                    scorers=scorers,
                    sos=asr_model.sos,
                    token_list=token_list,
                )
            elif "han" in asr_model_file:
                beam_search = BeamSearchHAN(
                    beam_size=beam_size,
                    weights=weights,
                    scorers=scorers,
                    sos=asr_model.sos,
                    eos=asr_model.eos,
                    vocab_size=len(token_list),
                    token_list=token_list,
                    pre_beam_score_key=None if ctc_weight == 1.0 else "full",
                )
                # # TODO(karita): make all scorers batchfied
                # if batch_size == 1:
                #     non_batch = [
                #         k
                #         for k, v in beam_search.full_scorers.items()
                #         if not isinstance(v, BatchScorerInterface)
                #     ]
                #     if len(non_batch) == 0:
                #         if streaming:
                #             beam_search.__class__ = BatchBeamSearchOnlineSim
                #             beam_search.set_streaming_config(asr_train_config)
                #             logging.info(
                #                 "BatchBeamSearchOnlineSim implementation is selected."
                #             )
                #         else:
                #             beam_search.__class__ = BatchBeamSearch
                #             logging.info("BatchBeamSearch implementation is selected.")
                #     else:
                #         logging.warning(
                #             f"As non-batch scorers {non_batch} are found, "
                #             f"fall back to non-batch implementation."
                #         )

            else:
                beam_search = BeamSearch(
                    beam_size=beam_size,
                    weights=weights,
                    scorers=scorers,
                    sos=asr_model.sos,
                    eos=asr_model.eos,
                    vocab_size=len(token_list),
                    token_list=token_list,
                    pre_beam_score_key=None if ctc_weight == 1.0 else "full",
                    return_hs=True
                    if asr_model.enc_context_method == "concatrelevance"
                    else False,
                )

                # TODO(karita): make all scorers batchfied
                if batch_size == 1:
                    non_batch = [
                        k
                        for k, v in beam_search.full_scorers.items()
                        if not isinstance(v, BatchScorerInterface)
                    ]
                    if len(non_batch) == 0:
                        if streaming:
                            beam_search.__class__ = BatchBeamSearchOnlineSim
                            beam_search.set_streaming_config(asr_train_config)
                            logging.info(
                                "BatchBeamSearchOnlineSim implementation is selected."
                            )
                        else:
                            beam_search.__class__ = BatchBeamSearch
                            logging.info("BatchBeamSearch implementation is selected.")
                    else:
                        logging.warning(
                            f"As non-batch scorers {non_batch} are found, "
                            f"fall back to non-batch implementation."
                        )

            beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logging.info(f"Beam_search: {beam_search}")
            logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif (
            token_type == "bpe"
            or token_type == "hugging_face"
            or "whisper" in token_type
        ):
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        if bpemodel not in ["whisper_en", "whisper_multilingual"]:
            converter = TokenIDConverter(token_list=token_list)
        else:
            converter = OpenAIWhisperTokenIDConverter(model_type=bpemodel)
            beam_search.set_hyp_primer(
                list(converter.tokenizer.sot_sequence_including_notimestamps)
            )
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.hugging_face_model = hugging_face_model
        self.hugging_face_linear_in = hugging_face_linear_in
        self.hugging_face_beam_size = beam_size
        self.hugging_face_decoder_max_length = hugging_face_decoder_max_length
        self.maxlenratio = maxlenratio
        self.orig_maxlenratio = maxlenratio
        self.orig_minlenratio = minlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.enh_s2t_task = enh_s2t_task
        self.multi_asr = multi_asr
        self.han = True if "han" in asr_model_file else False
        self.block_relevance = (
            1 if asr_model.enc_context_method == "concatrelevance" else None
        )

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Optional[torch.Tensor] = None,
        binary_relevance: Optional[torch.Tensor] = None,
        block_id: Optional[int] = None,
        final_block: Optional[bool] = False,
    ) -> List[
        Tuple[
            Optional[str],
            List[str],
            List[int],
            Union[Hypothesis, ExtTransHypothesis, TransHypothesis],
        ]
    ]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        # if isinstance(speech, np.ndarray):
        #     speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.to(getattr(torch, self.dtype))
        # lengths: (1,)
        # lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {
            "speech": speech,
            "speech_lengths": speech_lengths.long(),
            "block_id": block_id,
            "final_block": final_block,
        }
        if binary_relevance:
            batch["binary_relevance"] = binary_relevance
        logging.info(
            f"speech length: {str(speech.size(1))} Block {block_id} Final Block {final_block}"
        )

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        if "binary_relevance" in batch and batch["binary_relevance"] is not None:
            enc, _, _ = self.asr_model.encode(**batch)
        else:
            enc, _ = self.asr_model.encode(**batch)

        if self.block_relevance is not None and block_id > 0:
            # logging.info(f"Block relevance: {self.asr_model.gating_factor}")
            self.block_relevance = 1 if self.asr_model.gating_factor[0] > 0.5 else 0

        if self.block_relevance is None:
            if not (block_id == 0 or final_block):
                return []
        elif self.block_relevance == 0 and not final_block:
            return []

        if self.multi_asr:
            enc = enc.unbind(dim=1)  # (batch, num_inf, ...) -> num_inf x [batch, ...]
        if self.enh_s2t_task or self.multi_asr:
            # Enh+ASR joint task or Multispkr ASR task
            # NOTE (Wangyou): the return type in this case is List[default_return_type]
            if self.multi_asr:
                num_spk = getattr(self.asr_model, "num_inf", 1)
            else:
                num_spk = getattr(self.asr_model.enh_model, "num_spk", 1)
            assert len(enc) == num_spk, (len(enc), num_spk)
            results = []
            for spk, enc_spk in enumerate(enc, 1):
                logging.info(f"=== [{str(self.asr_model.__class__)}] Speaker {spk} ===")
                if isinstance(enc_spk, tuple):
                    enc_spk = enc_spk[0]
                assert len(enc_spk) == 1, len(enc_spk)

                # c. Passed the encoder result and the beam search
                ret = self._decode_single_sample(enc_spk[0])
                assert check_return_type(ret)
                results.append(ret)

        else:
            # Normal ASR
            if isinstance(enc, tuple):
                enc = enc[0]
            assert len(enc) == 1, len(enc)

            # c. Passed the encoder result and the beam search
            results = self._decode_single_sample(enc[0])
            assert check_return_type(results), f"Error in the return type: {results}"

        return results

    def _decode_single_sample(self, enc: torch.Tensor):
        if self.beam_search_transducer:
            logging.info("encoder output length: " + str(enc.shape[0]))
            nbest_hyps = self.beam_search_transducer(enc)

            best = nbest_hyps[0]
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(
                f"normalized log probability: {best.score / len(best.yseq):.2f}"
            )
            logging.info(
                "best hypo: " + "".join(self.converter.ids2tokens(best.yseq[1:])) + "\n"
            )
        elif self.hugging_face_model:
            decoder_start_token_id = (
                self.hugging_face_model.config.decoder_start_token_id
            )
            yseq = self.hugging_face_model.generate(
                encoder_outputs=ModelOutput(
                    last_hidden_state=self.hugging_face_linear_in(enc).unsqueeze(0)
                ),
                use_cache=True,
                decoder_start_token_id=decoder_start_token_id,
                num_beams=self.hugging_face_beam_size,
                max_length=self.hugging_face_decoder_max_length,
            )
            nbest_hyps = [Hypothesis(yseq=yseq[0])]
            logging.info(
                "best hypo: "
                + "".join(self.converter.ids2tokens(nbest_hyps[0].yseq[1:]))
                + "\n"
            )
        else:
            logging.info(f"Enc shape = {enc.shape} Device {enc.device}")
            if hasattr(self.beam_search.nn_dict, "decoder"):
                if isinstance(self.beam_search.nn_dict.decoder, S4Decoder):
                    # Setup: required for S4 autoregressive generation
                    for module in self.beam_search.nn_dict.decoder.modules():
                        if hasattr(module, "setup_step"):
                            module.setup_step()
            if not self.han:
                nbest_hyps = self.beam_search(
                    x=enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
                )
            else:
                nbest_hyps = self.beam_search(
                    x=enc,
                    maxlenratio=self.maxlenratio,
                    minlenratio=self.minlenratio,
                    context=self.asr_model.enc_context[-1],
                )

        nbest_hyps = nbest_hyps[: self.nbest]

        if (
            self.asr_model.enc_context_method == "concatrelevance"
            and len(nbest_hyps) > 0
        ):
            if self.block_relevance == 1:
                hs = torch.stack(nbest_hyps[0].hs, dim=0).unsqueeze(0)
                self.asr_model.topic_vector = hs

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = None if self.asr_model.use_transducer_decoder else -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            if text == "":
                continue

            results.append((text, token, token_int, hyp))

        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2Text(**kwargs)


def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    transducer_conf: Optional[dict],
    streaming: bool,
    enh_s2t_task: bool,
    quantize_asr_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    hugging_face_decoder: bool,
    hugging_face_decoder_max_length: int,
    time_sync: bool,
    multi_asr: bool,
    block_size: int,
    audio_clip: int = None,
    block_relevance: str = None,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        transducer_conf=transducer_conf,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        enh_s2t_task=enh_s2t_task,
        multi_asr=multi_asr,
        quantize_asr_model=quantize_asr_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        hugging_face_decoder=hugging_face_decoder,
        hugging_face_decoder_max_length=hugging_face_decoder_max_length,
        time_sync=time_sync,
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    ## Get Block Level relance
    if block_relevance is not None:
        with open(block_relevance, "r") as f:
            block_level = {
                line.strip().split(" ")[0]: [
                    int(x) for x in line.strip().split(" ")[1:]
                ]
                for line in f.readlines()
            }
    else:
        block_level = None

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            batch = {k: v[0] for k, v in batch.items()}
            if audio_clip is not None:
                batch["speech"] = batch["speech"][:audio_clip]
            full_batch = batch.copy()

            speech_len = batch["speech"].shape[0]
            full_batch["speech"] = full_batch["speech"].unsqueeze(0)  # to(device)
            full_batch["speech_lengths"] = full_batch["speech"].new_full(
                [1], dtype=torch.long, fill_value=full_batch["speech"].size(1)
            )
            if block_relevance is not None:
                full_batch["block_relevance"] = torch.from_numpy(
                    np.array(block_level[keys[0]])
                )
            
            st_time = time.time()

            block_id = 0
            sample_index = 0
            speech2text.asr_model.block_length = 10
            speech2text.asr_model.block_size = block_size
            speech2text.asr_model.reset_batch()
            curr_relevance = [1]
            while sample_index < speech_len:
                # logging.info(f"Speech length: {speech_len} {full_batch['speech'].shape} Block{block_id} Sample Index {sample_index}")
                # Get a block of speech
                if full_batch["speech"].shape[1] - (sample_index + block_size) < 100:
                    block_inp = (
                        full_batch["speech"][:, sample_index:, :]
                        if full_batch["speech"].ndim == 3
                        else full_batch["speech"][:, sample_index:]
                    )
                else:
                    if full_batch["speech"].ndim == 3:
                        block_inp = (
                            full_batch["speech"][
                                :, sample_index : sample_index + block_size, :
                            ]
                            if sample_index + block_size < full_batch["speech"].shape[1]
                            else full_batch["speech"][:, sample_index:, :]
                        )
                    else:
                        block_inp = (
                            full_batch["speech"][
                                :, sample_index : sample_index + block_size
                            ]
                            if sample_index + block_size < full_batch["speech"].shape[1]
                            else full_batch["speech"][:, sample_index:]
                        )

                block_inp_lens = (block_inp.shape[1]) * torch.ones_like(
                    full_batch["speech_lengths"]
                ).long()

                for j, l in enumerate(block_inp_lens):
                    if full_batch["speech_lengths"][j] < sample_index + l:
                        l = sample_index + l - full_batch["speech_lengths"][j]

                if full_batch["speech"].shape[1] - (sample_index + block_size) < 100:
                    final_block = True
                else:
                    final_block = False

                batch["speech"] = block_inp
                batch["speech_lengths"] = block_inp_lens
                batch["block_id"] = block_id
                batch["final_block"] = final_block
                batch["binary_relevance"] = (
                    full_batch["block_relevance"][block_id]
                    if "block_relevance" in full_batch
                    and block_id < len(full_batch["block_relevance"])
                    else None
                )

                if (
                    ("cat" in output_dir or "rbass" in output_dir)
                    and "att" not in output_dir
                    and "han" not in output_dir
                    and block_id > 0
                ):
                    num_stored_blocks = max(
                        speech2text.asr_model.enc_context[-1][0].shape[1] / (249),
                        1,
                    )
                    if num_stored_blocks > 1:
                        speech2text.maxlenratio = max(
                            speech2text.orig_maxlenratio / num_stored_blocks, 0.1
                        ) if max(
                            speech2text.orig_maxlenratio / num_stored_blocks, 0.1
                        ) < 0.3 else 0.3 
                        speech2text.minlenratio = max(
                            speech2text.orig_minlenratio / num_stored_blocks, 0.04
                        )  # max(0.5**block_id*speech2text.orig_minlenratio,0.01)
                        logging.warning(
                            f"Updated at Block {block_id} | Maxlenratio {speech2text.maxlenratio:.3f} Minlenratio {speech2text.minlenratio:.3f} with NUM BLOCKS {num_stored_blocks}"
                        )
                else:
                    speech2text.maxlenratio = speech2text.orig_maxlenratio
                    speech2text.minlenratio = speech2text.orig_minlenratio
                # N-best list of (text, token, token_int, hyp_object)
                try:
                    results = speech2text(**batch)
                except TooShortUttError as e:
                    logging.warning(f"Utterance {keys} {e}")
                    hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                    results = [[" ", ["<space>"], [2], hyp]] * nbest
                    if enh_s2t_task:
                        num_spk = getattr(speech2text.asr_model.enh_model, "num_spk", 1)
                        results = [results for _ in range(num_spk)]

                # Only supporting batch_size==1
                key = keys[0]

                # Normal ASR
                if speech2text.block_relevance is not None:
                    curr_relevance.append(speech2text.block_relevance)

                if results != []:  # and (block_id == 0 or final_block):
                    for n, (text, token, token_int, hyp) in zip(
                        range(1, nbest + 1), results
                    ):
                        # Create a directory: outdir/{n}best_recog
                        ibest_writer = writer[f"{n}best_recog"]

                        # Write the result to each file
                        if text is not None:
                            if final_block:
                                ibest_writer["text"][key] = text
                                ibest_writer["token"][key] = " ".join(token)
                                ibest_writer["token_int"][key] = " ".join(
                                    map(str, token_int)
                                )
                                ibest_writer["score"][key] = str(hyp.score)
                                ibest_writer["block_relevance"][key] = " ".join(
                                map(str, curr_relevance)
                            )
                                end_time = time.time()
                                ibest_writer["decoding_time"][key] = str(end_time - st_time)

                            if block_id == 0:
                                ibest_writer[f"block{block_id+1}_text"][key] = text

                block_id += 1
                sample_index += block_inp_lens


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file",
        type=str,
        help="N-gram parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="enhancement and asr joint model",
    )
    group.add_argument(
        "--multi_asr",
        type=str2bool,
        default=False,
        help="multi-speaker asr model",
    )
    group.add_argument(
        "--block_size",
        type=int,
        default=1000,
        help="The block size for inference",
    )
    group.add_argument(
        "--audio_clip",
        type=int,
        default=None,
        help="The block size for inference",
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_asr_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to ASR model.",
    )
    group.add_argument(
        "--quantize_lm",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to LM.",
    )
    group.add_argument(
        "--quantize_modules",
        type=str,
        nargs="*",
        default=["Linear"],
        help="""List of modules to be dynamically quantized.
        E.g.: --quantize_modules=[Linear,LSTM,GRU].
        Each specified module should be an attribute of 'torch.nn', e.g.:
        torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, ...""",
    )
    group.add_argument(
        "--quantize_dtype",
        type=str,
        default="qint8",
        choices=["float16", "qint8"],
        help="Dtype for dynamic quantization.",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths."
        "If maxlenratio<0.0, its absolute value is interpreted"
        "as a constant max output length",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)
    group.add_argument("--hugging_face_decoder", type=str2bool, default=False)
    group.add_argument("--hugging_face_decoder_max_length", type=int, default=256)

    group.add_argument(
        "--transducer_conf",
        default=None,
        help="The keyword arguments for transducer beam search.",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--time_sync",
        type=str2bool,
        default=False,
        help="Time synchronous beam search.",
    )
    group.add_argument(
        "--block_relevance",
        type=str,
        default=None,
        help="Block level relevance",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
