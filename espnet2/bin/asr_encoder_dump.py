#!/usr/bin/env python3

"""
This file can be used to dump encoder or frontend features
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import time 
import numpy as np
import torch
from kaldiio import WriteHelper
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.cli_utils import get_commandline_args


class EncoderDump:
    """EncoderDump class

    Examples:
        >>> import soundfile
        >>> dump = EncoderDump("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> dump(audio)

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        mode: str = "frontend",
        feats_type: str = "last",
    ):
        assert check_argument_types()

        # 1. Build ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )

        asr_model.to(dtype=getattr(torch, dtype)).eval()

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.feats_type = feats_type
        print(f"Computing {self.feats_type} layer of {self.mode} and ASR model has frontend {self.asr_model.frontend is not None}")
        if self.feats_type == "multilayer":
            from s3prl.nn import S3PRLUpstream
            self.s3prl_model = S3PRLUpstream("wavlm_large")
            print(
                f"Num Layers {self.s3prl_model.num_layers} Hidden Sizes {self.s3prl_model.hidden_sizes} Downsample Factor {self.s3prl_model.downsample_rates}"
            )
        self.maxlen = 160000

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[Optional[str], List[str], List[int]]]:
        """Inference

        Args:
            data: Input speech data
        Returns:

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        full_speech = speech
        full_out = []
        for index in range(0, len(full_speech), self.maxlen):
            # data: (Nsamples,) -> (1, Nsamples)
            speech = (
                full_speech[index : index + self.maxlen]
                if index + self.maxlen < len(full_speech)
                else full_speech[index:]
            )
            if len(full_speech) - (index + self.maxlen) < 8000:
                speech = full_speech[index:]
                index = len(full_speech)
                
            speech = speech.unsqueeze(0).unsqueeze(-1).to(getattr(torch, self.dtype))
            # lengths: (1,)
            # logging.info(f"speech shape: {speech.shape}")
            lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
            batch = {"speech": speech, "speech_lengths": lengths}

            # a. To device
            batch = to_device(batch, device=self.device)

            # b. Forward Encoder
            if self.feats_type == "last":
                if self.mode != "frontend":
                    enc, _ = self.asr_model.encode(**batch)
                else:
                    enc, _ = self.asr_model._extract_feats(**batch)
            elif self.feats_type == "multilayer":
                enc, _ = self.s3prl_model(speech.squeeze(-1), lengths)
                # print(f"Shapes {[x.shape for x in enc]} {len(enc)}")
                enc = torch.stack(enc, dim=-2)
                # print(f"After stack {enc.shape}")
                enc = enc.view(1, enc.shape[1], -1)
                # print(f"After view {enc.shape}")
            assert len(enc) == 1, len(enc)
            full_out.append(enc.detach().cpu())
            torch.cuda.empty_cache()
            del enc 
            if index >= len(full_speech) or len(full_speech) - index < 2000:
                break
        output = (
            torch.cat(full_out, dim=1).squeeze(0)
        )
        return output

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

        return EncoderDump(**kwargs)


def dump(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    model_tag: Optional[str],
    allow_variable_data_keys: bool = False,
    mode: str = "frontend",
    feats_type: str = "multilayer",
    dump_dir: str = None,
):
    assert check_argument_types()
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

    # 2. Build dump
    dump_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        device=device,
        dtype=dtype,
        mode=mode,
    )
    encoder_dump = EncoderDump.from_pretrained(
        **dump_kwargs,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(encoder_dump.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(encoder_dump.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    i = 0
    dump_dir = dump_dir if dump_dir is not None else output_dir
    index = key_file.replace(".scp", "").split(".")[-1]
    fout_ark = os.path.join(dump_dir, f"feats.{index}.ark")
    fout_scp = os.path.join(output_dir, f"feats.{index}.scp")

    print(f"Writing into {fout_scp} {fout_ark}")
    with WriteHelper("ark,scp:{},{}".format(fout_ark, fout_scp)) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            logging.info(f"Key {keys[0]}")
            st_time = time.time()
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            try:
                enc_output = encoder_dump(**batch)
                enc_output = enc_output.cpu().numpy()
                print(f"Output shape {enc_output.shape}")
                writer(keys[0], enc_output)
                if i % 1 == 0:
                    logging.info(f"Wrote {i}")
                logging.info(f"Key {keys[0]} processed in {time.time()-st_time} seconds.")
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
            i += 1


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Encoder Dumping",
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
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("Encoder dump related related")
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
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group.add_argument(
        "--mode", type=str, choices=["frontend", "encoder"], default="frontend"
    )
    group.add_argument(
        "--feats_type", type=str, choices=["last", "multilayer"], default="last"
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--dump_dir", type=str_or_none)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    dump(**kwargs)


if __name__ == "__main__":
    main()
