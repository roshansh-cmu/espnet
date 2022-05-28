# Copyright 2022 Roshan Sharma
# Copyright 2021 Tianzi Wang
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their original Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert


"""Encoder definition."""
import contextlib
import copy
import logging
import os
import torch
import yaml

from filelock import FileLock
from pathlib import Path
from typeguard import check_argument_types
from typing import Optional
from typing import Tuple

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class FairseqAVHubertEncoder(AbsEncoder):
    """FairSeq Hubert encoder module, used for loading pretrained weight and finetuning

    Args:
        input_size: input dim
        hubert_url: url to Hubert pretrained model
        hubert_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        output_size: dimension of attention
        normalize_before: whether to use layer_norm before the first block
        freeze_finetune_updates: steps that freeze all layers except output layer
            before tuning the whole model (nessasary to prevent overfit).
        dropout_rate: dropout rate
        activation_dropout: dropout rate in activation function
        attention_dropout: dropout rate in attention
    Hubert specific Args:
        Please refer to:
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/hubert/hubert.py
    """

    def __init__(
        self,
        input_size: int,
        avhubert_url: str = "./",
        avhubert_dir_path: str = "./",
        linear_units: int = 1024,
        output_size: int = 256,
        attention_heads: int = 12,
        num_blocks: int = 12,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        dropout_rate: float = 0.0,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mask_length: int = 10,
        mask_prob: float = 0.75,
        mask_selection: str = "static",
        mask_other: int = 0,
        apply_mask: bool = True,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_other: int = 0,
        mask_channel_selection: str = "static",
        layerdrop: float = 0.1,
        feature_grad_mult: float = 0.0,
        video_input_size: int = -1,
        video_frontend: str = "resnet",
        fuse_dimension: int = 1,
        convert_attention: bool = False,
        attention_type: str = "cos",
        pretrained_model_path: str = None,
        **kwargs,
    ):
        assert check_argument_types()
        super().__init__()
        self.apply_mask = apply_mask
        try:
            import fairseq
            from espnet2.avhubert.hubert import AVHubertModel
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        arg_overrides = {
            "dropout": dropout_rate,
            "activation_dropout": activation_dropout,
            "attention_dropout": attention_dropout,
            "mask_length": mask_length,
            "mask_prob": mask_prob,
            "mask_selection": mask_selection,
            "mask_other": mask_other,
            "mask_channel_length": mask_channel_length,
            "mask_channel_prob": mask_channel_prob,
            "mask_channel_selection": mask_channel_selection,
            "mask_channel_other": mask_channel_other,
            "encoder_layerdrop": layerdrop,
            "feature_grad_mult": feature_grad_mult,
            "data": avhubert_dir_path,
            "video_extractor": video_frontend,
            "video_feat_dim": video_input_size,
            "fuse_dimension": fuse_dimension,
            "masking_type": "feature",
            "audio_feat_dim": input_size,
        }

        if avhubert_url == "espnet":
            self.hubert_model_path = avhubert_dir_path
            s = torch.load(
                self.hubert_model_path,
                map_location=torch.device("cpu"),
            )

            if all("encoder.encoder" in k for k in s):
                try:
                    state = {
                        k.replace("encoder.encoder.", ""): v
                        for k, v in s.items()
                        if "label_embs_concat" not in k
                    }
                except Exception as e:
                    raise e

            config_file = os.path.join(
                "/".join(self.hubert_model_path.split("/")[:-1]),
                "config.yaml",
            )
            config_file = Path(config_file)

            with config_file.open("r", encoding="utf-8") as f:
                self.pretrained_cfg = yaml.safe_load(f)

            model = FairseqAVHubertPretrainEncoder(
                input_size=self.pretrained_cfg["input_size"],
                hubert_dict=self.pretrained_cfg["hubert_dict"],
                **self.pretrained_cfg["encoder_conf"],
            )
            model = model.encoder

            d = self.pretrained_cfg["encoder_conf"]["output_size"]
            self.pretrained_params = copy.deepcopy(state)

        else:

            self.hubert_model_path = avhubert_dir_path
            # download_avhubert(avhubert_url, avhubert_dir_path)

            (
                models,
                self.pretrained_cfg,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.hubert_model_path],
                arg_overrides=arg_overrides,
                strict=False,
            )
            model = models[0]

            d = self.pretrained_cfg.model.encoder_embed_dim
            self.pretrained_params = copy.deepcopy(model.state_dict())

        self._output_size = output_size

        if not isinstance(model, AVHubertModel):
            try:
                model = model.hubert_encoder.hubert_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'HubertModel, Hubertctc' classes, etc."
                )
                raise e

        self.encoders = model

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if output_size and output_size != d:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(d, output_size),
            )
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        video: torch.Tensor,
        video_length: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert ASR Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)
        vid_mask = make_pad_mask(video_length).to(video.device)
        vid_unmask = [x[: video_length[i], :] for i, x in enumerate(video)]
        video = pad_list(vid_unmask, pad_value=0.0)
        # logging.info(
        #     f"INP Forward AUDIO {masks.shape} AUDIO {xs_pad.shape} VID {vid_mask.shape} VID {video.shape} {max(video_length)} "
        # )
        # ys_pad = ys_pad[:, : min(ys_pad_length)]
        source = {"audio": xs_pad, "video": video}

        ft = self.freeze_finetune_updates <= self.num_updates

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning hubert parameters!")
        else:
            self.num_updates += 1
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                source,
                padding_mask=[masks, vid_mask],
                mask=self.apply_mask and self.training,
                features_only=True,
                output_layer=None,
        )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        masks = enc_outputs["padding_mask"]  # (B, T)

        # save gpu memory
        del enc_outputs

        olens = (~masks).sum(dim=1)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained Hubert model parameters reloaded!")


class FairseqAVHubertPretrainEncoder(AbsEncoder):
    """AVHubert pretrain encoder module, only used for pretraining stage

    Args:
        input_size: input dim
        output_size: dimension of attention
        linear_units: dimension of feedforward layers
        attention_heads: the number of heads of multi head attention
        num_blocks: the number of encoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        hubert_dict: target dictionary for Hubert pretraining
        label_rate: label frame rate. -1 for sequence label
        sample_rate: target sample rate.
        use_amp: whether to use automatic mixed precision
        normalize_before: whether to use layer_norm before the first block
    """

    def __init__(
        self,
        input_size: int = 104,
        output_size: int = 1024,
        linear_units: int = 1024,
        attention_heads: int = 12,
        num_blocks: int = 12,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        activation_dropout_rate: float = 0.0,
        hubert_dict: str = "./dict.txt",
        label_rate: int = 100,
        checkpoint_activations: bool = False,
        sample_rate: int = 16000,
        use_amp: bool = False,
        video_input_size: int = 2048,
        video_frontend: str = "linear",
        fuse_dimension: int = -1,
        convert_attention: bool = False,
        attention_type: str = "mha",
        attention_windows: list = None,
        attention_dilation: list = None,
        attention_mode: str = "tvm",
        pretrained_model_path: str = None,
        **kwargs,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.use_amp = use_amp
        try:
            from fairseq.data.dictionary import Dictionary
            from espnet2.avhubert.hubert import (
                AVHubertConfig,
                AVHubertModel,
                AVHubertPretrainingConfig,
            )
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        cfg_overides = {
            "encoder_embed_dim": output_size,
            "encoder_ffn_embed_dim": linear_units,
            "encoder_attention_heads": attention_heads,
            "encoder_layers": num_blocks,
            "final_dim": output_size,
            "dropout": dropout_rate,
            "attention_dropout": attention_dropout_rate,
            "label_rate": label_rate,
            "checkpoint_activations": checkpoint_activations,
            "video_extractor": video_frontend,
            "video_feat_dim": video_input_size,
            "fuse_dimension": fuse_dimension,
            "masking_type": "feature",
            "audio_feat_dim": input_size,
        }
        cfg_overides = {**cfg_overides, **kwargs}
        self.cfg = AVHubertConfig()
        orig_cfg = AVHubertConfig()

        for key, value in cfg_overides.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

        hubert_task_cfg = AVHubertPretrainingConfig()
        hubert_task_cfg_overides = {
            "label_rate": label_rate,
            "sample_rate": sample_rate,
            "fine_tuning": True if pretrained_model_path is not None else False,
        }
        for key, value in hubert_task_cfg_overides.items():
            if hasattr(hubert_task_cfg, key):
                setattr(hubert_task_cfg, key, value)

        d = Dictionary()
        self._build_dictionary(d, hubert_dict)

        self.encoder = AVHubertModel(self.cfg, hubert_task_cfg, self.dictionaries)

        if pretrained_model_path:
            import fairseq

            (
                models,
                self.pretrained_cfg,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [pretrained_model_path],
                strict=False,
            )
            model = models[0]
            keys_pt = ["encoder." + k for k, v in model.named_parameters()]
            keys_curr = [k for k, v in self.named_parameters()]
            keys_pt_load = [k for k in keys_curr if k in keys_pt]
            # print(f"Init from Scratch: {[k for k in keys_curr if k not in keys_pt]}")
            print(f"Init from PT: {keys_pt_load}")
            # print(f"Unused: {[k for k in keys_pt if k not in keys_curr]}")

            for name, parameter in self.encoder.named_parameters():
                dict_name = name  # ".".join(name.split(".")[1:])
                # print(name, dict_name)
                if "encoder." + name in keys_pt_load:
                    # print(
                    #     f"Name {dict_name} Shape {model.state_dict()[dict_name].shape}"
                    # )
                    if (
                        parameter.dtype == model.state_dict()[dict_name].dtype
                        and parameter.shape == model.state_dict()[dict_name].shape
                    ):
                        parameter = model.state_dict()[dict_name]
                    else:
                        print(
                            f"Skipped Name:{name}, {parameter.dtype==model.state_dict()[dict_name].dtype} {parameter.shape==model.state_dict()[dict_name].shape}"
                        )
        else:
               print(name, name in model.state_dict())

        # if convert_longformer:
        #     import longformer
        #     from longformer.longformer import LongformerConfig

        #     config = LongformerConfig(
        #         attention_window=attention_windows,
        #         attention_dilation=attention_dilation,
        #         autoregressive=False,
        #         num_attention_heads=attention_heads,
        #         hidden_size=output_size,
        #         attention_probs_dropout_prob=dropout_rate,
        #         attention_mode=attention_mode,
        #     )
        #     for i, layer in enumerate(self.encoder.encoder.layers):
        #         longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        #         longformer_self_attn.query = layer.attention.self.query
        #         longformer_self_attn.key = layer.attention.self.key
        #         longformer_self_attn.value = layer.attention.self.value

        #         longformer_self_attn.query_global = copy.deepcopy(
        #             layer.attention.self.query
        #         )
        #         longformer_self_attn.key_global = copy.deepcopy(
        #             layer.attention.self.key
        #         )
        #         longformer_self_attn.value_global = copy.deepcopy(
        #             layer.attention.self.value
        #         )

        #         layer.attention.self = longformer_self_attn
        # self.encoder = AVHubertModel(self.cfg, hubert_task_cfg, self.dictionaries)

    def _build_dictionary(self, dictionary, hubert_dict_path):
        if os.path.exists(f"{hubert_dict_path}"):
            setattr(dictionary, "symbols", [])
            setattr(dictionary, "count", [])
            setattr(dictionary, "indices", {})
            dictionary.add_from_file(f"{hubert_dict_path}")
        else:
            dictionary.add_symbol("0")

        self.dictionaries = [dictionary]

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_length: torch.Tensor,
        video: torch.Tensor,
        video_length: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert Pretrain Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        self.cast_mask_emb()
        masks = make_pad_mask(ilens).to(xs_pad.device)
        vid_mask = make_pad_mask(video_length).to(video.device)
        vid_unmask = [x[: video_length[i], :] for i, x in enumerate(video)]
        video = pad_list(vid_unmask, pad_value=0.0)
        # logging.info(
        #     f"INP Forward AUDIO {masks.shape} AUDIO {xs_pad.shape} VID {vid_mask.shape} VID {video.shape} {max(video_length)} "
        # )
        # ys_pad = ys_pad[:, : min(ys_pad_length)]
        source = {"audio": xs_pad, "video": video}
        # padding_masks = {"audio":masks,"video":vid_masks}
        enc_outputs = self.encoder(
            source,
            padding_mask=[masks, vid_mask],
            mask=True,
            target_list=[ys_pad],
            features_only=False,
        )
        return enc_outputs

    def cast_mask_emb(self):
        if self.use_amp and self.encoder.mask_emb.dtype != torch.cuda.HalfTensor:
            self.encoder.mask_emb = torch.nn.Parameter(self.encoder.mask_emb.half())

    def reload_pretrained_parameters(self):
        self.encoder.mask_emb = torch.nn.Parameter(
            torch.HalfTensor(self.cfg.encoder_embed_dim).uniform_()
        )
        logging.info(
            f"Hubert mask embedding re-initiallized!, \
            {self.encoder.mask_emb.dtype}, \
            {self.use_amp}"
        )


# def download_avhubert(model_url, dir_path):
#     os.makedirs(dir_path, exist_ok=True)

#     model_name = model_url.split("/")[-1]
#     model_path = os.path.join(dir_path, model_name)

#     with FileLock(model_path + ".lock"):
#         if not os.path.exists(model_path):
#             torch.hub.download_url_to_file(model_url, model_path)
#             logging.info(f"Hubert model downloaded {model_path}")
#         else:
#             logging.info(f"Hubert model {model_path} already exists.")

#     return model_path
