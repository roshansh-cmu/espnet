import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class S3prlFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        download_dir: str = None,
        multilayer_feature: bool = False,
<<<<<<< HEAD
        extracted_feature: bool = False,
        max_seq_len: int = None,
=======
        layer: int = -1,
>>>>>>> master
    ):
        try:
            import s3prl
            from s3prl.nn import Featurizer, S3PRLUpstream
        except Exception as e:
            print("Error: S3PRL is not properly installed.")
            print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
            raise e

        assert check_argument_types()
        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            logging.warning(
                "All the upstream models in S3PRL now only support 16 kHz audio."
            )

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert frontend_conf.get("upstream", None) in S3PRLUpstream.available_names()
        upstream = S3PRLUpstream(
            frontend_conf.get("upstream"),
            path_or_url=frontend_conf.get("path_or_url", None),
            normalize=frontend_conf.get("normalize", False),
            extra_conf=frontend_conf.get("extra_conf", None),
        )
        upstream.eval()
        if getattr(
            upstream, "model", None
        ) is not None and upstream.model.__class__.__name__ in [
            "Wav2Vec2Model",
            "HubertModel",
        ]:
            upstream.model.encoder.layerdrop = 0.0

        if layer != -1:
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when specific layer used"
        else:
            layer_selections = None
        featurizer = Featurizer(upstream, layer_selections=layer_selections)

        self.multilayer_feature = multilayer_feature
<<<<<<< HEAD
        self.upstream, self.featurizer = (
            upstream,
            featurizer,
        )
        self.pretrained_params = (
            copy.deepcopy(self.upstream.state_dict())
            if self.upstream is not None
            else None
        )
=======
        self.layer = layer
        self.upstream, self.featurizer = upstream, featurizer
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
>>>>>>> master
        self.frontend_type = "s3prl"
        self.hop_length = self.featurizer.downsample_rate
        self.tile_factor = frontend_conf.get("tile_factor", 1)
        self.extracted_feature = extracted_feature
        self.max_seq_len = max_seq_len
        logging.warning(f"Featurizer: {self.featurizer}")
        if extracted_feature:
            self.upstream = None

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def output_size(self) -> int:
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
<<<<<<< HEAD
        # logging.warning(f"Input: {input.shape} {input_lengths.shape} ")

        if not self.extracted_feature:
            feats, feats_lens = self.upstream(input, input_lengths)
        else:
            feats, feats_lens = input, input_lengths
            if self.max_seq_len:
                feats = feats[:, : self.max_seq_len, ::]
                feats_lens[feats_lens > self.max_seq_len] = self.max_seq_len
            if self.multilayer_feature:
                feats = feats.view(feats.size(0), feats.size(1), -1, 1024).permute(
                    2, 0, 1, 3
                )
                # logging.warning(
                #     f"Feats: {feats.shape}, feats_lengths: {feats_lens.shape}"
                # )
                feats_lens = [feats_lens for _ in range(feats.size(0))]
=======
        feats, feats_lens = self.upstream(input, input_lengths)
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

>>>>>>> master
        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        # logging.warning(
        #     f"Featurized Feats: {feats.shape}, feats_lengths: {feats_lens.shape}"
        # )
        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        if self.max_seq_len:
            feats = feats[: self.max_seq_len, ::]

        return feats, feats_lens

    def reload_pretrained_parameters(self):
        if self.upstream is not None:
            self.upstream.load_state_dict(self.pretrained_params)
            logging.info("Pretrained S3PRL frontend model parameters reloaded!")
