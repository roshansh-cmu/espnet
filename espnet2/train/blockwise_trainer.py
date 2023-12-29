"""Trainer module."""
import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torch.nn
import torch.optim
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.kwargs2args import kwargs2args
from espnet2.utils.types import str2bool

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

from torch.nn.functional import pad


autocast_args = dict()
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast

    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        autocast_args = dict(dtype=torch.bfloat16)
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


@dataclasses.dataclass
class BlockTrainerOptions(TrainerOptions):
    block_size: int
    audio_clip: int
    backprop_every_block: bool


class BlockTrainer(Trainer):
    """Trainer having a optimizer.

    If you'd like to use multiple optimizers, then inherit this class
    and override the methods if necessary - at least "train_one_epoch()"

    >>> class TwoOptimizerTrainer(Trainer):
    ...     @classmethod
    ...     def add_arguments(cls, parser):
    ...         ...
    ...
    ...     @classmethod
    ...     def train_one_epoch(cls, model, optimizers, ...):
    ...         loss1 = model.model1(...)
    ...         loss1.backward()
    ...         optimizers[0].step()
    ...
    ...         loss2 = model.model2(...)
    ...         loss2.backward()
    ...         optimizers[1].step()

    """

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> BlockTrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        assert check_argument_types()
        return build_dataclass(BlockTrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Reserved for future development of another Trainer"""
        parser.add_argument(
            "--backprop_every_block",
            type=str2bool,
            default=True,
            help="Whether to update model parameters every block.",
        )
        parser.add_argument(
            "--block_size",
            type=int,
            default=480000,
            help="Maximum number of warning shown",
        )
        parser.add_argument(
            "--audio_clip",
            type=int,
            default=960000,
            help="Maximum number of input audio frames",
        )

    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: BlockTrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        create_graph_in_tensorboard = options.create_graph_in_tensorboard
        distributed = distributed_option.distributed
        block_size = options.block_size
        audio_clip = options.audio_clip

        logging.info(f"block_size: {block_size} Audio Clip {audio_clip}")

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        torch.autograd.set_detect_anomaly(True)
        for iiter, (utt_id, batch) in enumerate(iterator, 1):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            if audio_clip is not None:
                # logging.warning(f"Before Clip {batch['speech'].shape} {audio_clip}")
                batch["speech"] = (
                    batch["speech"][:, :audio_clip, :]
                    if batch["speech"].ndim == 3
                    else batch["speech"][:, :audio_clip]
                )
                # logging.warning(f"After Clip {batch['speech'].shape} {audio_clip}")

            full_batch = batch.copy()
            block_idx = 0

            # model.reset_batch()

            # logging.warning(f"Reset the block {model.enc_context} {model.block_id} {full_batch['speech'].shape} {block_size}")
            if block_size is not None:
                sample_index = 0
                num_blocks = np.ceil(float(full_batch["speech"].shape[1]) / block_size)
                # logging.info(f"num_blocks: {num_blocks} Size of the input: {full_batch['speech'].shape[1]} Block size: {block_size}")
                while sample_index < full_batch["speech"].shape[1]:
                    if (
                            full_batch["speech"].shape[1] - (sample_index + block_size)
                            < 100
                        ):
                            block_inp = (
                                full_batch["speech"][:, sample_index:, ::]
                                if full_batch["speech"].ndim == 3
                                else full_batch["speech"][:, sample_index:]
                            )
                    else:
                        if full_batch["speech"].ndim == 3:
                            block_inp = (
                                full_batch["speech"][
                                    :, sample_index : sample_index + block_size, ::
                                ]
                                if sample_index + block_size
                                < full_batch["speech"].shape[1]
                                else full_batch["speech"][:, sample_index:, ::]
                            )
                        else:
                            block_inp = (
                                full_batch["speech"][
                                    :, sample_index : sample_index + block_size
                                ]
                                if sample_index + block_size
                                < full_batch["speech"].shape[1]
                                else full_batch["speech"][:, sample_index:]
                            )

                    ## PAD THE INPUT
                    if block_inp.shape[1] < block_size:
                        padlen = block_size - block_inp.shape[1]
                        pad_tensor = (
                            torch.zeros(
                                (block_inp.shape[0], padlen, block_inp.shape[2]),
                                device=block_inp.device,
                                dtype=block_inp.dtype,
                            )
                            if block_inp.ndim == 3
                            else torch.zeros(
                                (block_inp.shape[0], padlen),
                                device=block_inp.device,
                                dtype=block_inp.dtype,
                            )
                        )
                        block_inp = torch.cat((block_inp, pad_tensor), dim=1)

                    if (sample_index + block_size) >= full_batch["speech"].shape[
                        1
                    ] - 1:
                        final_block = True
                    else:
                        final_block = False

                    block_inp_lens = (block_inp.shape[1]) * torch.ones_like(
                        full_batch["speech_lengths"]
                    )

                    for j, l in enumerate(block_inp_lens):
                        if full_batch["speech_lengths"][j] < sample_index + l:
                            l = sample_index + l - full_batch["speech_lengths"][j]

                    batch["speech"] = block_inp
                    batch["speech_lengths"] = block_inp_lens
                    batch["block_id"] = block_idx
                    batch["final_block"] = final_block
                    
                    if "binary_relevance" in full_batch:
                        batch["binary_relevance"] = full_batch["binary_relevance"][
                            :, block_idx
                        ]
                        if torch.sum(torch.sum(batch["binary_relevance"])) == 0:
                            sample_index += block_inp.shape[1]
                            block_idx += 1
                            continue

                    with reporter.measure_time("iter_time"):
                        # logging.info(f"Block {block_idx} of {num_blocks} sample_index={sample_index}")
                        

                        batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
                        if no_forward_run:
                            all_steps_are_invalid = False
                            continue

                        if (
                            create_graph_in_tensorboard
                            and iiter == 1
                            and summary_writer is not None
                        ):
                            if distributed:
                                _model = getattr(model, "module")
                            else:
                                _model = model
                                if _model is not None:
                                    try:
                                        _args = kwargs2args(_model.forward, batch)
                                    except (ValueError, TypeError):
                                        logging.warning(
                                            "inpect.signature() is failed for the model. "
                                            "The graph can't be added for tensorboard."
                                        )
                                    else:
                                        try:
                                            summary_writer.add_graph(
                                                _model, _args, use_strict_trace=False
                                            )
                                        except Exception:
                                            logging.warning(
                                                "summary_writer.add_graph() "
                                                "is failed for the model. "
                                                "The graph can't be added for tensorboard."
                                            )
                                        del _args
                                else:
                                    logging.warning(
                                        "model.module is not found (This should be a bug.)"
                                    )
                            del _model

                        with autocast(
                            scaler is not None,
                            **autocast_args,
                        ):
                            with reporter.measure_time(f"forward_time"):
                                if block_idx == 0:
                                    retval = model(**batch, find_unused_parameters=True)
                                else:
                                    # logging.info(f"Block {block_idx} of {num_blocks} sample_index={sample_index}")
                                    retval = model(**batch, find_unused_parameters=False)
                                # Note(kamo):
                                # Supporting two patterns for the returned value from the model
                                #   a. dict type
                                if isinstance(retval, dict):
                                    loss = retval["loss"]
                                    stats = retval["stats"]
                                    weight = retval["weight"]
                                    optim_idx = retval.get("optim_idx")
                                    if optim_idx is not None and not isinstance(
                                        optim_idx, int
                                    ):
                                        if not isinstance(optim_idx, torch.Tensor):
                                            raise RuntimeError(
                                                "optim_idx must be int or 1dim torch.Tensor, "
                                                f"but got {type(optim_idx)}"
                                            )
                                        if optim_idx.dim() >= 2:
                                            raise RuntimeError(
                                                "optim_idx must be int or 1dim torch.Tensor, "
                                                f"but got {optim_idx.dim()}dim tensor"
                                            )
                                        if optim_idx.dim() == 1:
                                            for v in optim_idx:
                                                if v != optim_idx[0]:
                                                    raise RuntimeError(
                                                        "optim_idx must be 1dim tensor "
                                                        "having same values for all entries"
                                                    )
                                            optim_idx = optim_idx[0].item()
                                        else:
                                            optim_idx = optim_idx.item()

                                #   b. tuple or list type
                                else:
                                    loss, stats, weight = retval
                                    optim_idx = None

                            stats = {k: v for k, v in stats.items() if v is not None}
                            if ngpu > 1 or distributed:
                                # Apply weighted averaging for loss and stats
                                loss = (loss * weight.type(loss.dtype)).sum()

                                # if distributed, this method can also apply all_reduce()
                                stats, weight = recursive_average(
                                    stats, weight, distributed
                                )

                                # Now weight is summation over all workers
                                loss /= weight
                            if distributed:
                                # NOTE(kamo): Multiply world_size because DistributedDataParallel
                                # automatically normalizes the gradient by world_size.
                                loss *= torch.distributed.get_world_size()

                            loss /= accum_grad

                        reporter.register(stats, weight)

                        with reporter.measure_time(f"backward_time"):
                            if scaler is not None:
                                # Scales loss.  Calls backward() on scaled loss
                                # to create scaled gradients.
                                # Backward passes under autocast are not recommended.
                                # Backward ops run in the same dtype autocast chose
                                # for corresponding forward ops.
                                scaler.scale(loss).backward()
                            else:
                                # logging.info(f"Finish loss back prop {block_idx} of {num_blocks}")
                                loss.backward()

                                # if block_idx != num_blocks -1:
                                #     loss.backward(retain_graph=True)
                                # else:
                                #     loss.backward()

                        if iiter % accum_grad == 0:
                            if scaler is not None:
                                # Unscales the gradients of optimizer's assigned params in-place
                                for iopt, optimizer in enumerate(optimizers):
                                    if optim_idx is not None and iopt != optim_idx:
                                        continue
                                    scaler.unscale_(optimizer)

                            # gradient noise injection
                            if grad_noise:
                                add_gradient_noise(
                                    model,
                                    reporter.get_total_count(),
                                    duration=100,
                                    eta=1.0,
                                    scale_factor=0.55,
                                )

                            # compute the gradient norm to check if it is normal or not
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                max_norm=grad_clip,
                                norm_type=grad_clip_type,
                            )
                            # PyTorch<=1.4, clip_grad_norm_ returns float value
                            if not isinstance(grad_norm, torch.Tensor):
                                grad_norm = torch.tensor(grad_norm)

                            if not torch.isfinite(grad_norm):
                                logging.warning(
                                    f"The grad norm is {grad_norm}. Skipping updating the model."
                                )

                                # Must invoke scaler.update() if unscale_() is used in the iteration
                                # to avoid the following error:
                                #   RuntimeError: unscale_() has already been called
                                #   on this optimizer since the last update().
                                # Note that if the gradient has inf/nan values,
                                # scaler.step skips optimizer.step().
                                if scaler is not None:
                                    for iopt, optimizer in enumerate(optimizers):
                                        if optim_idx is not None and iopt != optim_idx:
                                            continue
                                        scaler.step(optimizer)
                                        scaler.update()

                            else:
                                all_steps_are_invalid = False
                                with reporter.measure_time("optim_step_time"):
                                    for iopt, (optimizer, scheduler) in enumerate(
                                        zip(optimizers, schedulers)
                                    ):
                                        if optim_idx is not None and iopt != optim_idx:
                                            continue
                                        if scaler is not None:
                                            # scaler.step() first unscales the gradients of
                                            # the optimizer's assigned params.
                                            scaler.step(optimizer)
                                            # Updates the scale for next iteration.
                                            scaler.update()
                                        else:
                                            optimizer.step()
                                        if isinstance(scheduler, AbsBatchStepScheduler):
                                            scheduler.step()
                            for iopt, optimizer in enumerate(optimizers):
                                if optim_idx is not None and iopt != optim_idx:
                                    continue
                                optimizer.zero_grad()

                            # Register lr and train/load time[sec/step],
                            # where step refers to accum_grad * mini-batch
                            reporter.register(
                                dict(
                                    {
                                        f"optim{i}_lr{j}": pg["lr"]
                                        for i, optimizer in enumerate(optimizers)
                                        for j, pg in enumerate(optimizer.param_groups)
                                        if "lr" in pg
                                    },
                                    train_time=time.perf_counter() - start_time,
                                ),
                            )
                            start_time = time.perf_counter()

                        # NOTE(kamo): Call log_message() after next()
                        reporter.next()
                        if iiter % log_interval == 0:
                            logging.info(reporter.log_message(-log_interval))
                            if summary_writer is not None:
                                reporter.tensorboard_add_scalar(
                                    summary_writer, -log_interval
                                )
                            if use_wandb:
                                reporter.wandb_log()

                    sample_index += block_inp.shape[1]
                    block_idx += 1
                    # logging.info(f"Updated block_idx: {block_idx} block_size={block_size}")

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: BlockTrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed
        block_size = options.block_size
        audio_clip = options.audio_clip

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for utt_id, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            if audio_clip is not None:
                batch["speech"] = batch["speech"][:, :audio_clip]

            full_batch = batch.copy()

            # model.reset_batch()


            batch["utt_id"] = utt_id
            if block_size is not None:
                sample_index = 0
                block_id = 0

                while sample_index < full_batch["speech"].shape[1]:
                    if (
                        full_batch["speech"].shape[1] - (sample_index + block_size)
                        < 1700
                    ):
                        block_inp = (
                            full_batch["speech"][:, sample_index:, ::]
                            if full_batch["speech"].ndim == 3
                            else full_batch["speech"][:, sample_index:]
                        )
                    else:
                        if full_batch["speech"].ndim == 3:
                            block_inp = (
                                full_batch["speech"][
                                    :, sample_index : sample_index + block_size, ::
                                ]
                                if sample_index + block_size
                                < full_batch["speech"].shape[1]
                                else full_batch["speech"][:, sample_index:, ::]
                            )
                        else:
                            block_inp = (
                                full_batch["speech"][
                                    :, sample_index : sample_index + block_size
                                ]
                                if sample_index + block_size
                                < full_batch["speech"].shape[1]
                                else full_batch["speech"][:, sample_index:]
                            )

                    ## PAD THE INPUT
                    if block_inp.shape[1] < block_size:
                        padlen = block_size - block_inp.shape[1]
                        pad_tensor = (
                            torch.zeros(
                                (block_inp.shape[0], padlen, block_inp.shape[2]),
                                device=block_inp.device,
                                dtype=block_inp.dtype,
                            )
                            if block_inp.ndim == 3
                            else torch.zeros(
                                (block_inp.shape[0], padlen),
                                device=block_inp.device,
                                dtype=block_inp.dtype,
                            )
                        )
                        block_inp = torch.cat((block_inp, pad_tensor), dim=1)

                    block_inp_lens = (block_inp.shape[1]) * torch.ones_like(
                        full_batch["speech_lengths"]
                    )

                    for j, l in enumerate(block_inp_lens):
                        if full_batch["speech_lengths"][j] < sample_index + l:
                            l = sample_index + l - full_batch["speech_lengths"][j]

                    if (sample_index + block_size) >= full_batch["speech"].shape[1] - 1:
                        final_block = True
                    else:
                        final_block = False

                    batch["speech"] = block_inp
                    batch["speech_lengths"] = block_inp_lens
                    batch["block_id"] = block_id
                    batch["final_block"] = final_block
                    if "binary_relevance" in full_batch:
                        
                        if block_id < full_batch["binary_relevance"].shape[1]:
                            batch["binary_relevance"] = full_batch["binary_relevance"][
                                :, block_id
                            ]
                        elif block_id == full_batch["binary_relevance"].shape[1]:
                            logging.info(f"Err {block_id} {final_block} {full_batch['binary_relevance'].shape}")

                    batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
                    if no_forward_run:
                        continue

                    retval = model(**batch)

                    if isinstance(retval, dict):
                        stats = retval["stats"]
                        weight = retval["weight"]
                    else:
                        _, stats, weight = retval

                    if ngpu > 1 or distributed:
                        # Apply weighted averaging for stats.
                        # if distributed, this method can also apply all_reduce()
                        stats, weight = recursive_average(stats, weight, distributed)

                    reporter.register(stats, weight)
                    reporter.next()

                    sample_index += block_size
                    block_id += 1

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
