"""Finds the best batch size given constraints."""

import argparse
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.tuner import tuning

from . import data, defaults, models, util


class Error(Exception):
    pass


def _max_batch_size(
    trainer: pl.Trainer,
    model: models.BaseEncoderDecoder,
    datamodule: data.DataModule,
    *,
    mode: str = defaults.FIND_BATCH_SIZE_MODE,
    steps_per_trial: int = defaults.FIND_BATCH_SIZE_STEPS_PER_TRIAL,
    max_trials: int = defaults.FIND_BATCH_SIZE_MAX_TRIALS,
) -> int:
    """Computes the maximum batch size that will reliably fit in memory.

    Args:
        trainer (pl.Trainer).
        model (models.BaseEncoderDecoder).
        datamodule (data.DataModule).
        max_trials (int, optional): maximum number of increases in batch size
            before terminating.
        mode (str, optional): one of "binsearch", "power".
        steps_per_trial (int, optional): number of steps to run with a given
            batch size.

    Returns:
        int: estiamted maximum batch size.
    """
    tuner = tuning.Tuner(trainer)
    return tuner.scale_batch_size(
        model,
        datamodule=datamodule,
        max_trials=max_trials,
        mode=mode,
        steps_per_trial=steps_per_trial,
    )


def _optimal_batch_size(
    desired_batch_size: int, max_batch_size: int
) -> Tuple[int, int]:
    r"""Computes optimal batch size and number of gradient accumulation steps.

    Given the desired batch size $b$ and a max batch size $n_{\textrm{max}}$,
    this returns a tuple $a \in \mathcal{N}_{+}, n \in
    \mathcal{N}_{\le n_\textrm{max}}$ such that $n$ is the largest integer
    $\le n_\textrm{max}$ and $a n = b$.

    If desired batch size` is smaller than max batch size, then we just use
    desired batch size without multiple steps of gradient accumulation.

    Args:
        desired_batch_size (int).
        max_batch_size (int).

    Returns:
        Tuple[int, int]: the tuple (a, n) as defined above.
    """
    # If the max size is larger than the non-zero desired size, use the
    # desired size.
    if desired_batch_size <= max_batch_size:
        return 1, desired_batch_size
    # This simulates the desired batch size with multiple steps of gradient
    # accumulation per update. The solution given here is a brute force one,
    # but pilot experiments with a more elegant solution using the divisors was
    # no more efficient.
    for batch_size in range(max_batch_size + 1, 0, -1):
        if desired_batch_size % batch_size == 0:
            accum_steps = desired_batch_size // batch_size
            return accum_steps, batch_size


def find_batch_size(
    method: str,
    trainer: pl.Trainer,
    model: models.BaseEncoderDecoder,
    datamodule: data.DataModule,
    *,
    max_trials: int = defaults.FIND_BATCH_SIZE_MAX_TRIALS,
    mode: str = defaults.FIND_BATCH_SIZE_MODE,
    steps_per_trial: int = defaults.FIND_BATCH_SIZE_STEPS_PER_TRIAL,
) -> None:
    """Computes maximum or optimal batch size.

    This sets the batch size on the datamodule, and if necessary, the
    gradient accumulation steps on the trainer.

    Args:
        method (str): one of "max" (find the maximum batch size) or "opt" (find
            the "optimal" batch size, using the gradient accumulation trick if
            necessary.)
        trainer (pl.Trainer).
        model (models.BaseEncoderDecoder).
        datamodule (data.DataModule).
        max_trials (int, optional): maximum number of increases in batch size
            before terminating.
        mode (str, optional): one of "binsearch" or "power".
        steps_per_trial (int, optional): number of steps to run with a given
            batch size.
    """
    desired_batch_size = datamodule.batch_size  # Takes a copy.
    max_batch_size = _max_batch_size(
        trainer,
        model,
        datamodule,
        max_trials=max_trials,
        mode=mode,
        steps_per_trial=steps_per_trial,
    )
    util.log_info(f"Max batch size: {max_batch_size}")
    if method == "max":
        datamodule.batch_size = max_batch_size
    elif method == "opt":
        steps, batch_size = _optimal_batch_size(
            desired_batch_size, max_batch_size
        )
        datamodule.batch_size = batch_size
    else:
        raise Error(f"Unknown batch sizing method: {method}")
    util.log_info(f"Using batch size: {datamodule.batch_size}")
    if trainer.accumulate_grad_batches != 1:
        util.log_info(
            "Using gradient accumulation steps: "
            f"{trainer.accumulate_grad_batches}"
        )


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds batch sizing configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--find_batch_size",
        choices=["max", "opt"],
        help="Automatically find either the `max`(imum) or the `opt`(imal; "
        "i.e., via gradient accumulation) batch size. Default: not enabled.",
    )
    parser.add_argument(
        "--find_batch_size_mode",
        choices=["binsearch", "power"],
        default=defaults.FIND_BATCH_SIZE_MODE,
        help="Search strategy to update the batch size. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--find_batch_size_steps_per_trial",
        type=int,
        default=defaults.FIND_BATCH_SIZE_STEPS_PER_TRIAL,
        help="Number of steps to run with a given batch size. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--find_batch_size_max_trials",
        type=int,
        default=defaults.FIND_BATCH_SIZE_MAX_TRIALS,
        help="Maximum number of increases in batch size before terminating. "
        "Default: %(default)s",
    )
