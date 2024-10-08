"""Finds the best batch size given constraints."""

import argparse
from typing import Tuple

import lightning
import numpy
from lightning.pytorch.tuner import tuning

from . import data, defaults, models, util


class Error(Exception):
    pass


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
    trainer: lightning.Trainer,
    model: models.BaseModel,
    datamodule: data.DataModule,
    *,
    steps_per_trial: int = defaults.FIND_BATCH_SIZE_STEPS_PER_TRIAL,
) -> None:
    """Computes maximum or optimal batch size.

    This sets the batch size on the datamodule, and if necessary, the
    gradient accumulation steps on the trainer.

    With the max method, batch size is doubled until a memory error, thus
    finding the largest batch size that fits in memory.

    With the optimal method, batch size is doubled until a memory error or
    once we confirm that the desired batch size fits in memory (whichever
    comes first).

    Args:
        method (str): one of "max" (find the maximum batch size) or "opt" (find
            the "optimal" batch size, using the gradient accumulation trick if
            necessary.)
        trainer (lightning.Trainer).
        model (models.BaseModel).
        datamodule (data.DataModule).
        steps_per_trial (int, optional): number of steps to run with a given
            batch size.
    """
    tuner = tuning.Tuner(trainer)
    if method == "max":
        max_batch_size = tuner.scale_batch_size(
            model,
            datamodule=datamodule,
            steps_per_trial=steps_per_trial,
        )
        # This automatically overwrites the datamodule batch size argument.
        # So we don't need to.
    elif method == "opt":
        # This is a copy since it'll be automatically overwritten with the
        # maximum value.
        desired_batch_size = datamodule.batch_size
        if desired_batch_size <= 2:
            raise Error("--find_batch_size opt requires --batch_size > 2")
        max_batch_size = tuner.scale_batch_size(
            model,
            datamodule=datamodule,
            # This computes the maximum number of steps that'll be needed to
            # exceed the desired batch size. This seems like it would be "off
            # by one" but the way it's defined in the batch size finder is
            # itself "off by one" in the opposite direction, so it cancels out.
            max_trials=int(numpy.log2(desired_batch_size - 1)),
            steps_per_trial=steps_per_trial,
        )
        steps, batch_size = _optimal_batch_size(
            desired_batch_size, max_batch_size
        )
        datamodule.batch_size = batch_size
        util.log_info(f"Using optimal batch size: {datamodule.batch_size}")
        if steps != 1:
            trainer.accumulate_grad_batches = steps
            util.log_info(
                "Using gradient accumulation steps: "
                f"{trainer.accumulate_grad_batches}"
            )
    else:
        raise Error(f"Unknown batch sizing method: {method}")


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds batch sizing configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--find_batch_size",
        choices=["max", "opt"],
        help="Automatically find either the `max`(imum) or the `opt`(imal; "
        "i.e., via gradient accumulation) batch size.",
    )
    parser.add_argument(
        "--find_batch_size_steps_per_trial",
        type=int,
        default=defaults.FIND_BATCH_SIZE_STEPS_PER_TRIAL,
        help="Number of steps to run with a given batch size. "
        "Default: %(default)s.",
    )
