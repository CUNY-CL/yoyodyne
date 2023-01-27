"""Custom schedulers."""


import argparse
import math
from typing import Dict

from torch import optim


ALL_SCHEDULER_ARGS = [
    "warmup_steps",
    "start_factor",
    "end_factor",
    "total_decay_steps",
]


class WarmupInverseSquareRootSchedule(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to 1 over the warmup steps, then
    decreases learning rate from 1 to 0 using an inverse root square schedule
    over the remaining steps.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    warmup_steps: int
    last_epoch: int

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps,
        last_epoch=-1,
        **kwargs,
    ):
        """Initializes the LR scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.
            warmup_steps (int): number of warmup steps.
            last_epoch (int): last epoch for the scheduler.
            **kwargs: ignored.
        """
        self.warmup_steps = warmup_steps
        self.decay_factor = math.sqrt(warmup_steps)
        super(WarmupInverseSquareRootSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step: int) -> float:
        """Computes the learning rate lambda at a given step.

        Args:
            step (int): current step.

        Returns:
            float: lr_lambda.
        """
        if self.warmup_steps < 1:
            return self.decay_factor
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5


class LinearDecay(optim.lr_scheduler.LinearLR):
    """Linear decay scheduler."""

    def __init__(
        self,
        optimizer,
        start_factor,
        end_factor,
        total_decay_steps,
        **kwargs,
    ):
        """Initializes the LR scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.
            start_factor (float): the start_factor to multiply by the LR.
            end_factor (float): the end_factor to multiply by the LR
                after the total decay steps have finished.
            total_decay_steps (int): number of steps to linearly update
                the multiplied factor until end_factor.
            **kwargs: ignored.
        """
        super(LinearDecay, self).__init__(
            optimizer,
            total_iters=total_decay_steps,
            start_factor=start_factor,
            end_factor=end_factor,
        )


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared configuration options to the argument parser.

    These are only needed at training time.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--scheduler",
        choices=["warmupinvsqrt", "lineardecay"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps (warmupinvsqrt scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--start_factor",
        type=float,
        default=1 / 3,
        help="The starting multiplier for the LR "
        "(lineardecay scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--end_factor",
        type=float,
        default=1.0,
        help="The multiplier for the LR after total_decay_steps "
        "(lineardecay scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--total_decay_steps",
        type=int,
        default=5,
        help="The number of iterations until the LR multiplier "
        "reaches end_factor (lineardecay scheduler only). "
        "Default: %(default)s.",
    )


def get_scheduler_kwargs_from_argparse_args(**kwargs) -> Dict:
    """Gets the Dict of kwargs that will be used to instantiate the scheduler.

    Returns:
        Dict: hyperparameters for the scheduler.
    """
    return {k: kwargs.get(k) for k in ALL_SCHEDULER_ARGS}
