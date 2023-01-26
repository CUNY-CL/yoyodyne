"""Custom schedulers."""


import argparse
import math
import inspect
from typing import Dict

from torch import optim


ALL_SCHEDULER_ARGS = ["warmup_steps", "start_factor", "end_factor", "total_decay_steps"]


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
    ):
        """Initializes the LR scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.
            warmup_steps (int): number of warmup steps.
            last_epoch (int): last epoch for the scheduler.
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
        help="Start factor for lr decay. Default: %(default)s.",
    )
    parser.add_argument(
        "--end_factor",
        type=float,
        default=1.0,
        help="End factor for lr decay. Default: %(default)s.",
    )
    parser.add_argument(
        "--total_decay_steps",
        type=int,
        default=5,
        help="The number of iterations until linear decay factor "
        "reaches end_factor. "
        "Default: %(default)s.",
    )


def _map_scheduler_kwargs(k: str) -> Dict:
    """Maps the scheduler kwargs from argparse into the names expected by
    the actual scheduler.

    This is to avoid issues between the yoyodyne namespace,
    and the optim.lr_scheduler namespace.

    Args:
        k (str): name of the argparse argument

    Returns:
        Dict: scheduler kwargs in the optim.lr_scheduler namespace.
    """
    KWARGS_MAP = {"total_decay_steps": "total_iters"}

    return KWARGS_MAP[k] if k in KWARGS_MAP else k


def get_scheduler_kwargs_from_argparse_args(**kwargs) -> Dict:
    """Get's the Dict of kwargs that will be used to instantiate the scheduler.

    Returns:
        Dict: hyperparameters for the scheduler.
    """
    return {_map_scheduler_kwargs(k): kwargs.get(k) for k in ALL_SCHEDULER_ARGS}


def filter_scheduler_kwargs(scheduler_cls: optim.lr_scheduler, **kwargs) -> Dict:
    """Filter all scheduler kwargs from argparse to only those
    needed by the particular scheduler.

    Args:
        scheduler_cls (optim.lr_scheduler): class of the requested scheduler.

    Returns:
        Dict: kwargs for the requested scheduler.
    """
    valid_kwargs = [
        p for p in inspect.signature(scheduler_cls.__init__).parameters if p != "self"
    ]
    # Overrides scheduler defaults that are provided in **kwargs.
    return {k: kwargs[k] for k in valid_kwargs if kwargs.get(k) is not None}
