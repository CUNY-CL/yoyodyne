"""Custom schedulers."""

import argparse
import math
from typing import Dict

from torch import optim

from . import defaults

ALL_SCHEDULER_ARGS = [
    "warmup_steps",
    "start_factor",
    "end_factor",
    "total_decay_steps",
    "reduceonplateau_mode",
    "reduceonplateau_factor",
    "reduceonplateau_patience",
    "min_learning_rate",
    "check_val_every_n_epoch",
]


class WarmupInverseSquareRootSchedule(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to the learning rate over the
    warmup steps, then decreases learning rate according to an inverse root
    square schedule.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    warmup_steps: int
    decay_factor: float

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps,
        **kwargs,
    ):
        """Initializes the LR scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.
            warmup_steps (int): number of warmup steps.
            **kwargs: ignored.
        """
        self.warmup_steps = warmup_steps
        self.decay_factor = math.sqrt(warmup_steps)
        super().__init__(optimizer, self.lr_lambda)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.optimizer}, {self.warmup_steps})"
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
        super().__init__(
            optimizer,
            total_iters=total_decay_steps,
            start_factor=start_factor,
            end_factor=end_factor,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.optimizer}, "
            f"{self.start_factor}, {self.end_factor}, "
            f"{self.total_decay_steps})"
        )


class ReduceOnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce on plateau scheduler."""

    def __init__(
        self,
        optimizer,
        reduceonplateau_mode,
        reduceonplateau_factor,
        reduceonplateau_patience,
        min_learning_rate,
        **kwargs,
    ):
        """Initializes the LR scheduler.

        The following hyperparameters are inherited from the PyTorch defaults:
        threshold, threshold_mode, cooldown, eps.

        Args:
            optimizer (optim.Optimizer): optimizer.
            reduceonplateau_mode (str): whether to reduce the LR when the
                validation loss stops decreasing ("loss") or when
                validation accuracy stops increasing ("accuracy").
            reduceonplateau_factor (float): factor by which the
                learning rate will be reduced: `new_lr *= factor`.
            reduceonplateau_patience (int): number of epochs with no
                improvement before reducing LR.
            min_learning_rate (float): lower bound on the learning rate.
            **kwargs: ignored.
        """
        super().__init__(
            optimizer,
            mode="min" if reduceonplateau_mode == "loss" else "max",
            factor=reduceonplateau_factor,
            patience=reduceonplateau_patience,
            min_lr=min_learning_rate,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.optimizer}, {self.mode}, "
            f"{self.factor}, {self.min_learning_rate}, {self.patience})"
        )


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared configuration options to the argument parser.

    These are only needed at training time.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--scheduler",
        choices=["warmupinvsqrt", "lineardecay", "reduceonplateau"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=defaults.WARMUP_STEPS,
        help="Number of warmup steps (warmupinvsqrt scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--start_factor",
        type=float,
        default=defaults.START_FACTOR,
        help="Starting multiplier for the LR (lineardecay scheduler "
        "only). Default: %(default)s.",
    )
    parser.add_argument(
        "--end_factor",
        type=float,
        default=defaults.END_FACTOR,
        help="Multiplier for the LR after --total_decay_steps (lineardecay "
        "scheduler only). Default: %(default)s.",
    )
    parser.add_argument(
        "--total_decay_steps",
        type=int,
        default=defaults.TOTAL_DECAY_STEPS,
        help="Number of iterations until the LR multiplier reaches "
        "--end_factor (lineardecay scheduler only). Default: %(default)s.",
    )
    parser.add_argument(
        "--reduceonplateau_mode",
        type=str,
        choices=["loss", "accuracy"],
        default=defaults.REDUCEONPLATEAU_MODE,
        help="Whether to reduce the LR when the validation loss stops "
        "decreasing (`loss`) or when validation accuracy stops increasing "
        "(`accuracy`) (reduceonplateau scheduler only). Default: %(default)s.",
    )
    parser.add_argument(
        "--reduceonplateau_factor",
        type=float,
        default=defaults.REDUCEONPLATEAU_FACTOR,
        help="Factor by which the learning rate will be reduced: "
        "new_lr = lr * factor (reduceonplateau scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--reduceonplateau_patience",
        type=int,
        default=defaults.REDUCEONPLATEAU_PATIENCE,
        help="Number of epochs with no improvement before "
        "reducing LR (reduceonplateau scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=defaults.MIN_LR,
        help="Lower bound on the learning rate (reduceonplateau "
        "scheduler only). Default: %(default)s.",
    )


def get_scheduler_kwargs_from_argparse_args(args: argparse.Namespace) -> Dict:
    """Gets the Dict of kwargs that will be used to instantiate the scheduler.

    Args:
        args (argparse.Namespace).

    Returns:
        Dict: hyperparameters for the scheduler.
    """
    kwargs = vars(args)
    return {k: kwargs.get(k) for k in ALL_SCHEDULER_ARGS}
