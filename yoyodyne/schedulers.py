"""Custom learning-rate schedulers.

All schedulers are assumed to use epochs rather than steps and we redefine the
classic linear-warmup inverse square-root scheduler in these terms.

The restriction to epoch-based schedulers is implemented in the base model's
`_get_lr_scheduler` method. To implement a step-based optimizer, this method
needs a special case to add the key-value pair `"interval": "step"` to the
scheduler_cfg` dictionary just in case a step-based optimizer is used.
"""

import argparse
from typing import Dict

import numpy
from torch import optim

from . import defaults, metrics

ALL_SCHEDULER_ARGS = [
    "warmup_epochs",
    "reduceonplateau_metric",
    "reduceonplateau_factor",
    "reduceonplateau_patience",
    "min_learning_rate",
    "check_val_every_n_epoch",
]


class WarmupInverseSquareRootSchedule(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to the learning rate over the
    warmup epochs, then decreases learning rate according to an inverse root
    square schedule.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.

    Args:
        optimizer (optim.Optimizer): optimizer.
        warmup_epochs (int): number of warmup epochs.
        **kwargs: ignored.
    """

    warmup_epochs: int
    decay_factor: float

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs,
        **kwargs,
    ):
        self.warmup_epochs = warmup_epochs
        self.decay_factor = numpy.sqrt(warmup_epochs)
        super().__init__(optimizer, self.lr_lambda)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.optimizer}, {self.warmup_epochs})"
        )

    def lr_lambda(self, epoch: int) -> float:
        """Computes the learning rate lambda at a given epoch.

        Args:
            epoch (int): current epoch.

        Returns:
            float: lr_lambda.
        """
        if epoch < self.warmup_epochs:
            # Adding 1 avoids the case where the initial LR is 0.
            return (1 + epoch) / self.warmup_epochs
        return self.decay_factor * epoch**-0.5


class ReduceOnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce on plateau scheduler.

    The following hyperparameters are inherited from the PyTorch defaults:
    threshold, threshold_mode, cooldown, eps.

    Args:
        optimizer (optim.Optimizer): optimizer.
        reduceonplateau_metric (str): reduces the LR when validation
            `accuracy` stops increasing or when validation `loss` stops
            decreasing.
        reduceonplateau_factor (float): factor by which the learning rate will
            be reduced: `new_lr *= factor`.
        reduceonplateau_patience (int): number of epochs with no
            improvement before reducing LR.
        min_learning_rate (float): lower bound on the learning rate.
        **kwargs: ignored.
    """

    def __init__(
        self,
        optimizer,
        reduceonplateau_metric,
        reduceonplateau_factor,
        reduceonplateau_patience,
        min_learning_rate,
        **kwargs,
    ):
        self.metric = metrics.ValidationMetric(reduceonplateau_metric)
        super().__init__(
            optimizer,
            factor=reduceonplateau_factor,
            min_lr=min_learning_rate,
            mode=self.metric.mode,
            patience=reduceonplateau_patience,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.optimizer}, {self.metric}, "
            f"{self.factor}, {self.patience}, {self.min_learning_rate})"
        )


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared configuration options to the argument parser.

    These are only needed at training time. Note that the actual scheduler
    arg is specified in models/base.py.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=defaults.WARMUP_EPOCHS,
        help="Number of warmup epochs (warmupinvsqrt scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--reduceonplateau_metric",
        type=str,
        choices=["loss", "accuracy"],
        default=defaults.REDUCEONPLATEAU_METRIC,
        help="Reduces the LR when validation `accuracy` stops increasing or "
        "when validation `loss` stops decreasing (reduceonplateau scheduler "
        "only. Default: %(default)s.",
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
