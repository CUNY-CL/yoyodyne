"""Custom learning-rate schedulers.

Additional schedulers will become available on the migration to LightingCLI.
"""

import argparse
from typing import Any, Dict

import numpy
from torch import optim

from . import defaults, metrics

SCHEDULER_ARGS = [
    "warmup_steps",
    "reduceonplateau_metric",
    "reduceonplateau_factor",
    "reduceonplateau_patience",
    "min_learning_rate",
    "check_val_every_n_epoch",
]


class WarmupInverseSquareRoot(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to the learning rate over the
    warmup steps, then decreases learning rate according to an inverse root
    square schedule.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.

    Args:
        optimizer (optim.Optimizer): optimizer.
        warmup_steps (int): number of warmup steps.
        *args: ignored.
        **kwargs: ignored.
    """

    warmup_steps: int
    decay_factor: float

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps,
        *args,
        **kwargs,
    ):
        self.warmup_steps = warmup_steps
        self.decay_factor = numpy.sqrt(warmup_steps)
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        """Computes the learning rate lambda at a given steps.

        Args:
            step (int): current step.

        Returns:
            float: lr_lambda.
        """
        if step < self.warmup_steps:
            # Adding 1 avoids the case where the initial LR is 0.
            return (1 + step) / self.warmup_steps
        return self.decay_factor * step**-0.5

    def config_dict(self) -> Dict[str, Any]:
        return {"interval": "step", "frequency": 1}


class ReduceOnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce on plateau scheduler.

    This is patched to make use of Yoyodyne's metrics library.

    Args:
        optimizer (optim.Optimizer): optimizer.
        check_val_every_n_epoch (int): frequency at which validation metrics
            are recomputed.
        reduceonplateau_metric (str): reduces the LR when validation
            `accuracy` stops increasing or when validation `loss` stops
            decreasing.
        *args: ignored.
        **kwargs: ignored.
    """

    check_val_every_n_epoch: int
    metric: metrics.Metric

    def __init__(
        self,
        optimizer: optim.Optimizer,
        check_val_every_n_epoch: int,
        reduceonplateau_factor: str,
        reduceonplateau_metric: str,
        *args,
        **kwargs,
    ):
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.metric = metrics.get_metric(reduceonplateau_metric)
        super().__init__(
            optimizer,
            factor=reduceonplateau_factor,
            mode=self.metric.mode,
        )

    def config_dict(self) -> Dict[str, Any]:
        return {
            "frequency": self.check_val_every_n_epoch,
            "interval": "epoch",
            "monitor": self.metric.monitor,
        }


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared configuration options to the argument parser.

    These are only needed at training time. Note that the actual scheduler
    arg is specified in models/base.py.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=defaults.WARMUP_STEPS,
        help="Number of warmup steps (warmupinvsqrt scheduler only). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--reduceonplateau_metric",
        type=str,
        choices=[
            "accuracy",
            "loss",
            "ser",
        ],
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


_scheduler_fac = {
    "reduceonplateau": ReduceOnPlateau,
    "warmupinvsqrt": WarmupInverseSquareRoot,
}
SCHEDULERS = _scheduler_fac.keys()


def get_scheduler_cfg(
    scheduler: str, optimizer: optim.Optimizer, *args, **kwargs
) -> Dict[str, Any]:
    try:
        scheduler_cls = _scheduler_fac[scheduler]
    except KeyError:
        raise NotImplementedError(f"Scheduler not found: {scheduler}")
    scheduler = scheduler_cls(optimizer, *args, **kwargs)
    config = scheduler.config_dict()
    # We also add the scheduler itself to the dictionary.
    config["scheduler"] = scheduler
    return config


def get_scheduler_kwargs_from_argparse_args(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Gets the Dict of kwargs that will be used to instantiate the scheduler.

    Args:
        args (argparse.Namespace).

    Returns:
        Dict: hyperparameters for the scheduler.
    """
    kwargs = vars(args)
    return {k: kwargs.get(k) for k in SCHEDULER_ARGS}
