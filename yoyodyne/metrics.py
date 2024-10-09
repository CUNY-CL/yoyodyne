"""Metadata for validation metrics.

The actual computation of the metrics is handled either by the models
themselves (in the case of loss) or by the evaluators module.
"""

import argparse
import collections

from . import defaults


Metric = collections.namedtuple("Metric", ["filename", "mode", "monitor"])


_metric_fac = {
    "accuracy": Metric(
        "model-{epoch:03d}-{val_accuracy:.3f}", "max", "val_accuracy"
    ),
    "loss": Metric(
        "model-{epoch:03d}-{val_loss:.3f}",
        "min",
        "val_loss",
    ),
    "ser": Metric(
        "model-{epoch:03d}-{val_ser:.3f}",
        "min",
        "val_ser",
    ),
}


def get_metric(name: str) -> Metric:
    return _metric_fac[name]


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared configuration options to the argument parser.

    These are only needed at training time.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--checkpoint_metric",
        choices=_metric_fac.keys(),
        default=defaults.CHECKPOINT_METRIC,
        help="Selects checkpoints to maximize validation `accuracy`, "
        "or to minimize validation `loss` or `ser`. "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--patience_metric",
        choices=_metric_fac.keys(),
        default=defaults.PATIENCE_METRIC,
        help="Stops early when validation `accuracy` does not increase or "
        "when validation `loss` or `ser` does not decrease. "
        "Default: %(default)s.",
    )
