"""Optimizer support.

Here we just put logic for initializing them."""

from typing import Iterable

from torch import optim

from . import defaults


_optimizers_fac = {
    "adadelta": optim.Adadelta,
    "adam": optim.Adam,
    "sgd": optim.SGD,
}
OPTIMIZERS = _optimizers_fac.keys()


def get_optimizer_cfg(
    optimizer: str,
    parameters: Iterable,
    learning_rate: str,
    beta1: float = defaults.BETA1,
    beta2: float = defaults.BETA2,
) -> optim.Optimizer:
    try:
        optimizer_cls = _optimizers_fac[optimizer]
    except KeyError:
        raise NotImplementedError(f"Optimizer not found: {optimizer}")
    if optimizer == "adam":
        return optimizer_cls(
            parameters, lr=learning_rate, betas=(beta1, beta2)
        )
    else:
        return optimizer_cls(parameters, lr=learning_rate)
