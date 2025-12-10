"""Custom schedulers."""

import math
from typing import List

from torch import optim


class Dummy(optim.lr_scheduler.LRScheduler):
    """A dummy scheduler that holds learning rate constant.

    Args:
        optimizer: optimizer.
    """

    def __init__(self, optimizer: optim.Optimizer):
        super().__init__(optimizer)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.optimizer})"

    def get_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class WarmupInverseSquareRoot(optim.lr_scheduler.LambdaLR):
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
        *args: ignored.
        **kwargs: ignored.
    """

    warmup_epochs: int
    decay_factor: float

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        *args,
        **kwargs,
    ):
        self.warmup_epochs = warmup_epochs
        self.decay_factor = math.sqrt(warmup_epochs)
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch: int) -> float:
        """Computes the learning rate lambda at a given epoch.

        Args:
            epoch (int): current epoch.

        Returns:
            float: lr_lambda.
        """
        if epoch < self.warmup_epochs:
            # +1 in numerator avoids a zero-LR first epoch.
            return (epoch + 1) / self.warmup_epochs
        # +1 in base of exponent avoids an undefined operation (0 to a negative
        # exponent) in the unlikely case one is using this without warmup.
        return self.decay_factor * (epoch + 1) ** -0.5
