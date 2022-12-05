"""Custom schedulers."""

import math

from torch import optim


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
            last_epoch (int, optional): last epoch for the scheduler.
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
