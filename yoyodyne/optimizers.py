"""Custom optimizers."""

from torch import optim

BETA1 = 0.9
BETA2 = 0.999


# The Adam family of optimizers use a pair of numerical coefficients in the
# range [0, 1), usually denoted by \beta or (\beta_1, \beta_2), to keep a
# running average of the gradients. Idiotically, the PyTorch implementation
# passes these hyperparameters as a tuple of two floats rather than as two
# separate floats. While this is already annoying, it is mainly a problem if
# one wishes to tune these two values using the W&B hyperparameter sampling
# system, because this engine samples scalar values, not tuples. Any way to
# convert the scalar to a tuple is necessarily going to be a hack.
#
# Here we minimally modify the optimizer constructors so that one passes
# `beta1: float` and `beta2: float` in place of `betas: Tuple[float, float]`.
# This is implemented by subclassing each one separately rather than
# fancy metaprogramming. The one complexity is that it is necesssary to
# explicitly copy the signature of the function up to but not including
# `betas`, which includes the parameter tensor and the default learning rate,
# which varies from optimizer to optimizer; the rest can be done safely with
# `*args` and `**kwargs.
#
# One can use these subclasses if separate betas are desired. For example
# use, see examples/wandb_sweeps.


class Adam(optim.Adam):
    """Adam optimizer with separate betas."""

    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = BETA1,
        beta2: float = BETA2,
        *args,
        **kwargs,
    ):
        super().__init__(params, lr, (beta1, beta2), *args, **kwargs)


class AdamW(optim.AdamW):
    """AdamW optimizer with separate betas."""

    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = BETA1,
        beta2: float = BETA2,
        *args,
        **kwargs,
    ):
        super().__init__(params, lr, (beta1, beta2), *args, **kwargs)


class Adamax(optim.Adamax):
    """Adamax optimizer with separate betas."""

    def __init__(
        self,
        params,
        lr: float = 0.002,
        beta1: float = BETA1,
        beta2: float = BETA2,
        *args,
        **kwargs,
    ):
        super().__init__(params, lr, (beta1, beta2), *args, **kwargs)


class NAdam(optim.NAdam):
    """Adamax optimizer with separate betas."""

    def __init__(
        self,
        params,
        lr: float = 0.002,
        beta1: float = BETA1,
        beta2: float = BETA2,
        *args,
        **kwargs,
    ):
        super().__init__(params, lr, (beta1, beta2), *args, **kwargs)


class RAdam(optim.RAdam):
    """RAdam optimizer with separate betas."""

    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = BETA1,
        beta2: float = BETA2,
        *args,
        **kwargs,
    ):
        super().__init__(params, lr, (beta1, beta2), *args, **kwargs)


class SparseAdam(optim.SparseAdam):
    """Sparse Adam optimizer with separate betas."""

    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = BETA1,
        beta2: float = BETA2,
        *args,
        **kwargs,
    ):
        super().__init__(params, lr, (beta1, beta2), *args, **kwargs)
