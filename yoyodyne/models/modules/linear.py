"""Linear module class."""

from ... import data
from . import base

import torch
from torch import nn


class LinearEncoder(base.BaseModule):
    """Embeds the input tensor, and then applies a affine projection.

    This produces a simple non-contextual encoding of the input tensor.

    Args:
        bidirectional (bool).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    linear: nn.Linear

    def __init__(self, *args, output_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(self.embedding_size, output_size)

    def forward(self, symbols: data.PaddedTensor) -> torch.Tensor:
        """Encodes the input.

        Args:
            symbols (data.PaddedTensor): padded tensor.

        Returns:
            torch.Tensor.
        """
        return self.linear(self.embed(symbols.padded))

    @property
    def name(self) -> str:
        return "linear"

    @property
    def output_size(self) -> int:
        return self.linear.out_features
