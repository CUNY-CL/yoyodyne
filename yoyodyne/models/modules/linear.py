"""Linear module class."""

from ... import data
from . import base

import torch
from torch import nn


class LinearEncoder(base.BaseModule):
    """Embeds the input tensor, and then applies a linear projection.

    This produces a simple non-contextual encoding of the input tensor.
    """

    linear: nn.Linear

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)

    def forward(self, symbols: data.PaddedTensor) -> torch.Tensor:
        """Encodes the input.

        Args:
            symbols (data.PaddedTensor): padded tensor.

        Returns:
            torch.Tensor.
        """
        # TODO: averaging across length is a particular interpretation
        # and it's not clear it's the "right" thing to do here; test.
        embedded = self.embed(symbols.padded).mean(dim=1, keepdim=True)
        return self.linear(embedded)

    @property
    def name(self) -> str:
        return "linear"

    @property
    def output_size(self) -> int:
        return self.hidden_size
