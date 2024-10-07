"""Linear model classes."""

from typing import Tuple

import torch

from ... import data
from . import base


class LinearModule(base.BaseModule):
    """Simple linear embedding module."""

    pass


class LinearEncoder(LinearModule):
    def forward(
        self, source: data.PaddedTensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                encoded timesteps, and the RNN h0 and c0 cells.
        """
        return base.ModuleOutput(self.embed(source.padded))

    @property
    def name(self) -> str:
        return "linear"

    @property
    def output_size(self) -> int:
        return self.embedding_size
