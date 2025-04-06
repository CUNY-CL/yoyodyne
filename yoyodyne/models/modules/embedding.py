"""Embedding module class."""

from ... import data
from . import base

import torch


class EmbeddingEncoder(base.BaseModule):
    """Embeds the input tensor.

    This encoder produces non-contextual encoding of the input tensor simply
    by embedding it. No other work is done.
    """

    def forward(self, source: data.PaddedTensor) -> torch.Tensor:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors.

        Returns:
            torch.Tensor.
        """
        return self.embed(source.padded)

    @property
    def name(self) -> str:
        return "embedding"

    @property
    def output_size(self) -> int:
        return self.embedding_size
