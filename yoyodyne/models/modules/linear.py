"""Linear model classes."""

from typing import Tuple

import torch
from torch import nn

from ... import data
from . import base


class LinearModule(base.BaseModule):
    """Simple linear embedding module."""

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        return self._normal_embedding_initialization(
            num_embeddings, embedding_size, pad_idx
        )


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
                encoded timesteps, and the LSTM h0 and c0 cells.
        """
        return base.ModuleOutput(self.embed(source.padded))

    @property
    def name(self) -> str:
        return "linear"

    @property
    def output_size(self) -> int:
        return self.embedding_size
