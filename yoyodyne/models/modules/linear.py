"""LSTM model classes."""

from typing import Tuple

import torch
from torch import nn

from ... import batches
from . import base


class LinearEncoder(base.BaseEncoder):
    """Simple linear embedding encoder."""

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

    def forward(
        self, source: batches.PaddedTensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes the input.

        Args:
            source (batches.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                encoded timesteps, and the LSTM h0 and c0 cells.
        """
        return self.embed(source.padded)
