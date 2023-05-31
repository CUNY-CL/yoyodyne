"""LSTM model classes."""

import argparse
import heapq
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from . import base_encoder
from ... import batches, defaults


class LinearEncoder(base_encoder.BaseEncoder):
    """Simple linear embedding encoder."""

    # Constructed inside __init__.
    embeddings: nn.Embedding
    encoder: nn.LSTM
    h0: nn.Parameter
    c0: nn.Parameter

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            *args: passed to superclass.
            bidirectional (bool).
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        self.embeddings = self.init_embeddings(
            self.num_embeddings, self.embedding_size, self.pad_idx
        )

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

    def embed(self, symbols: torch.Tensor):
        embedded = self.embeddings(symbols)
        return self.dropout_layer(embedded)

    def _encode(self, embedding, *args, **kwargs):
        # Dummy function
        return embedding

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
        embedded = self.embed(source.padded)
        return embedded

