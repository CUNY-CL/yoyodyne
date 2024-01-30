"""Base model class, with PL integration.

This also includes init_embeddings, which has to go somewhere.
"""

import dataclasses
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from ... import defaults


@dataclasses.dataclass
class ModuleOutput:
    """For tracking outputs of forward passes over varying architectures."""

    output: torch.Tensor
    hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    embeddings: Optional[torch.Tensor] = None

    @property
    def has_hiddens(self) -> bool:
        return self.hiddens is not None

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None


class BaseModule(pl.LightningModule):
    # Indices.
    pad_idx: int
    start_idx: int
    end_idx: int
    # Sizes.
    num_embeddings: int
    # Regularization arguments.
    dropout: float
    # Model arguments.
    embedding_size: int
    hidden_size: int
    layers: int
    # Constructed inside __init__.
    dropout_layer: nn.Dropout
    embeddings: nn.Embedding

    def __init__(
        self,
        *,
        pad_idx,
        start_idx,
        end_idx,
        num_embeddings,
        dropout=defaults.DROPOUT,
        embedding_size=defaults.EMBEDDING_SIZE,
        layers=defaults.ENCODER_LAYERS,
        hidden_size=defaults.HIDDEN_SIZE,
        **kwargs,  # Ignored.
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_embeddings = num_embeddings
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
        self.embeddings = self.init_embeddings(
            self.num_embeddings, self.embedding_size, self.pad_idx
        )

    @staticmethod
    def _xavier_embedding_initialization(
        num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embeddings layer using Xavier initialization.

        The pad embeddings are also zeroed out.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        embedding_layer = nn.Embedding(num_embeddings, embedding_size)
        # Xavier initialization.
        nn.init.normal_(
            embedding_layer.weight, mean=0, std=embedding_size**-0.5
        )
        # Zeroes out pad embeddings.
        if pad_idx is not None:
            nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
        return embedding_layer

    @staticmethod
    def _normal_embedding_initialization(
        num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embeddings layer from a normal distribution.

        The pad embeddings are also zeroed out.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        embedding_layer = nn.Embedding(num_embeddings, embedding_size)
        # Zeroes out pad embeddings.
        if pad_idx is not None:
            nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
        return embedding_layer

    @staticmethod
    def init_embeddings(
        num_embed: int, embed_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Method interface for initializing the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Raises:
            NotImplementedError: This method needs to be overridden.

        Returns:
            nn.Embedding: embedding layer.
        """
        raise NotImplementedError

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        embedded = self.embeddings(symbols)
        return self.dropout_layer(embedded)

    @property
    def output_size(self) -> int: ...
