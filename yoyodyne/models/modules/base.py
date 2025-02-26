"""Base module class with PL integration."""

import abc

import lightning
import torch
from torch import nn

from ... import defaults


class BaseModule(abc.ABC, lightning.LightningModule):
    """Abstract base class for encoder and decoder modules."""

    # Sizes.
    num_embeddings: int
    # Regularization arguments.
    dropout: float
    # Model arguments.
    embeddings: nn.Embedding
    embedding_size: int
    hidden_size: int
    layers: int
    # Constructed inside __init__.
    dropout_layer: nn.Dropout

    def __init__(
        self,
        *,
        embeddings,
        embedding_size,
        num_embeddings,
        dropout=defaults.DROPOUT,
        layers=defaults.ENCODER_LAYERS,
        hidden_size=defaults.HIDDEN_SIZE,
        **kwargs,  # Ignored.
    ):
        super().__init__()
        self.dropout = dropout
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            torch.Tensor: embedded tensor of shape B x seq_len x embed_dim.
        """
        return self.dropout_layer(self.embeddings(symbols))

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def output_size(self) -> int: ...
