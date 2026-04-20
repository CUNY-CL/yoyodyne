"""Base module class with PL integration."""

import abc

import lightning
import torch
from torch import nn

from ... import data, defaults


class BaseModule(abc.ABC, lightning.LightningModule):
    """Abstract base class for encoder and decoder modules.

    Unknown positional or keyword args from the superclass are ignored.

    Args:
        *args: ignored.
        dropout (float, optional): dropout probability.
        embedding_size (int, optional).
        **kwargs: ignored.
    """

    dropout: float
    dropout_layer: nn.Dropout
    embedding_size: int

    def __init__(
        self,
        *args,
        dropout: float = defaults.DROPOUT,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.embedding_size = embedding_size

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def output_size(self) -> int: ...


class BaseEncoder(BaseModule):
    """Base class for encoder modules."""

    @abc.abstractmethod
    def forward(
        self,
        symbols: data.PaddedTensor,
        embeddings: nn.Embedding,
        *args,
        **kwargs,
    ) -> torch.Tensor: ...

    def set_max_length(self, max_length: int) -> None:
        """Sets maximum input length.

        This is no-op by default. Encoders with positional encodings (e.g.,
        transformers) should override this to resize their positional encoding
        table. This is called by BaseModel after encoders are injected, since
        the correct length is only known at that point.

        Args:
            max_length (int).
        """
        pass
