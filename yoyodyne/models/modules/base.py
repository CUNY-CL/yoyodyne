"""Base module class with PL integration."""

import abc

import lightning
from torch import nn

from ... import defaults


class BaseModule(abc.ABC, lightning.LightningModule):
    """Abstract base class for encoder and decoder modules.

    Unknown positional or keyword args from the superclass are ignored.

    Args:
        *args: ignored.
        dropout (float, optional): dropout probability.
        embedding_size (int, optional): the dimensionality of the embedding.
        **kwargs: ignored.
    """

    dropout: float
    dropout_layer: nn.Dropout
    embedding_size: int

    def __init__(
        self,
        *args,  # Ignored here.
        dropout: float = defaults.DROPOUT,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        **kwargs,  # Ignored here.
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
