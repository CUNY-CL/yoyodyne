"""Base module class with PL integration."""

import abc

import lightning

from ... import defaults


class BaseModule(abc.ABC, lightning.LightningModule):
    """Abstract base class for encoder and decoder modules.

    Unknown positional or keyword args from the superclass are ignored.

    Args:
        dropout (float, optional): dropout probability.
        embedding_size (int, optional): the dimensionality of the embedding.
        hidden_size (int, optional): size of the hidden layer.
        layers (int, optional): number of layers.
    """

    dropout: float
    embedding_size: int
    hidden_size: int
    layers: int

    def __init__(
        self,
        *args,  # Ignored here.
        dropout: float = defaults.DROPOUT,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        hidden_size: int = defaults.HIDDEN_SIZE,
        layers: int = defaults.LAYERS,
        **kwargs,  # Ignored here.
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def output_size(self) -> int: ...
