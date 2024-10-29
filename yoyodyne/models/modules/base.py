"""Base module class with PL integration."""

import dataclasses
from typing import Optional, Union, Tuple

import lightning
import torch
from torch import nn

from ... import defaults


@dataclasses.dataclass
class ModuleOutput:
    """Output for forward passes."""

    output: torch.Tensor
    hiddens: Optional[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ] = None
    embeddings: Optional[torch.Tensor] = None

    @property
    def has_hiddens(self) -> bool:
        return self.hiddens is not None

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None


class BaseModule(lightning.LightningModule):
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
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            torch.Tensor: embedded tensor of shape B x seq_len x embed_dim.
        """
        embedded = self.embeddings(symbols)
        return self.dropout_layer(embedded)

    @property
    def output_size(self) -> int:
        raise NotImplementedError
