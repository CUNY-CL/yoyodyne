"""Base model class, with PL integration.

This also includes init_embeddings, which has to go somewhere.
"""

import argparse
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn

from ... import defaults


class BaseEncoder(pl.LightningModule):
    # Indices.
    pad_idx: int
    start_idx: int
    end_idx: int
    # Sizes.
    num_embeddings: int
    # Regularization arguments.
    dropout: float
    # Model arguments.
    decoder_size: int
    embedding_size: int
    hidden_size: int
    layers: int
    # Constructed inside __init__.
    dropout_layer: nn.Dropout

    def __init__(
        self,
        *,
        pad_idx,
        start_idx,
        end_idx,
        num_embeddings,
        decoder_size=0,
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
        self.decoder_size=decoder_size
        self.num_embeddings = num_embeddings
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)

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

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds shared configuration options to the argument parser.

        These are only needed at training time.

        Args:
            parser (argparse.ArgumentParser).
        """
        # Regularization arguments.
        parser.add_argument(
            "--encoder_layers",
            type=int,
            default=defaults.ENCODER_LAYERS,
            help="Number of encoder layers. Default: %(default)s.",
        )

    @property
    def output_size(self) -> int:
        return self.embedding_size
