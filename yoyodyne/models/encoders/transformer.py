"""Transformer model classes."""

import argparse
import math

import torch
from torch import nn

from . import base_encoder, positional_encoding
from ... import batches, defaults


class TransformerEncoder(base_encoder.BaseEncoder):
    """Transformer encoder-decoder."""

    # Model arguments.
    attention_heads: int
    max_source_length: int
    # Constructed inside __init__.
    esq: float
    embeddings: nn.Embedding
    positional_encoding: positional_encoding.PositionalEncoding
    module: nn.TransformerEncoder

    def __init__(
        self,
        *args,
        attention_heads=defaults.ATTENTION_HEADS,
        max_source_length=defaults.MAX_SOURCE_LENGTH,
        **kwargs,
    ):
        """Initializes the encoder-decoder with attention.

        Args:
            attention_heads (int).
            max_source_length (int).
            *args: passed to superclass.
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        self.attention_heads = attention_heads
        self.max_source_length = max_source_length
        self.esq = math.sqrt(self.embedding_size)
        self.embeddings = self.init_embeddings(
            self.num_embeddings, self.embedding_size, self.pad_idx
        )
        self.positional_encoding = positional_encoding.PositionalEncoding(
            self.embedding_size, self.pad_idx, self.max_source_length
        )
        self.module = self.load_module()

    def load_module(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
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
        return self._xavier_embedding_initialization(
            num_embeddings, embedding_size, pad_idx
        )

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        word_embedding = self.esq * self.embeddings(symbols)
        positional_embedding = self.positional_encoding(symbols)
        out = self.dropout_layer(word_embedding + positional_embedding)
        return out

    def forward(self, source: batches.PaddedTensor) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (batches.PaddedTensor).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedding = self.embed(source.padded)
        return self.module(embedding, src_key_padding_mask=source.mask)


class TransformerDecoder(TransformerEncoder):
    def load_module(self):
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        return nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
        )

    def forward(
            self,
            encoder_hidden: torch.Tensor,
            source_mask: torch.Tensor,
            target: torch.Tensor,
            target_mask: torch.Tensor,
        ) -> torch.Tensor:
        target_embedding = self.embed(target)
        target_sequence_length = target_embedding.size(1)
        # -> seq_len x seq_len.
        causal_mask = self.generate_square_subsequent_mask(
            target_sequence_length
        ).to(self.device)
        # -> B x seq_len x d_model
        return self.module(
            target_embedding,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
        )

    @staticmethod
    def generate_square_subsequent_mask(length: int) -> torch.Tensor:
        """Generates the target mask so the model cannot see future states.

        Args:
            length (int): length of the sequence.

        Returns:
            torch.Tensor: mask of shape length x length.
        """
        return torch.triu(torch.full((length, length), -math.inf), diagonal=1)


class FeatureInvariantTransformerEncoder(TransformerEncoder):
    """Transformer encoder-decoder with feature invariance.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    features_vocab_size: int
    # Constructed inside __init__.
    type_embedding: nn.Embedding

    def __init__(self, *args, features_vocab_size, **kwargs):
        super().__init__(*args, **kwargs)
        # Distinguishes features vs. character.
        self.features_vocab_size = features_vocab_size
        self.type_embedding = self.init_embeddings(
            2, self.embedding_size, self.pad_idx
        )

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols.

        This adds positional encodings and special embeddings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        # Distinguishes features and chars.
        char_mask = (
            symbols < (self.num_embeddings - self.features_vocab_size)
        ).long()
        # 1 or 0.
        type_embedding = self.esq * self.type_embedding(char_mask)
        word_embedding = self.esq * self.embeddings(symbols)
        positional_embedding = self.positional_encoding(
            symbols, mask=char_mask
        )
        out = self.dropout_layer(
            word_embedding + positional_embedding + type_embedding
        )
        return out
