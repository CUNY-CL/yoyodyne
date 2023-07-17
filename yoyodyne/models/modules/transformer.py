"""Transformer model classes."""

import math
from typing import Optional

import torch
from torch import nn

from ... import data
from . import base


class PositionalEncoding(nn.Module):
    """Positional encoding.

    After:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    # Model arguments.
    pad_idx: int

    def __init__(
        self,
        d_model: int,
        pad_idx,
        max_source_length: int,
    ):
        """
        Args:
            d_model (int).
            pad_idx (int).
            max_source_length (int).
        """
        super().__init__()
        self.pad_idx = pad_idx
        positional_encoding = torch.zeros(max_source_length, d_model)
        position = torch.arange(
            0, max_source_length, dtype=torch.float
        ).unsqueeze(1)
        scale_factor = -math.log(10000.0) / d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * scale_factor
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self, symbols: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            symbols (torch.Tensor): symbol indices to encode B x seq_len.
            mask (torch.Tensor, optional): defaults to None; optional mask for
                positions not to be encoded.
        Returns:
            torch.Tensor: positional embedding.
        """
        out = self.positional_encoding.repeat(symbols.size(0), 1, 1)
        if mask is not None:
            # Indices should all be 0's until the first unmasked position.
            indices = torch.cumsum(mask, dim=1)
        else:
            indices = torch.arange(symbols.size(1)).long()
        # Selects the tensors from `out` at the specified indices.
        out = out[torch.arange(out.shape[0]).unsqueeze(-1), indices]
        # Zeros out pads.
        pad_mask = symbols.ne(self.pad_idx).unsqueeze(2)
        out = out * pad_mask
        return out


class TransformerModule(base.BaseModule):
    """Base module for Transformer."""

    # Model arguments.
    attention_heads: int
    # Constructed inside __init__.
    esq: float
    module: nn.TransformerEncoder
    positional_encoding: PositionalEncoding

    def __init__(
        self,
        *args,
        attention_heads,
        max_source_length: int,
        **kwargs,
    ):
        """Initializes the module with attention.

        Args:
            *args: passed to superclass.
            attention_heads (int).
            max_source_length (int).
            **kwargs: passed to superclass.
        """
        super().__init__(
            *args,
            attention_heads=attention_heads,
            max_source_length=max_source_length,
            **kwargs,
        )
        self.attention_heads = attention_heads
        self.esq = math.sqrt(self.embedding_size)
        self.module = self.get_module()
        self.positional_encoding = PositionalEncoding(
            self.embedding_size, self.pad_idx, max_source_length
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


class TransformerEncoder(TransformerModule):
    def forward(self, source: data.PaddedTensor) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (data.PaddedTensor).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedding = self.embed(source.padded)
        output = self.module(embedding, src_key_padding_mask=source.mask)
        return base.ModuleOutput(output)

    def get_module(self) -> nn.TransformerEncoder:
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

    @property
    def output_size(self) -> int:
        return self.embedding_size

    @property
    def name(self) -> str:
        return "transformer"


class FeatureInvariantTransformerEncoder(TransformerEncoder):
    """Encoder for Transformer with feature invariance.

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

    @property
    def name(self) -> str:
        return "feature-invariant transformer"


class TransformerDecoder(TransformerModule):
    """Decoder for Transformer."""

    # Output arg.
    decoder_input_size: int
    # Constructed inside __init__.
    module: nn.TransformerDecoder

    def __init__(self, *args, decoder_input_size, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Performs single pass of decoder module.

        Args:
            encoder_hidden (torch.Tensor): source encoder hidden state, of
                shape B x seq_len x hidden_size.
            source_mask (torch.Tensor): encoder hidden state mask.
            target (torch.Tensor): current state of targets, which may be the
                full target, or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): target mask.

        Returns:
            torch.Tensor: torch tensor of decoder outputs.
        """
        target_embedding = self.embed(target)
        target_sequence_length = target_embedding.size(1)
        # -> seq_len x seq_len.
        causal_mask = self.generate_square_subsequent_mask(
            target_sequence_length
        ).to(self.device)
        # -> B x seq_len x d_model
        output = self.module(
            target_embedding,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
        )
        return base.ModuleOutput(output)

    def get_module(self) -> nn.TransformerDecoder:
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_input_size,
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

    @staticmethod
    def generate_square_subsequent_mask(length: int) -> torch.Tensor:
        """Generates the target mask so the model cannot see future states.

        Args:
            length (int): length of the sequence.

        Returns:
            torch.Tensor: mask of shape length x length.
        """
        return torch.triu(torch.full((length, length), -math.inf), diagonal=1)

    @property
    def output_size(self) -> int:
        return self.num_embeddings

    @property
    def name(self) -> str:
        return "transformer"
