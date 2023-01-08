"""Transformer model classes."""

import argparse
import math

import torch
from torch import nn

from . import base, positional_encoding


class Error(Exception):
    pass


class TransformerEncoderDecoder(base.BaseEncoderDecoder):
    """Transformer encoder-decoder."""

    # Model arguments.
    attention_heads: int
    max_sequence_length: int
    # Constructed inside __init__.
    esq: float
    source_embeddings: nn.Embedding
    target_embeddings: nn.Embedding
    positional_encoding: positional_encoding.PositionalEncoding
    log_softmax: nn.LogSoftmax
    encoder: nn.TransformerEncoder
    decoder: nn.TransformerDecoder
    classifier: nn.Linear

    def __init__(
        self,
        *,
        attention_heads=4,
        max_sequence_length=128,
        **kwargs,
    ):
        """Initializes the encoder-decoder with attention.

        Args:
            attention_heads (int).
            max_sequence_length (int).
            **kwargs: passed to superclass.
        """
        super().__init__(**kwargs)
        self.attention_heads = attention_heads
        self.max_sequence_length = max_sequence_length
        self.esq = math.sqrt(self.embedding_size)
        self.source_embeddings = self.init_embeddings(
            self.vocab_size, self.embedding_size, self.pad_idx
        )
        self.target_embeddings = self.init_embeddings(
            self.output_size, self.embedding_size, self.pad_idx
        )
        self.positional_encoding = positional_encoding.PositionalEncoding(
            self.embedding_size, self.pad_idx, self.max_sequence_length
        )
        self.log_softmax = nn.LogSoftmax(dim=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.encoder_layers,
            norm=nn.LayerNorm(self.embedding_size),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.decoder_layers,
            norm=nn.LayerNorm(self.embedding_size),
        )
        self.classifier = nn.Linear(self.embedding_size, self.output_size)

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

    def source_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x sequence_length.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x sequence_length x embed_dim.
        """
        word_embedding = self.esq * self.source_embeddings(symbols)
        positional_embedding = self.positional_encoding(symbols)
        out = self.dropout_layer(word_embedding + positional_embedding)
        return out

    def target_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the target symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x sequence_length.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x sequence_length x embed_dim.
        """
        word_embedding = self.esq * self.target_embeddings(symbols)
        positional_embedding = self.positional_encoding(symbols)
        out = self.dropout_layer(word_embedding + positional_embedding)
        return out

    def encode(
        self, source: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (torch.Tensor).
            source_mask (torch.Tensor).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedding = self.source_embed(source)
        return self.encoder(embedding, src_key_padding_mask=source_mask)

    def decode(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the logits for each step of the output sequence.

        Args:
            encoder_hidden (torch.Tensor): source encoder hidden state, of
                shape B x sequence_length x hidden_size.
            source_mask (torch.Tensor): encoder hidden state mask.
            target (torch.Tensor): current state of targets, which may be the
                full target, or previous decoded, of shape
                B x sequence_length x hidden_size.
            target_mask (torch.Tensor): target mask.

        Returns:
            _type_: log softmax over targets.
        """
        target_embedding = self.target_embed(target)
        target_sequence_length = target_embedding.size(1)
        # -> sequence_length x sequence_length.
        causal_mask = self.generate_square_subsequent_mask(
            target_sequence_length
        ).to(self.device)
        # -> B x sequence_length x d_model
        decoder_hidden = self.decoder(
            target_embedding,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
        )
        # -> B x sequence_length x vocab_size.
        output = self.classifier(decoder_hidden)
        output = self.log_softmax(output)
        return output

    def _decode_greedy(
        self, encoder_hidden: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        # The output distributions to be returned.
        outputs = []
        batch_size = encoder_hidden.size(0)
        # The predicted symbols at each iteration.
        predictions = [
            torch.tensor(
                [self.start_idx for _ in range(encoder_hidden.size(0))],
                device=self.device,
            )
        ]
        # Tracking when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size, device=self.device)
        for _ in range(self.max_decode_length):
            target_tensor = torch.stack(predictions, dim=1)
            # Uses a dummy mask of all ones.
            target_mask = torch.ones_like(target_tensor, dtype=torch.float)
            target_mask = target_mask == 0
            output = self.decode(
                encoder_hidden, source_mask, target_tensor, target_mask
            )
            # We only care about the last prediction in the sequence.
            last_output = output[:, -1, :]
            outputs.append(last_output)
            pred = self._get_predicted(last_output.unsqueeze(1))
            predictions.append(pred.squeeze(1))
            # Updates to track which sequences have decoded an EOS.
            finished = torch.logical_or(
                finished, (predictions[-1] == self.end_idx)
            )
            # Break when all batches predicted an EOS symbol.
            if finished.all():
                break
        # -> B x vocab_size x sequence_length
        return torch.stack(outputs).transpose(0, 1).transpose(1, 2)

    def forward(self, batch: base.Batch) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (Tuple[torch.Tensor, ...]): Tuple of tensors in the batch
                of shape (source, source_mask, target, target_mask) during
                training or shape (source, source_mask) during inference.

        Returns:
            torch.Tensor.
        """
        # Training mode with targets.
        if len(batch) == 4:
            source, source_mask, target, target_mask = batch
            # Initializes the start symbol for decoding.
            starts = (
                torch.tensor(
                    [self.start_idx], device=self.device, dtype=torch.long
                )
                .repeat(target.size(0))
                .unsqueeze(1)
            )
            target = torch.cat((starts, target), dim=1)
            target_mask = torch.cat(
                (starts == self.pad_idx, target_mask), dim=1
            )
            encoder_hidden = self.encode(source, source_mask)
            output = self.decode(
                encoder_hidden, source_mask, target, target_mask
            )
            # -> B x vocab_size x sequence_length
            output = output.transpose(1, 2)[:, :, :-1]
        # No targets given at inference.
        elif len(batch) == 2:
            source, source_mask = batch
            encoder_hidden = self.encode(source, source_mask)
            # -> B x vocab_size x sequence_length.
            output = self._decode_greedy(encoder_hidden, source_mask)
        else:
            raise Error(f"Batch of {len(batch)} elements is invalid")
        return output

    @staticmethod
    def generate_square_subsequent_mask(length: int) -> torch.Tensor:
        """Generates the target mask so the model cannot see future states.

        Args:
            length (int): length of the sequence.

        Returns:
            torch.Tensor: mask of shape length x length.
        """
        return torch.triu(torch.full((length, length), -math.inf), diagonal=1)

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds transformer configuration options to the argument parser.

        These are only needed at training time.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--attention_heads",
            type=int,
            default=4,
            help="Number of attention heads "
            "(transformer-backed architectures only. Default: %(default)s.",
        )
        parser.add_argument(
            "--max_sequence_length",
            type=int,
            default=128,
            help="Maximum sequence length. Default: %(default)s.",
        )


class FeatureInvariantTransformerEncoderDecoder(TransformerEncoderDecoder):
    """Transformer encoder-decoder with feature invariance.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    # Indices.
    features_idx: int
    # Constructed inside __init__.
    type_embedding: nn.Embedding

    def __init__(self, *, features_idx, **kwargs):
        super().__init__(**kwargs)
        # Distinguishes features vs. character.
        self.features_idx = features_idx
        self.type_embedding = self.init_embeddings(
            2, self.embedding_size, self.pad_idx
        )

    def source_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols.

        This adds positional encodings and special embeddings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x sequence_length.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x sequence_length x embed_dim.
        """
        # Distinguishes features and chars.
        char_mask = (symbols < self.features_idx).long()
        # 1 or 0.
        type_embedding = self.esq * self.type_embedding(char_mask)
        word_embedding = self.esq * self.source_embeddings(symbols)
        positional_embedding = self.positional_encoding(
            symbols, mask=char_mask
        )
        out = self.dropout_layer(
            word_embedding + positional_embedding + type_embedding
        )
        return out
