"""Pointer-generator model classes."""

import math
from typing import Optional, Tuple

import torch
from torch import nn

from .. import data
from . import lstm, modules


class Error(Exception):
    pass


class GenerationProbability(nn.Module):
    """Calculates the generation probability for a pointer generator."""

    stdev = 1 / math.sqrt(100)

    W_attention: nn.Linear
    W_hs: nn.Linear
    W_inp: nn.Linear
    bias: nn.Parameter

    def __init__(
        self, embedding_size: int, hidden_size: int, attention_size: int
    ):
        """Initializes the generation probability operator.

        Args:
            embedding_size (int): embedding dimensions.
            hidden_size (int): decoder hidden state dimensions.
            attention_size (int): dimensions of combined encoder attentions.
        """
        super().__init__()
        self.W_attention = nn.Linear(attention_size, 1, bias=False)
        self.W_hs = nn.Linear(hidden_size, 1, bias=False)
        self.W_inp = nn.Linear(embedding_size, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-self.stdev, self.stdev)

    def forward(
        self,
        h_attention: torch.Tensor,
        decoder_hs: torch.Tensor,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        # TODO(Adamits): improve documentation.
        """Computes Wh * ATTN_t + Ws * HIDDEN_t + Wy * Y_{t-1} + b.

        Args:
            h_attention (torch.Tensor): combined context vector over source and
                features of shape B x 1 x attention_size.
            decoder_hs (torch.Tensor): decoder hidden state of shape
                B x 1 x hidden_size.
            inp (torch.Tensor): decoder input of shape B x 1 x embedding_size.

        Returns:
            (torch.Tensor): generation probability of shape B.
        """
        # -> B x 1 x 1.
        p_gen = self.W_attention(h_attention) + self.W_hs(decoder_hs)
        p_gen += self.W_inp(inp) + self.bias.expand(h_attention.size(0), 1, -1)
        # -> B.
        p_gen = torch.sigmoid(p_gen.squeeze(1))
        return p_gen


class PointerGeneratorLSTMEncoderDecoder(lstm.LSTMEncoderDecoder):
    """Pointer-generator model with an LSTM backend.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    # Constructed inside __init__.
    geneneration_probability: GenerationProbability

    def __init__(self, *args, **kwargs):
        """Initializes the pointer-generator model with an LSTM backend."""
        super().__init__(*args, **kwargs)
        self._check_layer_sizes()
        # We use the inherited defaults for the source embeddings/encoder.
        # Overrides classifier to take larger input.
        if not self.has_features_encoder:
            self.classifier = nn.Linear(
                self.hidden_size + self.source_encoder.output_size,
                self.target_vocab_size,
            )
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.hidden_size,
                self.source_encoder.output_size,
            )
        else:
            self.merge_h = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.merge_c = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.features_attention = modules.attention.Attention(
                self.features_encoder.output_size, self.hidden_size
            )
            self.classifier = nn.Linear(
                self.hidden_size
                + self.source_encoder.output_size
                + self.features_encoder.output_size,
                self.target_vocab_size,
            )
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.hidden_size,
                self.source_encoder.output_size
                + self.features_encoder.output_size,
            )

    def get_decoder(self) -> modules.lstm.LSTMAttentiveDecoder:
        return modules.lstm.LSTMAttentiveDecoder(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            decoder_input_size=self.source_encoder.output_size
            + self.features_encoder.output_size
            if self.has_features_encoder
            else self.source_encoder.output_size,
            num_embeddings=self.target_vocab_size,
            dropout=self.dropout,
            bidirectional=False,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
            attention_input_size=self.source_encoder.output_size,
        )

    def _check_layer_sizes(self) -> None:
        """Checks that encoder and decoder layers are the same number.

        Raises:
            Error.
        """
        if self.encoder_layers != self.decoder_layers:
            raise Error(
                f"The number of encoder layers ({self.encoder_layers}) and "
                f"decoder layers ({self.decoder_layers}) must match"
            )

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        features_enc: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).
            features_enc (Optional[torch.Tensor]).
            features_mask (Optional[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        embedded = self.decoder.embed(symbol)
        last_h0, last_c0 = last_hiddens
        context, attention_weights = self.decoder.attention(
            last_h0.transpose(0, 1), source_enc, source_mask
        )
        if self.has_features_encoder:
            features_context, _ = self.features_attention(
                last_h0.transpose(0, 1), features_enc, features_mask
            )
            # -> B x 1 x 4*hidden_size.
            context = torch.cat([context, features_context], dim=2)
        _, (h, c) = self.decoder.module(
            torch.cat((embedded, context), 2), (last_h0, last_c0)
        )
        # -> B x 1 x hidden_size
        hidden = h[-1, :, :].unsqueeze(1)
        output_probs = self.classifier(torch.cat([hidden, context], dim=2))
        output_probs = nn.functional.softmax(output_probs, dim=2)
        # -> B x 1 x target_vocab_size.
        ptr_probs = torch.zeros(
            symbol.size(0),
            self.target_vocab_size,
            device=self.device,
            dtype=attention_weights.dtype,
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x target_vocab_size.
        ptr_probs.scatter_add_(
            2, source_indices.unsqueeze(1), attention_weights
        )
        # Probability of generating (from output_probs).
        gen_probs = self.generation_probability(
            context, hidden, embedded
        ).unsqueeze(2)
        ptr_scores = (1 - gen_probs) * ptr_probs
        gen_scores = gen_probs * output_probs
        scores = gen_scores + ptr_scores
        # Puts scores in log space.
        scores = torch.log(scores)
        return scores, (h, c)

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.
        """
        source_encoded, (h_source, c_source) = self.encode(
            batch.source, self.encoder
        )
        if self.beam_width is not None and self.beam_width > 1:
            # predictions = self.beam_decode(
            #     batch_size, x_mask, encoder_out, beam_width=self.beam_width
            # )
            raise NotImplementedError
        else:
            predictions = self.decode(
                len(batch),
                (h_source, c_source),
                batch.source.padded,
                source_encoded,
                batch.source.mask,
                self.teacher_forcing if self.training else False,
                batch.target.padded if batch.target else None,
            )
        # -> B x seq_len x target_vocab_size.
        predictions = predictions.transpose(0, 1)
        return predictions

    @staticmethod
    def _reshape_hiddens(
        h: torch.Tensor, c: torch.Tensor, layers: int, num_directions: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = h.view(layers, num_directions, h.size(1), h.size(2)).sum(axis=1)
        c = c.view(layers, num_directions, c.size(1), c.size(2)).sum(axis=1)
        return h, c

    @property
    def name(self) -> str:
        return "pointer-generator"