"""LSTM model classes."""

import argparse
import heapq
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from . import attention, base_encoder, LSTMAttentiveDecoder, generation_probability
from ... import batches, defaults


class PointerGeneratorDecoder(LSTMAttentiveDecoder):
    # Constructed inside __init__.
    source_attention: attention.Attention
    geneneration_probability: generation_probability.GenerationProbability

    def __init__(self, *args, **kwargs):
        """Initializes the encoder-decoder with attention."""
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(self.decoder_size, self.hidden_size)
        self.classifier = nn.Linear(3 * self.hidden_size, self.output_size)
        self.generation_probability = (
            generation_probability.GenerationProbability(  # noqa: E501
                self.embedding_size, self.hidden_size, self.decoder_size
            )
        )
    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        embedded = self.embed(symbol)
        # -> 1 x B x decoder_dim.
        last_h, last_c = last_hiddens
        source_context, source_attention_weights = self.attention(
            last_h.transpose(0, 1), source_enc, source_mask
        )
        _, (h, c) = self.module(
            torch.cat((embedded, source_context), 2), (last_h, last_c)
        )
        # -> B x 1 x hidden_size
        hidden = h[-1, :, :].unsqueeze(1)
        output_probs = self.classifier(
            torch.cat([hidden, source_context], dim=2)
        )
        # Ordinary softmax, log will be taken at the end.
        output_probs = nn.functional.softmax(output_probs, dim=2)
        # -> B x 1 x output_size.
        ptr_probs = torch.zeros(
            symbol.size(0),
            self.output_size,
            device=self.device,
            dtype=source_attention_weights.dtype,
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x output_size.
        ptr_probs.scatter_add_(
            2, source_indices.unsqueeze(1), source_attention_weights
        )
        # Probability of generating (from output_probs).
        gen_probs = self.generation_probability(
            source_context, hidden, embedded
        ).unsqueeze(2)
        gen_scores = gen_probs * output_probs
        ptr_scores = (1 - gen_probs) * ptr_probs
        scores = gen_scores + ptr_scores
        # Puts scores in log space.
        scores = torch.log(scores)
        return scores, (h, c)