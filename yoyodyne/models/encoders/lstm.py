"""LSTM model classes."""

import argparse
import heapq
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from . import attention, base_encoder
from ... import batches, defaults


class LSTMEncoder(base_encoder.BaseEncoder):
    """LSTM encoder-decoder without attention.

    We achieve this by concatenating the last (non-padding) hidden state of
    the encoder to the decoder hidden state."""

    # Model arguments.
    bidirectional: bool
    # Constructed inside __init__.
    embeddings: nn.Embedding
    module: nn.LSTM

    def __init__(
        self,
        *args,
        bidirectional=defaults.BIDIRECTIONAL,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            *args: passed to superclass.
            bidirectional (bool).
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.embeddings = self.init_embeddings(
            self.num_embeddings, self.embedding_size, self.pad_idx
        )
        self.module = self.load_module()

    def load_module(self):
        return nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    @property
    def output_size(self):
        return self.hidden_size * self.num_directions

    @property
    def num_directions(self):
        return 2 if self.bidirectional else 1

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
        return self._normal_embedding_initialization(
            num_embeddings, embedding_size, pad_idx
        )

    def embed(self, symbols: torch.Tensor):
        embedded = self.embeddings(symbols)
        return self.dropout_layer(embedded)

    def forward(
        self, source: batches.PaddedTensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes the input.

        Args:
            source (batches.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                encoded timesteps, and the LSTM h0 and c0 cells.
        """
        embedded = self.embed(source.padded)
        # Packs embedded source symbols into a PackedSequence.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source.lengths(), batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim, (h0, c0).
        packed_outs, (H, C) = self.module(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        return encoded, (H, C)


class LSTMDecoder(LSTMEncoder):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            *args: passed to superclass.
            bidirectional (bool).
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded = self.embed(symbol)
        # -> 1 x B x decoder_dim.
        last_h0, last_c0 = last_hiddens
        # Get the index of the last unmasked tensor.
        # -> B.
        last_encoder_out_idxs = (~encoder_mask).sum(dim=1) - 1
        # -> B x 1 x 1.
        last_encoder_out_idxs = last_encoder_out_idxs.view(
            [encoder_out.size(0)] + [1, 1]
        )
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_encoder_out_idxs = last_encoder_out_idxs.expand(
            [-1, -1, encoder_out.size(-1)]
        )
        # -> B x 1 x encoder_dim.
        last_encoder_out = torch.gather(encoder_out, 1, last_encoder_out_idxs)
        # The input to decoder LSTM is the embedding concatenated to the
        # weighted, encoded, inputs.
        output, hiddens = self.module(
            torch.cat((embedded, last_encoder_out), 2), (last_h0, last_c0)
        )
        output = self.dropout_layer(output)
        return output, hiddens

    def load_module(self):
        return nn.LSTM(
            self.decoder_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

class LSTMAttentiveDecoder(LSTMDecoder):
    def __init__(self, *args, **kwargs):
        """Initializes the encoder-decoder with attention."""
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(self.decoder_size, self.hidden_size)

    def load_module(self):
        return nn.LSTM(
            self.decoder_size+self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes one step.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): last hidden
                states from the decoder of shape
                (1 x B x decoder_dim, 1 x B x decoder_dim).
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: softmax scores over all outputs,
                and the previous hidden states from the decoder LSTM.
        """
        embedded = self.embed(symbol)
        # -> 1 x B x decoder_dim.
        last_h0, last_c0 = last_hiddens
        context, _ = self.attention(
            last_h0.transpose(0, 1), encoder_out, encoder_mask
        )
        output, hiddens = self.module(
            torch.cat((embedded, context), 2), (last_h0, last_c0)
        )
        output = self.dropout_layer(output)
        # Classifies into output vocab.
        # -> B x 1 x output_size
        return output, hiddens