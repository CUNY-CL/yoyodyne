"""RNN module classes."""

from typing import Tuple

import torch
from torch import nn

from ... import data, defaults, special
from . import attention, base


class RNNModule(base.BaseModule):
    """Base class for RNN modules.

    Args:
        bidirectional (bool).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    bidirectional: bool

    def __init__(
        self,
        *args,
        bidirectional=defaults.BIDIRECTIONAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.module = self.get_module()

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    def get_module(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class RNNEncoder(RNNModule):
    """Base class for RNN encoders."""

    def forward(self, source: data.PaddedTensor) -> base.ModuleOutput:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            base.ModuleOutput.
        """
        embedded = self.embed(source.padded)
        # Packs embedded source symbols into a PackedSequence.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source.lengths(),
            batch_first=True,
            enforce_sorted=False,
        )
        # -> B x seq_len x encoder_dim, hiddens
        packed_outs, hiddens = self.module(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=special.PAD_IDX,
            total_length=None,
        )
        return base.ModuleOutput(encoded, hiddens)

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions


class GRUEncoder(RNNEncoder):
    """GRU encoder."""

    def get_module(self) -> nn.GRU:
        return nn.GRU(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "GRU"


class LSTMEncoder(RNNEncoder):
    """LSTM encoder."""

    def get_module(self) -> nn.LSTM:
        return nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "LSTM"


class RNNDecoder(RNNModule):
    """Base class for RNN decoders."""

    def __init__(self, decoder_input_size, *args, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> base.ModuleOutput:
        """Single decode pass.

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
            base.ModuleOutput: decoder output, and the previous hidden states
                from the decoder RNN.
        """
        embedded = self.embed(symbol)
        # -> 1 x B x decoder_dim.
        # Get the index of the last unmasked tensor.
        # -> B.
        last_encoder_out_idxs = (~encoder_mask).sum(dim=1) - 1
        # -> B x 1 x 1.
        last_encoder_out_idxs = last_encoder_out_idxs.view(
            encoder_out.size(0), 1, 1
        )
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_encoder_out_idxs = last_encoder_out_idxs.expand(
            -1, -1, encoder_out.size(-1)
        )
        # -> B x 1 x encoder_dim.
        last_encoder_out = torch.gather(encoder_out, 1, last_encoder_out_idxs)
        # The input to decoder RNN is the embedding concatenated to the
        # weighted, encoded, inputs.
        output, hiddens = self.module(
            torch.cat((embedded, last_encoder_out), 2), last_hiddens
        )
        output = self.dropout_layer(output)
        return base.ModuleOutput(output, hiddens)

    @property
    def output_size(self) -> int:
        return self.hidden_size


class GRUDecoder(RNNDecoder):
    """GRU decoder."""

    def get_module(self) -> nn.GRU:
        return nn.GRU(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    @property
    def name(self) -> str:
        return "GRU"


class LSTMDecoder(RNNDecoder):
    """LSTM decoder."""

    def get_module(self) -> nn.LSTM:
        return nn.LSTM(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    @property
    def name(self) -> str:
        return "LSTM"


class AttentiveRNNDecoder(RNNDecoder):
    """Base class for attentive RNN decoders."""

    def __init__(self, attention_input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(
            attention_input_size, self.hidden_size
        )


class AttentiveGRUDecoder(AttentiveRNNDecoder, GRUDecoder):
    """Attentive GRU decoder."""

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> base.ModuleOutput:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (torch.Tensor): last hidden states from the decoder
                of shape 1 x B x decoder_dim.
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            base.ModuleOutput: decoder output, and the previous hidden states
                from the decoder RNN.
        """
        embedded = self.embed(symbol)
        context, _ = self.attention(
            last_hiddens.transpose(0, 1), encoder_out, encoder_mask
        )
        output, hiddens = self.module(
            torch.cat((embedded, context), 2), last_hiddens
        )
        output = self.dropout_layer(output)
        return base.ModuleOutput(output, hiddens)

    @property
    def name(self) -> str:
        return "attentive GRU"


class AttentiveLSTMDecoder(AttentiveRNNDecoder, LSTMDecoder):
    """Attentive LSTM decoder."""

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> base.ModuleOutput:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): last hidden
                and cell state from the decoder of shape 1 x B x decoder_dim.
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            base.ModuleOutput: decoder output, and the previous hidden states
                from the decoder RNN.
        """
        # The last hiddens includes the cell state, which isn't needed for the
        # forward pass.
        embedded = self.embed(symbol)
        last_h0, last_c0 = last_hiddens
        context, _ = self.attention(
            last_h0.transpose(0, 1), encoder_out, encoder_mask
        )
        output, hiddens = self.module(
            torch.cat((embedded, context), 2), last_hiddens
        )
        output = self.dropout_layer(output)
        return base.ModuleOutput(output, hiddens)

    @property
    def name(self) -> str:
        return "attentive LSTM"


class HardAttentionRNNDecoder(RNNDecoder):
    """Base module for zeroth-order HMM hard attention RNN decoders."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Activates emission probs.
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_size, self.output_size), nn.Tanh()
        )
        # Projects transition probabilities to depth of module.
        self.scale_encoded = nn.Linear(
            self.decoder_input_size, self.hidden_size
        )

    def _alignment_step(
        self,
        decoded: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Creates alignment matrix for current timestep.

        Given the current encoder repreesentation and the decoder
        representation at the current time step, this calculates the alignment
        scores between all potential source sequence pairings. These
        alignments are used to predict the likelihood of state transitions
        for the output.

        After:
            Wu, S. and Cotterell, R. 2019. Exact hard monotonic attention for
            character-level transduction. In _Proceedings of the 57th Annual
            Meeting of the Association for Computational Linguistics_, pages
            1530-1537.

        Args:
            decoded (torch.Tensor): output from decoder for current timesstep
                of shape B x 1 x decoder_dim.
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            torch.Tensor: alignment scores across the source sequence of shape
                B x seq_len.
        """
        alignment_scores = torch.bmm(
            self.scale_encoded(encoder_out), decoded.transpose(1, 2)
        ).squeeze(-1)
        # Gets probability of alignments.
        alignment_probs = nn.functional.softmax(alignment_scores, dim=-1)
        # Mask padding.
        alignment_probs = alignment_probs * (~encoder_mask) + 1e-7
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=-1, keepdim=True
        )
        # Expands over all time steps. Log probs for quicker computation.
        return (
            alignment_probs.log()
            .unsqueeze(1)
            .expand(-1, encoder_out.shape[1], -1)
        )

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> base.ModuleOutput:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape (B x 1).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): last hidden
                states from the decoder of shape
                (1 x B x decoder_dim, 1 x B x decoder_dim).
            encoder_out (torch.Tensor): encoded input sequence of shape
                (B x seq_len x encoder_dim).
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape (B x seq_len).

        Returns:
            base.ModuleOutput: step-wise emission probabilities, alignment
                matrix, and hidden states of decoder.
        """
        # Encodes current symbol.
        embedded = self.embed(symbol)
        decoded, hiddens = self.module(embedded, last_hiddens)
        # Gets emission probabilities over each hidden state (source symbol).
        output = decoded.expand(-1, encoder_out.shape[1], -1)
        output = torch.cat([output, encoder_out], dim=-1)
        output = self.output_proj(output)
        # Gets transition probabilities (alignment) for current states.
        alignment = self._alignment_step(decoded, encoder_out, encoder_mask)
        return base.ModuleOutput(output, hiddens, embeddings=alignment)

    @property
    def output_size(self) -> int:
        return self.decoder_input_size + self.hidden_size


class HardAttentionGRUDecoder(HardAttentionRNNDecoder):
    """Zeroth-order HMM hard attention GRU decoder."""

    def get_module(self) -> nn.GRU:
        return nn.GRU(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "hard attention GRU"


class HardAttentionLSTMDecoder(HardAttentionRNNDecoder):
    """Zeroth-order HMM hard attention LSTM decoder."""

    def get_module(self) -> nn.LSTM:
        return nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "hard attention LSTM"


class ContextHardAttentionRNNDecoder(HardAttentionRNNDecoder):
    """Base module for first-order HMM hard attention RNN decoder."""

    def __init__(self, attention_context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = attention_context
        # Window size must include center and both sides.
        self.alignment_proj = nn.Linear(
            self.hidden_size * 2, (attention_context * 2) + 1
        )

    def _alignment_step(
        self,
        decoded: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Matrix multiplies encoding and decoding for alignment
        # representations. See: https://aclanthology.org/P19-1148/.
        # Expands decoded to concatenate with alignments
        decoded = decoded.expand(-1, encoder_out.shape[1], -1)
        # -> B x seq_len.
        alignment_scores = torch.cat(
            [self.scale_encoded(encoder_out), decoded], dim=2
        )
        alignment_scores = self.alignment_proj(alignment_scores)
        alignment_probs = nn.functional.softmax(alignment_scores, dim=-1)
        # Limits context to window of self.delta (context length).
        alignment_probs = alignment_probs.split(1, dim=1)
        alignment_probs = torch.cat(
            [
                nn.functional.pad(
                    t,
                    (
                        -self.delta + i,
                        encoder_mask.shape[1] - (self.delta + 1) - i,
                    ),
                )
                for i, t in enumerate(alignment_probs)
            ],
            dim=1,
        )
        # Gets probability of alignments, masking padding.
        alignment_probs = (
            alignment_probs * (~encoder_mask).unsqueeze(1) + defaults.EPSILON
        )
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=-1, keepdim=True
        )
        # Log probs for quicker computation.
        return alignment_probs.log()


class ContextHardAttentionGRUDecoder(
    ContextHardAttentionRNNDecoder, HardAttentionGRUDecoder
):
    """First-order HMM hard attention GRU decoder."""

    @property
    def name(self) -> str:
        return "contextual hard attention GRU"


class ContextHardAttentionLSTMDecoder(
    ContextHardAttentionRNNDecoder, HardAttentionLSTMDecoder
):
    """First-order HMM hard attention LSTM decoder."""

    @property
    def name(self) -> str:
        return "contextual hard attention LSTM"
