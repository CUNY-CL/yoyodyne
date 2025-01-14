"""RNN module classes."""

import abc
from typing import Tuple, Union

import torch
from torch import nn

from ... import data, defaults, special
from . import attention, base


class RNNModule(base.BaseModule):
    """Abstract base class for RNN modules.

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

    @abc.abstractmethod
    def get_module(self) -> nn.RNNBase: ...

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1


class RNNEncoder(RNNModule):
    """Abstract base class for RNN encoders."""

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
        # -> B x seq_len x target_vocab_size, hiddens
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
    """Abstract base class for RNN decoders.

    This implementation is inattentive; it uses the last (non-padding) hidden
    state of the encoder as the input to the decoder.
    """

    def __init__(self, decoder_input_size, *args, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> base.ModuleOutput:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (Union[torch.Tensor, Tuple[torch.Tensor,
                torch.Tensor]]): last hidden states from the decoder of shape
                1 x B x decoder_dim.
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            base.ModuleOutput: decoder output, and the previous hidden states
                from the decoder RNN.
        """
        embedded = self.embed(symbol)
        last_encoder_out = self._last_encoder_out(encoder_out, encoder_mask)
        decoder_input = torch.cat((embedded, last_encoder_out), dim=2)
        output, hiddens = self.module(decoder_input, last_hiddens)
        return base.ModuleOutput(self.dropout_layer(output), hiddens)

    @staticmethod
    def _last_encoder_out(
        encoder_out: torch.Tensor, encoder_mask: torch.Tensor
    ) -> torch.Tensor:
        """Gets the encoding at the first END for each tensor.

        Args:
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            torch.Tensor: indices of shape B x 1 x encoder_dim.
        """
        # Gets the index of the last unmasked tensor.
        # -> B.
        last_idx = (~encoder_mask).sum(dim=1) - 1
        # -> B x 1 x encoder_dim.
        last_idx = last_idx.view(encoder_out.size(0), 1, 1).expand(
            -1, -1, encoder_out.size(2)
        )
        # -> B x 1 x encoder_dim.
        return encoder_out.gather(1, last_idx)

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
    """Abstract base class for attentive RNN decoders.

    Subsequent concrete implementations use the attention module to
    differentially attend to different parts of the encoder output.
    """

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
        decoder_input = torch.cat((embedded, context), dim=2)
        output, hiddens = self.module(decoder_input, last_hiddens)
        return base.ModuleOutput(self.dropout_layer(output), hiddens)

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
        embedded = self.embed(symbol)
        # Cell state isn't needed for the forward pass.
        last_h0, _ = last_hiddens
        context, _ = self.attention(
            last_h0.transpose(0, 1), encoder_out, encoder_mask
        )
        decoder_input = torch.cat((embedded, context), dim=2)
        output, hiddens = self.module(decoder_input, last_hiddens)
        return base.ModuleOutput(self.dropout_layer(output), hiddens)

    @property
    def name(self) -> str:
        return "attentive LSTM"
