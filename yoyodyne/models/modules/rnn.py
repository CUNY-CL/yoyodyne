"""RNN module classes.

RNNModule is the base class.

We abstract away from the different formats used by GRUs and LSTMs; the latter
also tracks "cell state" and stores it as part of a tuple of tensors along
with the hidden state. RNNState hides this detail. For encoder modules,
GRUEncoderModule and LSTMEncoderModule wrap nn.GRU and nn.LSTM, respectively,
taking responsibility for packing and padding; GRUDecoderModule and
LSTMDecoderModule are similar wrappers for decoder modules.
"""

import abc
from typing import Optional, Tuple

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


class RNNState(nn.Module):
    """Represents the state of an RNN."""

    hidden: torch.Tensor
    cell: Optional[torch.Tensor]  # LSTMs only.

    def __init__(self, hidden, cell=None):
        super().__init__()
        self.register_buffer("hidden", hidden)
        self.register_buffer("cell", cell)


class RNNEncoderModule:
    """Patches RNN encoder modules to work with packing.

    The derived modules do not pass an initial hidden state (or cell state, in
    the case of GRUs, so it is effectively zero.
    """

    @staticmethod
    def _pad(sequence: nn.utils.rnn.PackedSequence) -> torch.Tensor:
        packed, _ = nn.utils.rnn.pad_packed_sequence(
            sequence,
            batch_first=True,
            padding_value=special.PAD_IDX,
        )
        return packed

    @staticmethod
    def _pack(
        sequence: torch.Tensor, lengths: torch.Tensor
    ) -> nn.utils.rnn.PackedSequence:
        return nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

    @abc.abstractmethod
    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, RNNState]: ...


class GRUEncoderModule(nn.GRU, RNNEncoderModule):
    """Patches GRU API to work with RNNState and packing."""

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, RNNState]:
        packed, hidden = super().forward(self._pack(sequence, lengths))
        return self._pad(packed), RNNState(hidden)


class LSTMEncoderModule(nn.LSTM, RNNEncoderModule):
    """Patches LSTM API to work with RNNState and packing."""

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, RNNState]:
        packed, (hidden, cell) = super().forward(self._pack(sequence, lengths))
        return self._pad(packed), RNNState(hidden, cell)


class RNNEncoder(RNNModule):
    """Abstract base class for RNN encoders."""

    def forward(self, source: data.PaddedTensor) -> torch.Tensor:
        """Encodes the source.

        Subsequent operations don't use the encoder's final hidden state for
        anything so it's discarded here.

        Args:
            source (data.PaddedTensor): source padded tensors of shape
                B x seq_len x 1.

        Returns:
            torch.Tensor.
        """
        encoded, _ = self.module(self.embed(source.padded), source.lengths())
        return encoded

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions


class GRUEncoder(RNNEncoder):
    """GRU encoder."""

    def get_module(self) -> nn.GRU:
        return GRUEncoderModule(
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
        return LSTMEncoderModule(
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


class GRUDecoderModule(nn.GRU):
    """Patches GRU API to work with RNNState."""

    def forward(
        self, symbol: torch.Tensor, state: RNNState
    ) -> Tuple[torch.Tensor, RNNState]:
        decoded, hidden = super().forward(symbol, state.hidden)
        return decoded, RNNState(hidden)


class LSTMDecoderModule(nn.LSTM):
    """Patches LSTM API to work with RNNState."""

    def forward(
        self, symbol: torch.Tensor, state: RNNState
    ) -> Tuple[torch.Tensor, RNNState]:
        assert state.cell is not None, "expected cell state"
        decoded, (hidden, cell) = super().forward(
            symbol, (state.hidden, state.cell)
        )
        return decoded, RNNState(hidden, cell)


class RNNDecoder(RNNModule):
    """Abstract base class for RNN decoders.

    This implementation is inattentive; it uses the last (non-padding) hidden
    state of the encoder as the input to the decoder.

    The initial decoder hidden state is a learned parameter.
    """

    h0: nn.Parameter

    def __init__(self, decoder_input_size, *args, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))

    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        symbol: torch.Tensor,
        state: RNNState,
    ) -> Tuple[torch.Tensor, RNNState]:
        """Single decode pass.

        Args:
            encoded (torch.Tensor): encoded source sequence of shape
                B x seq_len x encoder_dim.
            mask (torch.Tensor): mask of shape B x seq_len.
            symbol (torch.Tensor): previously decoded symbol(s) of shape B x 1.
            state (RNNState).

        Returns:
            Tuple[torch.Tensor, RNNState].
        """
        embedded = self.embed(symbol)
        last = self._last_encoded(encoded, mask)
        decoded, state = self.module(torch.cat((embedded, last), dim=2), state)
        decoded = self.dropout_layer(decoded)
        return decoded, state

    @staticmethod
    def _last_encoded(
        encoded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Gets the encoding at the first END for each tensor.

        Args:
            encoded (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            mask (torch.Tensor): mask of shape B x seq_len.

        Returns:
            torch.Tensor: indices of shape B x 1 x encoder_dim.
        """
        # Gets the index of the last unmasked tensor.
        # -> B.
        last_idx = (~mask).sum(dim=1) - 1
        # -> B x 1 x encoder_dim.
        last_idx = last_idx.view(encoded.size(0), 1, 1).expand(
            -1, -1, encoded.size(2)
        )
        # -> B x 1 x encoder_dim.
        return encoded.gather(1, last_idx)

    @abc.abstractmethod
    def initial_state(self, batch_size: int) -> RNNState: ...

    @property
    def output_size(self) -> int:
        return self.hidden_size


class GRUDecoder(RNNDecoder):
    """GRU decoder."""

    def get_module(self) -> nn.GRU:
        return GRUDecoderModule(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def initial_state(self, batch_size: int) -> RNNState:
        return RNNState(self.h0.repeat(self.layers, batch_size, 1))

    @property
    def name(self) -> str:
        return "GRU"


class LSTMDecoder(RNNDecoder):
    """LSTM decoder.

    This also has an initial cell state parameter.
    """

    c0: nn.Parameter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))

    def get_module(self) -> nn.LSTM:
        return LSTMDecoderModule(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def initial_state(self, batch_size: int) -> RNNState:
        return RNNState(
            self.h0.repeat(self.layers, batch_size, 1),
            self.c0.repeat(self.layers, batch_size, 1),
        )

    @property
    def name(self) -> str:
        return "LSTM"


class AttentiveRNNDecoder(RNNDecoder):
    """Abstract base class for attentive RNN decoders.

    The attention module differentially attends to different parts of the
    encoder output.
    """

    def __init__(self, attention_input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(
            attention_input_size, self.hidden_size
        )

    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        symbol: torch.Tensor,
        state: RNNState,
    ) -> Tuple[torch.Tensor, RNNState]:
        """Single decode pass.

        Args:
            encoded (torch.Tensor): encoded source sequence of shape
                B x seq_len x encoder_dim.
            mask (torch.Tensor): mask of shape B x seq_len.
            symbol (torch.Tensor): previously decoded symbol(s) of shape
                B x 1.
            state (RNNState): RNN state.

        Returns:
            Tuple[torch.Tensor, RNNState].
        """
        embedded = self.embed(symbol)
        context, _ = self.attention(
            encoded, state.hidden.transpose(0, 1), mask
        )
        decoded, state = self.module(
            torch.cat((embedded, context), dim=2), state
        )
        decoded = self.dropout_layer(decoded)
        return decoded, state


class AttentiveGRUDecoder(AttentiveRNNDecoder, GRUDecoder):
    """Attentive GRU decoder."""

    @property
    def name(self) -> str:
        return "attentive GRU"


class AttentiveLSTMDecoder(AttentiveRNNDecoder, LSTMDecoder):
    """Attentive LSTM decoder."""

    @property
    def name(self) -> str:
        return "attentive LSTM"
