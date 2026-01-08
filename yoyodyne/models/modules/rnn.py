"""RNN module classes.

RNNModule is the base class.

We abstract away from the different formats used by GRUs and LSTMs; the latter
also tracks "cell state" and stores it as part of a tuple of tensors along
with the hidden state. RNNState hides this detail. For encoder modules,
WrappedGRUEncoderWrapped and WrappedLSTMEncoder wrap nn.GRU and nn.LSTM,
respectively, taking responsibility for packing and padding; WrappedGRUDecoder
and WrappedLSTMDecoder are similar wrappers for decoder modules.
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
        *args: passed to superclass.
        hidden_size (int, optional): size of the hidden layer.
        layers (int, optional): number of layers.
        **kwargs: passed to superclass.
    """

    hidden_size: int
    layers: int

    def __init__(
        self,
        *args,
        hidden_size: int = defaults.HIDDEN_SIZE,
        layers: int = defaults.LAYERS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.layers = layers
        self.module = self.get_module()

    @abc.abstractmethod
    def get_module(self) -> nn.RNNBase: ...

    def embed(
        self, symbols: torch.Tensor, embeddings: nn.Embedding
    ) -> torch.Tensor:
        return self.dropout_layer(embeddings(symbols))


class RNNState(nn.Module):
    """Represents the state of an RNN."""

    hidden: torch.Tensor
    cell: Optional[torch.Tensor]  # LSTMs only.

    def __init__(self, hidden, cell=None):
        super().__init__()
        self.register_buffer("hidden", hidden)
        self.register_buffer("cell", cell)


class WrappedRNNEncoder:
    """Wraps RNN encoder modules to work with packing.

    The derived modules do not pass an initial hidden state (or cell state, in
    the case of LSTMs), so it is effectively zero.
    """

    @staticmethod
    def _pad(sequence: nn.utils.rnn.PackedSequence) -> torch.Tensor:
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            sequence,
            batch_first=True,
            padding_value=special.PAD_IDX,
        )
        return padded

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
    ) -> torch.Tensor: ...


class WrappedGRUEncoder(nn.GRU, WrappedRNNEncoder):
    """Wraps GRU API to work with packing."""

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed, _ = super().forward(self._pack(sequence, lengths))
        return self._pad(packed)


class WrappedLSTMEncoder(nn.LSTM, WrappedRNNEncoder):
    """Wraps LSTM API to work with packing."""

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        packed, _ = super().forward(self._pack(sequence, lengths))
        return self._pad(packed)


class RNNEncoder(RNNModule):
    """Abstract base class for RNN encoders.

    Args:
        *args: passed to superclass.
        bidirectional (bool, optional): should the RNN be bidirectional?
        **kwargs: passed to superclass.
    """

    def __init__(self, *args, bidirectional=defaults.BIDIRECTIONAL, **kwargs):
        self.bidirectional = bidirectional
        super().__init__(*args, **kwargs)

    def forward(
        self,
        symbols: data.PaddedTensor,
        embeddings: nn.Embedding,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encodes the symbols.

        Subsequent operations don't use the encoder's final hidden state for
        anything so it's discarded.

        Args:
            symbols (data.PaddedTensor).
            embeddings (nn.Embedding).
            *args: ignored.
            **kwargs: ignored.

        Returns:
            torch.Tensor.
        """
        return self.module(
            self.embed(symbols.padded, embeddings), symbols.lengths()
        )

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions


class GRUEncoder(RNNEncoder):
    """GRU encoder."""

    def get_module(self) -> WrappedGRUEncoder:
        return WrappedGRUEncoder(
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

    def get_module(self) -> WrappedLSTMEncoder:
        return WrappedLSTMEncoder(
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


class WrappedGRUDecoder(nn.GRU):
    """Wraps GRU API to work with RNNState."""

    def forward(
        self, symbol: torch.Tensor, state: RNNState
    ) -> Tuple[torch.Tensor, RNNState]:
        decoded, hidden = super().forward(symbol, state.hidden)
        return decoded, RNNState(hidden)


class WrappedLSTMDecoder(nn.LSTM):
    """Wraps LSTM API to work with RNNState."""

    def forward(
        self, symbol: torch.Tensor, state: RNNState
    ) -> Tuple[torch.Tensor, RNNState]:
        assert state.cell is not None, "expected cell state"
        decoded, (hidden, cell) = super().forward(
            symbol, (state.hidden, state.cell)
        )
        return decoded, RNNState(hidden, cell)


class RNNDecoder(RNNModule):
    """Base class for RNN decoders.

    This implementation lacks a learned attention mechanism; rather, it uses
    the encodings of the sequence-final hidden states as the input to the
    decoder.

    The initial decoder hidden state is a learned parameter.

    Args:
        *args: passed to superclass.
        decoder_input_size (int, optional): decoder input size.
        **kwargs: passed to superclass.
    """

    decoder_input_size: int
    h0: nn.Parameter

    def __init__(
        self, decoder_input_size=defaults.HIDDEN_SIZE * 2, *args, **kwargs
    ):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))

    def forward(
        self,
        symbol: torch.Tensor,
        embeddings: nn.Embedding,
        context: torch.Tensor,
        mask: torch.Tensor,
        state: RNNState,
    ) -> Tuple[torch.Tensor, RNNState]:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol(s) of shape B x 1.
            embeddings (nn.Embedding).
            context (torch.Tensor).
            mask (torch.Tensor): ignored.
            state (RNNState).

        Returns:
            Tuple[torch.Tensor, RNNState].
        """
        embedded = self.embed(symbol, embeddings)
        decoded, state = self.module(
            torch.cat((embedded, context), dim=2), state
        )
        return self.dropout_layer(decoded), state

    @staticmethod
    def get_context(
        sequence: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the source context vector.

        Here we use the sequence-final encodings.

        Args:
            sequence (torch.Tensor).
            encoded (torch.Tensor).
            state (RNNState): ignored.

        Returns:
            torch.Tensor.
        """
        mask = torch.unsqueeze(sequence == special.END_IDX, dim=2)
        return torch.mean(encoded * mask, dim=1, keepdim=True)

    @abc.abstractmethod
    def initial_state(self, batch_size: int) -> RNNState: ...

    @property
    def output_size(self) -> int:
        return self.hidden_size


class GRUDecoder(RNNDecoder):
    """GRU decoder."""

    def get_module(self) -> WrappedGRUDecoder:
        return WrappedGRUDecoder(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout,
            num_layers=self.layers,
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

    def get_module(self) -> WrappedLSTMDecoder:
        return WrappedLSTMDecoder(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    def initial_state(self, batch_size: int) -> RNNState:
        return RNNState(
            self.h0.repeat(self.layers, batch_size, 1),
            self.c0.repeat(self.layers, batch_size, 1),
        )

    @property
    def name(self) -> str:
        return "LSTM"


class SoftAttentionRNNDecoder(RNNDecoder):
    """Abstract base class for soft attention RNN decoders.

    The attention module learns to differentially attend to different symbols
    in the encoder output.

    Args:
        *args: passed to superclass.
        attention_input_size (int, optional).
        **kwargs: passed to superclass.
    """

    def __init__(
        self,
        attention_input_size: int = defaults.HIDDEN_SIZE * 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(
            attention_input_size, self.hidden_size
        )

    def forward(
        self,
        symbol: torch.Tensor,
        embeddings: nn.Embedding,
        context: torch.Tensor,
        mask: torch.Tensor,
        state: RNNState,
    ) -> Tuple[torch.Tensor, RNNState]:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol(s) of shape B x 1.
            embeddings (nn.Embedding).
            context (torch.Tensor).
            mask (torch.Tensor).
            state (RNNState).

        Returns:
            Tuple[torch.Tensor, RNNState].
        """
        embedded = self.embed(symbol, embeddings)
        # Here we weigh the context using attention.
        context, _ = self.attention(
            context, state.hidden.transpose(0, 1), mask
        )
        decoded, state = self.module(
            torch.cat((embedded, context), dim=2), state
        )
        return self.dropout_layer(decoded), state

    @staticmethod
    def get_context(
        sequence: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the source context vector.

        Here the entire encoding is the context.

        Args:
            sequence (torch.Tensor): ignored.
            encoded (torch.Tensor).

        Returns:
            torch.Tensor.
        """
        return encoded


class SoftAttentionGRUDecoder(SoftAttentionRNNDecoder, GRUDecoder):
    """Soft attention GRU decoder."""

    @property
    def name(self) -> str:
        return "soft attention GRU"


class SoftAttentionLSTMDecoder(SoftAttentionRNNDecoder, LSTMDecoder):
    """Soft attention LSTM decoder."""

    @property
    def name(self) -> str:
        return "soft attention LSTM"
