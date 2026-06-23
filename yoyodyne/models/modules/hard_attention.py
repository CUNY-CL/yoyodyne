"""Hard attention module classes.

Hard attention models use unadulterated RNN encoders, but extend RNN decoders.
There are separate decoders for zeroth-order (HardAttentionRNNDecoder) and
first-order (ContextHardAttentionRNNDecoder) decoders."""

import torch
from torch import nn

from ... import defaults
from . import position, rnn, transformer, transformer_layers


class HardAttentionRNNDecoder(rnn.RNNDecoder):
    """Abstract base class for zeroth-order HMM hard attention RNN decoders.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    output_proj: nn.Sequential
    scale_encoded: nn.Linear

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Activates emission probs.
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_size, self.output_size), nn.Tanh()
        )
        # Projects transition probabilities to depth of module.
        self.scale_encoded = nn.Linear(
            self.decoder_input_size, self.hidden_size
        )

    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        symbol: torch.Tensor,
        state: rnn.RNNState,
        embeddings: nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor, rnn.RNNState]:
        """Single decode pass.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).
            symbol (torch.Tensor): previously decoded symbol(s) of shape B x 1.
            state (RNNState).
            embeddings (nn.Embedding).

        Returns:
            tuple[torch.Tensor, torch.Tensor, RNNState]: the emission and
                transition tensors and RNN state.
        """
        embedded = self.embed(symbol, embeddings)
        decoded, state = self.module(embedded, state)
        emissions = self._get_emissions(decoded, encoded)
        transitions = self._get_transitions(decoded, encoded, mask)
        return emissions, transitions, state

    def _get_emissions(
        self, decoded: torch.Tensor, encoded: torch.Tensor
    ) -> torch.Tensor:
        """Gets emission probabilities for current timestep.

        Args:
            decoded (torch.Tensor).
                shape B x 1 x decoder_dim.
            encoded (torch.Tensor).

        Returns:
            torch.Tensor: emissions.
        """
        output = decoded.expand(-1, encoded.size(1), -1)
        output = torch.cat((output, encoded), dim=2)
        output = self.output_proj(output)
        return output

    def _get_transitions(
        self,
        decoded: torch.Tensor,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gets transition probabilities for current timestep.

        Given the current encoder representation and the decoder
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
            decoded (torch.Tensor): output from decoder for current timestep
                of shape B x 1 x decoder_dim.
            encoded (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            torch.Tensor: alignment scores across the source sequence of shape
                B x seq_len.
        """
        alignment_scores = torch.bmm(
            self.scale_encoded(encoded), decoded.transpose(1, 2)
        ).squeeze(2)
        # Gets probability of alignments.
        alignment_probs = nn.functional.softmax(alignment_scores, dim=1)
        # Masks padding.
        alignment_probs = alignment_probs * (~mask) + defaults.EPSILON
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=1, keepdim=True
        )
        # Expands over all time steps; uses log probabilities for quicker
        # computations.
        return (
            alignment_probs.log().unsqueeze(1).expand(-1, encoded.size(1), -1)
        )

    @property
    def output_size(self) -> int:
        return self.decoder_input_size + self.hidden_size


class HardAttentionGRUDecoder(HardAttentionRNNDecoder, rnn.GRUDecoder):
    """Zeroth-order HMM hard attention GRU decoder."""

    def get_module(self) -> rnn.WrappedGRUDecoder:
        return rnn.WrappedGRUDecoder(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "hard attention GRU"


class HardAttentionLSTMDecoder(HardAttentionRNNDecoder, rnn.LSTMDecoder):
    """Zeroth-order HMM hard attention LSTM decoder."""

    def get_module(self) -> rnn.WrappedLSTMDecoder:
        return rnn.WrappedLSTMDecoder(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "hard attention LSTM"


class ContextHardAttentionRNNDecoder(HardAttentionRNNDecoder):
    """Abstract base class for first-order HMM hard attention RNN decoder.

    This overrides the definition of _get_transitions.
    """

    def __init__(
        self,
        attention_context: int = defaults.ATTENTION_CONTEXT,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.delta = attention_context
        # The window size must include center and both sides.
        self.alignment_proj = nn.Linear(
            self.hidden_size * 2, (self.delta * 2) + 1
        )

    def _get_transitions(
        self,
        decoded: torch.Tensor,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gets transition probabilities for current timestep.

        Args:
            decoded (torch.Tensor): output from decoder for current timesstep
                of shape B x 1 x decoder_dim.
            encoded (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            torch.Tensor: alignment scores across the source sequence of shape
                B x seq_len.
        """
        # Matrix multiplies encoding and decoding for alignment
        # representations. See: https://aclanthology.org/P19-1148/.
        # Expands decoded so it can concatenate with alignments.
        decoded = decoded.expand(-1, encoded.size(1), -1)
        # -> B x seq_len.
        alignment_scores = torch.cat(
            (self.scale_encoded(encoded), decoded), dim=2
        )
        alignment_scores = self.alignment_proj(alignment_scores)
        alignment_probs = nn.functional.softmax(alignment_scores, dim=1)
        # Limits context to a window of the size delta (i.e., the context
        # length).
        alignment_probs = alignment_probs.split(1, dim=1)
        alignment_probs = torch.cat(
            [
                nn.functional.pad(
                    t,
                    (
                        -self.delta + i,
                        mask.size(1) - (self.delta + 1) - i,
                    ),
                )
                for i, t in enumerate(alignment_probs)
            ],
            dim=1,
        )
        # Gets probability of alignments, masking padding.
        alignment_probs = (
            alignment_probs * (~mask).unsqueeze(1) + defaults.EPSILON
        )
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=2, keepdim=True
        )
        # Uses log probabilities for quicker computations.
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


class HardAttentionTransformerDecoder(transformer.TransformerModule):
    """Zeroth-order HMM hard attention transformer decoder.

    Replaces the RNN cell in the hard attention model with a causally-masked
    self-attention-only stack. Cross-attention to the source is intentionally
    absent: source context enters through the emission concatenation and the
    transition dot-product, so the HMM DP and the neural network do not
    double-count source-position information.

    The forward interface, output_size property, and initial_state method are
    compatible with HardAttentionRNNDecoder, so HardAttentionRNNModel drives
    both backends without modification. The "state" passed through decode_step
    is the accumulated target prefix (a B x t long tensor) rather than an RNN
    hidden-state tuple; it starts as an empty B x 0 tensor from initial_state
    and grows by one column per step.

    Emission and transition computations mirror HardAttentionRNNDecoder.
    Because the transformer produces a B x 1 x embedding_size hidden state at
    the last position (parallel to the RNN's B x 1 x hidden_size decoded),
    _get_emissions and _get_transitions are structurally identical to the RNN
    versions, with embedding_size in place of hidden_size.

    Args:
        *args: passed to TransformerModule.
        decoder_input_size (int, optional): dimensionality of the source
            encoder output.
        **kwargs: passed to TransformerModule.
    """

    decoder_input_size: int
    output_proj: nn.Sequential
    scale_encoded: nn.Linear

    def __init__(
        self,
        *args,
        decoder_input_size: int = defaults.EMBEDDING_SIZE,
        **kwargs,
    ):
        # Must be set before super().__init__ so output_size is available
        # when output_proj and scale_encoded are built below.
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_size, self.output_size), nn.Tanh()
        )
        self.scale_encoded = nn.Linear(
            self.decoder_input_size, self.embedding_size
        )

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
            enable_nested_tensor=False,
        )

    def _get_emissions(
        self, decoded: torch.Tensor, encoded: torch.Tensor
    ) -> torch.Tensor:
        """Gets emission probabilities for current timestep.

        Args:
            decoded (torch.Tensor): B x 1 x embedding_size.
            encoded (torch.Tensor).

        Returns:
            torch.Tensor: emissions.
        """
        output = decoded.expand(-1, encoded.size(1), -1)
        output = torch.cat((output, encoded), dim=2)
        output = self.output_proj(output)
        return output

    def _get_transitions(
        self,
        decoded: torch.Tensor,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gets transition probabilities for current timestep.

        Args:
            decoded (torch.Tensor): B x 1 x embedding_size.
            encoded (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            torch.Tensor: B x src_len x src_len log-probs.
        """
        alignment_scores = torch.bmm(
            self.scale_encoded(encoded), decoded.transpose(1, 2)
        ).squeeze(2)
        # Gets probability of alignments.
        alignment_probs = nn.functional.softmax(alignment_scores, dim=1)
        # Masks padding.
        alignment_probs = alignment_probs * (~mask) + defaults.EPSILON
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=1, keepdim=True
        )
        # Expands over all time steps; uses log probabilities for quicker
        # computations.
        return (
            alignment_probs.log().unsqueeze(1).expand(-1, encoded.size(1), -1)
        )

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """Returns the initial (empty) prefix state.

        The transformer carries no recurrent hidden state; the accumulated
        target-symbol prefix is the analogue of the RNN state tuple.

        Args:
            batch_size (int).

        Returns:
            torch.Tensor: B x 0 long tensor on this module's device.
        """
        return torch.zeros(batch_size, 0, dtype=torch.long, device=self.device)

    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        symbol: torch.Tensor,
        state: torch.Tensor,
        embeddings: nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decode pass.

        Appends symbol to the accumulated prefix, runs the transformer, and
        returns emission vectors and transition log-probs for the new position,
        plus the updated prefix.

        Args:
            encoded (torch.Tensor): B x src_len x decoder_input_size.
            mask (torch.Tensor): B x src_len padding mask (True = pad).
            symbol (torch.Tensor): B x 1 current input token indices.
            state (torch.Tensor): B x t accumulated prefix.
            embeddings (nn.Embedding).

        Returns:
            emissions, transitions, and the new state.
        """
        state = torch.cat((state, symbol), dim=1)
        embedded = self.embed(state, embeddings)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            embedded.size(1), device=self.device
        )
        hidden = self.module(embedded, mask=causal_mask, is_causal=True)
        decoded = hidden[:, -1:, :]
        return (
            self._get_emissions(decoded, encoded),
            self._get_transitions(decoded, encoded, mask),
            state,
        )

    @property
    def output_size(self) -> int:
        return self.embedding_size + self.decoder_input_size

    @property
    def name(self) -> str:
        return "hard attention transformer"


class ContextHardAttentionTransformerDecoder(HardAttentionTransformerDecoder):
    """First-order HMM hard attention transformer decoder.

    Overrides _get_transitions with the windowed context mechanism from
    ContextHardAttentionRNNDecoder. For each source position i the model
    predicts a distribution over the window [i-delta, i+delta], giving a
    sparse B x src_len x src_len matrix rather than a uniform broadcast.
    alignment_proj uses embedding_size * 2 in place of the RNN's hidden_size
    * 2; the body of _get_transitions is otherwise identical to the RNN
    version.

    Args:
        *args: passed to HardAttentionTransformerDecoder.
        attention_context (int): half-width of the transition window (delta).
        **kwargs: passed to HardAttentionTransformerDecoder.
    """

    alignment_proj: nn.Linear

    def __init__(
        self,
        *args,
        attention_context: int = defaults.ATTENTION_CONTEXT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.delta = attention_context
        # The window size must include center and both sides.
        self.alignment_proj = nn.Linear(
            self.embedding_size * 2, (self.delta * 2) + 1
        )

    def _get_transitions(
        self,
        decoded: torch.Tensor,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gets transition probabilities for current timestep.

        Args:
            decoded (torch.Tensor): B x 1 x embedding_size.
            encoded (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            torch.Tensor: B x src_len x src_len log-probs.
        """
        decoded = decoded.expand(-1, encoded.size(1), -1)
        alignment_scores = torch.cat(
            (self.scale_encoded(encoded), decoded), dim=2
        )
        alignment_scores = self.alignment_proj(alignment_scores)
        alignment_probs = nn.functional.softmax(alignment_scores, dim=1)
        # Limits context to a window of the size delta (i.e., the context
        # length).
        alignment_probs = alignment_probs.split(1, dim=1)
        alignment_probs = torch.cat(
            [
                nn.functional.pad(
                    t,
                    (
                        -self.delta + i,
                        mask.size(1) - (self.delta + 1) - i,
                    ),
                )
                for i, t in enumerate(alignment_probs)
            ],
            dim=1,
        )
        # Gets probability of alignments, masking padding.
        alignment_probs = (
            alignment_probs * (~mask).unsqueeze(1) + defaults.EPSILON
        )
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=2, keepdim=True
        )
        # Uses log probabilities for quicker computations.
        return alignment_probs.log()

    @property
    def name(self) -> str:
        return "contextual hard attention transformer"


class RotaryHardAttentionTransformerDecoder(
    transformer.RotaryTransformerModule,
    HardAttentionTransformerDecoder,
):
    """HardAttentionTransformerDecoder with rotary positional encodings.

    RotaryTransformerModule must be first in the MRO so its set_max_length
    and _set_rope override the sinusoidal defaults in TransformerModule.

    Args:
        *args: passed to HardAttentionTransformerDecoder.
        **kwargs: passed to HardAttentionTransformerDecoder.
    """

    def get_module(self) -> nn.TransformerEncoder:
        rope = position.RotaryPositionalEncoding(
            embedding_size=self._rope_head_dim,
            max_length=self._rope_max_length,
        )
        encoder_layer = transformer_layers.RotaryTransformerEncoderLayer(
            rope,
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
            enable_nested_tensor=False,
        )

    @property
    def name(self) -> str:
        return "rotary hard attention transformer"


class RotaryContextHardAttentionTransformerDecoder(
    transformer.RotaryTransformerModule,
    ContextHardAttentionTransformerDecoder,
):
    """ContextHardAttentionTransformerDecoder with rotary positional encodings.

    RotaryTransformerModule must be first in the MRO so its set_max_length
    and _set_rope override the sinusoidal defaults in TransformerModule.

    Args:
        *args: passed to ContextHardAttentionTransformerDecoder.
        **kwargs: passed to ContextHardAttentionTransformerDecoder.
    """

    def get_module(self) -> nn.TransformerEncoder:
        rope = position.RotaryPositionalEncoding(
            embedding_size=self._rope_head_dim,
            max_length=self._rope_max_length,
        )
        encoder_layer = transformer_layers.RotaryTransformerEncoderLayer(
            rope,
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
            enable_nested_tensor=False,
        )

    @property
    def name(self) -> str:
        return "rotary contextual hard attention transformer"
