"""Pointer-generator RNN model classes."""

import torch
from torch import nn

from ... import data, special
from .. import beam_search, defaults, embeddings, modules
from . import base


def _split_state(
    state: modules.RNNState, batch_size: int
) -> list[modules.RNNState]:
    """Splits a batched RNN state into a list of per-item states.

    Args:
        state (modules.RNNState): shape layers x B x hidden_size, or None.
        batch_size (int).

    Returns:
        list[modules.RNNState]: B states each of shape
            layers x 1 x hidden_size.
    """
    if state is None:
        return [None] * batch_size
    if isinstance(state, tuple):
        # LSTM: (h, c) each of shape layers x B x hidden_size.
        h, c = state
        return [
            (h[:, i : i + 1, :], c[:, i : i + 1, :]) for i in range(batch_size)
        ]
    # GRU: shape layers x B x hidden_size.
    return [state[:, i : i + 1, :] for i in range(batch_size)]


def _batch_states(states: list[modules.RNNState]) -> modules.RNNState:
    """Concatenates a list of per-item states into a single batched state.

    Args:
        states (list[modules.RNNState]): B states each of shape
            layers x 1 x hidden_size.

    Returns:
        modules.RNNState: shape layers x B x hidden_size.
    """
    if states[0] is None:
        return None
    if isinstance(states[0], tuple):
        h = torch.cat([s[0] for s in states], dim=1)
        c = torch.cat([s[1] for s in states], dim=1)
        return (h, c)
    return torch.cat(states, dim=1)


def _split_output_states(
    state: modules.RNNState, n: int
) -> list[modules.RNNState]:
    """Splits a batched output state back into per-item states.

    Args:
        state (modules.RNNState): shape layers x B x hidden_size, or None.
        n (int): number of items B.

    Returns:
        list[modules.RNNState]: B states each of shape
            layers x 1 x hidden_size.
    """
    if state is None:
        return [None] * n
    if isinstance(state, tuple):
        h, c = state
        return [(h[:, i : i + 1, :], c[:, i : i + 1, :]) for i in range(n)]
    return [state[:, i : i + 1, :] for i in range(n)]


class PointerGeneratorRNNModel(base.PointerGeneratorModel):
    """Abstract base class for pointer-generator models with RNN backends.

    If features are provided, a separate features attention module computes
    the feature encodings. Because of this, the source and features encoders
    can have differently-sized hidden layers. However, because of the way
    they are combined, they must have the same number of hidden layers.

    Whereas See et al. use a two-layer MLP  for the vocabulary distribution,
    this uses a single linear layer.

    This supports optional student forcing during training/validation.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In _Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers)_, pages 1073-1083.

    Args:
        *args: passed to superclass.
        teacher_forcing (bool, optional).
        **kwargs: passed to superclass.

    Raises:
        base.ConfigurationError: Number of encoder layers and decoder layers
            must match.
    """

    classifier: nn.Linear
    generation_probability: modules.GenerationProbability
    teacher_forcing: bool

    def __init__(
        self,
        *args,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.has_features_encoder:
            self.features_attention = modules.Attention(
                self.features_encoder.output_size, self.decoder_hidden_size
            )
        self.decoder = self.get_decoder()
        self.generation_probability = modules.GenerationProbability(
            self.embedding_size,
            self.decoder_hidden_size,
            self.decoder_input_size,
        )
        self.classifier = nn.Linear(
            self.decoder_hidden_size + self.decoder_input_size,
            self.target_vocab_size,
        )
        # Compatibility check.
        if self.source_encoder.layers != self.decoder_layers:
            raise base.ConfigurationError(
                f"Number of encoder layers ({self.source_encoder.layers}) and "
                f"decoder layers ({self.decoder_layers}) must match"
            )
        self.teacher_forcing = teacher_forcing
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                # Modules.
                "classifier",
                "decoder",
                "embeddings",
                "generation_probability",
                "features_encoder",
                "source_encoder",
                # Options that can change between training and prediction.
                "beam_width",
            ]
        )

    def beam_decode(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        features_encoded: torch.Tensor | None = None,
        features_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decodes with beam search, supporting arbitrary batch sizes.

        Each item in the batch gets its own independent beam of width
        beam_width. Decoding halts once every beam across every batch item
        has reached END, or max_target_length steps have elapsed.

        Args:
            source (torch.Tensor).
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            features_encoded (torch.Tensor, optional).
            features_mask (torch.Tensor, optional).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predictions of shape
                B x beam_width x seq_len and log-likelihoods of shape
                B x beam_width.
        """
        batch_size = source_mask.size(0)
        batched_state = self.decoder.initial_state(batch_size)
        per_item_states = _split_state(batched_state, batch_size)
        batched_beam = beam_search.BatchedBeam(
            self.beam_width, batch_size, per_item_states
        )
        for _ in range(self.max_target_length):
            if batched_beam.final:
                break
            self._beam_decode_step(
                batched_beam,
                source,
                source_encoded,
                source_mask,
                features_encoded,
                features_mask,
            )
        return (
            batched_beam.predictions(self.device),
            batched_beam.scores(self.device),
        )

    def _beam_decode_step(
        self,
        batched_beam: beam_search.BatchedBeam,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        features_encoded: torch.Tensor | None,
        features_mask: torch.Tensor | None,
    ) -> None:
        """Runs one decode step for all active cells and updates the beam.

        Args:
            batched_beam (beam_search.BatchedBeam): beam to update in place.
            source (torch.Tensor).
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            features_encoded (torch.Tensor, optional).
            features_mask (torch.Tensor, optional).
        """
        symbols, item_indices, states, index_map = (
            batched_beam._collect_active(self.device)
        )
        if not index_map:
            return
        expanded_source = source[item_indices]
        expanded_source_encoded = source_encoded[item_indices]
        expanded_source_mask = source_mask[item_indices]
        expanded_features_encoded = (
            features_encoded[item_indices]
            if features_encoded is not None
            else None
        )
        expanded_features_mask = (
            features_mask[item_indices] if features_mask is not None else None
        )
        # decode_step already returns log-probs:
        # B x 1 x vocab_size -> B x vocab_size.
        batched_cell_state = _batch_states(states)
        log_probs, new_batched_state = self.decode_step(
            expanded_source,
            expanded_source_encoded,
            expanded_source_mask,
            symbols,
            batched_cell_state,
            features_encoded=expanded_features_encoded,
            features_mask=expanded_features_mask,
        )
        scores = log_probs.squeeze(dim=1)
        new_states = _split_output_states(new_batched_state, symbols.size(0))
        batched_beam.push_final_cells()
        batched_beam.fan_out_stateful(scores, new_states, index_map)
        batched_beam.update()

    def decode_step(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        symbol: torch.Tensor,
        state: modules.RNNState,
        *,
        features_encoded: torch.Tensor | None = None,
        features_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, modules.RNNState]:
        """Single decoder step; predicts a distribution for one symbol.

        Args:
            source (torch.Tensor): shape B x src_len.
            source_encoded (torch.Tensor): shape B x src_len x encoder_dim.
            source_mask (torch.Tensor): shape B x src_len.
            symbol (torch.Tensor): shape B x 1.
            state (modules.RNNState).
            features_encoded (torch.Tensor, optional):
                shape B x feat_len x encoder_dim.
            features_mask (torch.Tensor, optional): shape B x feat_len.

        Returns:
            tuple[torch.Tensor, modules.RNNState]: log-probs of shape
                B x 1 x vocab_size and the updated RNN state.
        """
        # TODO: there are a few Law of Demeter violations here. Is there an
        # obvious refactoring?
        embedded = self.decoder.embed(symbol, self.embeddings)
        context, attention_weights = self.decoder.attention(
            source_encoded,
            state.hidden.transpose(0, 1),
            source_mask,
        )
        if self.has_features_encoder:
            features_context, _ = self.features_attention(
                features_encoded,
                state.hidden.transpose(0, 1),
                features_mask,
            )
            context = torch.cat((context, features_context), dim=2)
        _, state = self.decoder.module(
            torch.cat((embedded, context), dim=2), state
        )
        hidden = state.hidden[-1, :, :].unsqueeze(1)
        output_dist = nn.functional.softmax(
            self.classifier(torch.cat((hidden, context), dim=2)),
            dim=2,
        )
        # -> B x 1 x target_vocab_size.
        pointer_dist = torch.zeros(
            (symbol.size(0), 1, self.target_vocab_size),
            device=self.device,
            dtype=attention_weights.dtype,
        )
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        pointer_dist.scatter_add_(2, source.unsqueeze(1), attention_weights)
        # Probability of generating from output_dist.
        gen_probs = self.generation_probability(context, hidden, embedded)
        scaled_output_dist = output_dist * gen_probs
        scaled_pointer_dist = pointer_dist * (1 - gen_probs)
        # First argument is log-probs.
        return torch.log(scaled_output_dist + scaled_pointer_dist), state

    def init_embeddings(
        self,
        num_embeddings: int,
        embedding_size: int,
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int).
            embedding_size (int).

        Returns:
            nn.Embedding: embedding layer.
        """
        return embeddings.normal_embedding(num_embeddings, embedding_size)

    def forward(
        self,
        batch: data.Batch,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        Beam search returns a tuple with a tensor of top predictions
        and the likelihoods (unnormalized sub of sequence log-probabilities)
        for each prediction; greedy search just returns the tensor of the
        one-best predictions.

        Args:
            batch (data.Batch).

        Returns:
            tuple[torch.Tensor, torch.Tensor] | torch.Tensor.

        Raises:
            base.ConfigurationError: Features encoder specified but no feature
                column specified.
            base.ConfigurationError: Features column specified but no feature
                encoder specified.
        """
        source_encoded = self.source_encoder(
            batch.source, self.embeddings, is_source=True
        )
        if self.has_features_encoder:
            if not batch.has_features:
                raise base.ConfigurationError(
                    "Features encoder specified but "
                    "no feature column specified"
                )
            features_encoded = self.features_encoder(
                batch.features,
                self.embeddings,
                is_source=False,
            )
            if self.beam_width > 1:
                return self.beam_decode(
                    batch.source.tensor,
                    source_encoded,
                    batch.source.mask,
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
            elif self.training or self.validating:
                # This version supports teacher forcing.
                return self.greedy_decode_train_validate(
                    batch.source.tensor,
                    source_encoded,
                    batch.source.mask,
                    target=(
                        batch.target.tensor if self.teacher_forcing else None
                    ),
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
            else:
                return self.greedy_decode_predict_test(
                    batch.source.tensor,
                    source_encoded,
                    batch.source.mask,
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
        elif batch.has_features:
            raise base.ConfigurationError(
                "Features column specified but no feature encoder specified"
            )
        elif self.beam_width > 1:
            return self.beam_decode(
                batch.source.tensor, source_encoded, batch.source.mask
            )
        elif self.training or self.validating:
            # This version supports teacher forcing.
            return self.greedy_decode_train_validate(
                batch.source.tensor,
                source_encoded,
                batch.source.mask,
                target=batch.target.tensor if self.teacher_forcing else None,
            )
        else:
            return self.greedy_decode_predict_test(
                batch.source.tensor,
                source_encoded,
                batch.source.mask,
            )

    def greedy_decode_train_validate(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        features_encoded: torch.Tensor | None = None,
        features_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decodes greedily during training and validation.

        Provide the target for teacher forcing.

        If the target is provided, decoding halts at target length. If the
        target is not provided, decoding will halt once each sequence in the
        batch generates END or the maximum target length is reached,
        whichever comes first.

        Args:
            source (torch.Tensor): source symbols, needed to compute pointer
                weights.
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            features_encoded (torch.Tensor, optional).
            features_mask (torch.Tensor, optional).
            target (torch.Tensor, optional): target symbols; if provided,
                these are used for teacher forcing.

        Returns:
            torch.Tensor: predictions of B x target_vocab_size x seq_len.
        """
        batch_size = source_mask.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        predictions = []
        if target is None:
            target_length = self.max_target_length
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            target_length = target.size(1)
        for t in range(target_length):
            log_probs, state = self.decode_step(
                source,
                source_encoded,
                source_mask,
                symbol,
                state,
                features_encoded=features_encoded,
                features_mask=features_mask,
            )
            predictions.append(log_probs.squeeze(1))
            if target is None:
                # Student forcing.
                symbol = log_probs.argmax(dim=2)
                final = torch.logical_or(final, symbol == special.END_IDX)
                if final.all():
                    break
            else:
                # Teacher forcing.
                symbol = target[:, t].unsqueeze(1)
        predictions = torch.stack(predictions, dim=2)
        return predictions

    def greedy_decode_predict_test(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        features_encoded: torch.Tensor | None = None,
        features_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decodes greedily during prediction and testing.

        These are different because teacher forcing is not supported, but
        decoding halts once each sequence in the batch generates END or the
        maximum target length is reached, whichever comes first.

        Args:
            source (torch.Tensor): source symbols, needed to compute pointer
                weights.
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            features_encoded (torch.Tensor, optional).
            features_mask (torch.Tensor, optional).

        Returns:
            torch.Tensor: predictions of B x target_vocab_size x seq_len.
        """
        batch_size = source_mask.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        predictions = []
        final = torch.zeros(batch_size, device=self.device, dtype=bool)
        for _ in range(self.max_target_length):
            log_probs, state = self.decode_step(
                source,
                source_encoded,
                source_mask,
                symbol,
                state,
                features_encoded=features_encoded,
                features_mask=features_mask,
            )
            predictions.append(log_probs.squeeze(1))
            symbol = log_probs.argmax(dim=2)
            final = torch.logical_or(final, symbol == special.END_IDX)
            if final.all():
                break
        predictions = torch.stack(predictions, dim=2)
        return predictions

    @property
    def decoder_input_size(self) -> int:
        # We concatenate along the encoding dimension.
        if self.has_features_encoder:
            return (
                self.source_encoder.output_size
                + self.features_encoder.output_size
            )
        else:
            return self.source_encoder.output_size


class PointerGeneratorGRUModel(PointerGeneratorRNNModel):
    """Pointer-generator model with an GRU backend."""

    def get_decoder(self) -> modules.SoftAttentionGRUDecoder:
        return modules.SoftAttentionGRUDecoder(
            attention_input_size=self.source_encoder.output_size,
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
        )

    @property
    def name(self) -> str:
        return "pointer-generator GRU"


class PointerGeneratorLSTMModel(PointerGeneratorRNNModel):
    """Pointer-generator model with an LSTM backend."""

    def get_decoder(self) -> modules.SoftAttentionLSTMDecoder:
        return modules.SoftAttentionLSTMDecoder(
            attention_input_size=self.source_encoder.output_size,
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
        )

    @property
    def name(self) -> str:
        return "pointer-generator LSTM"
