"""RNN model classes.

RNNModel is the base class; it has encoder and decoder modules, the classifier,
and the initial decoder hidden state. Subclassing is used to inject the
concrete decoder modules.
"""

import abc

import torch
from torch import nn

from .. import data, defaults, special
from . import base, beam_search, embeddings, modules


class RNNModel(base.BaseModel):
    """Abstract base class for RNN models.

    The implementation of `get_decoder` in the subclasses determines what kind
    of RNN is used (i.e., GRU or LSTM), and this determines whether the model
    uses a learned attention or not.

    If features are provided, the encodings are fused by concatenation of the
    features encoding with the source encoding on the sequence length
    dimension.

    This supports optional student forcing during training.

    Args:
        *args: passed to superclass.
        teacher_forcing (bool, optional): should teacher (rather than student)
            forcing be used?
        **kwargs: passed to superclass.
    """

    teacher_forcing: bool
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decoder = self.get_decoder()
        self.teacher_forcing = teacher_forcing
        self.classifier = nn.Linear(
            self.decoder_hidden_size, self.target_vocab_size
        )
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                # Modules.
                "classifier",
                "decoder",
                "embeddings",
                "features_encoder",
                "source_encoder",
                # Options that can change between training and prediction.
                "beam_width",
            ]
        )

    def beam_decode(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes with beam search.

        Each item in the batch gets its own independent beam of width
        beam_width. Decoding halts once every beam across every batch item
        has reached END, or max_target_length steps have elapsed.

        Args:
            context (torch.Tensor): shape B x src_len x encoder_dim.
            mask (torch.Tensor): shape B x src_len.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predictions of shape
                B x beam_width x seq_len and log-likelihoods of shape
                B x beam_width.
        """
        batch_size = context.size(0)
        per_item_states = self.decoder.initial_state(batch_size).split(
            batch_size
        )
        batched_beam = beam_search.BatchedBeam(
            self.beam_width, batch_size, per_item_states
        )
        for _ in range(self.max_target_length):
            if batched_beam.final:
                break
            self._beam_decode_step(batched_beam, context, mask)
        return (
            batched_beam.predictions(self.device),
            batched_beam.scores(self.device),
        )

    def _beam_decode_step(
        self,
        batched_beam: beam_search.BatchedBeam,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Runs one decode step for all active cells and updates the beam.

        Collects active cells, runs a single batched forward pass, then fans
        scores and new states back into the per-item beams.

        Args:
            batched_beam (beam_search.BatchedBeam).
            context (torch.Tensor).
            mask (torch.Tensor).
        """
        symbols, item_indices, states, index_map = batched_beam.collect_active(
            self.device
        )
        if not index_map:
            return
        expanded_context = context[item_indices]
        expanded_mask = mask[item_indices]
        batched_cell_state = modules.RNNState.batch(states)
        logits, new_batched_state = self.decode_step(
            symbols, expanded_context, expanded_mask, batched_cell_state
        )
        # logits: B x 1 x vocab_size -> B x vocab_size log-probs.
        scores = nn.functional.log_softmax(logits.squeeze(dim=1), dim=1)
        new_states = new_batched_state.split(symbols.size(0))
        batched_beam.push_final_cells()
        batched_beam.fan_out_stateful(scores, new_states, index_map)
        batched_beam.update()

    def decode_step(
        self,
        symbol: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        state: modules.RNNState,
    ) -> tuple[torch.Tensor, modules.RNNState]:
        """Single step of the decoder.

        Args:
            symbol (torch.Tensor).
            context (torch.Tensor).
            mask (torch.Tensor).
            state (modules.RNNState).

        Returns:
            tuple[torch.Tensor, modules.RNNState]: logits of shape
                B x 1 x vocab_size and the updated RNN state.
        """
        decoded, state = self.decoder(
            symbol, self.embeddings, context, mask, state
        )
        logits = self.classifier(decoded)
        return logits, state

    @property
    def decoder_input_size(self) -> int:
        # Features concatenation does not change this.
        return self.source_encoder.output_size

    def forward(
        self,
        batch: data.Batch,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        Beam search returns a tuple with a tensor of top predictions
        and the log-likelihoods for each prediction; greedy search just
        returns the tensor of the one-best predictions.

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
        sequence = batch.source.tensor
        encoded = self.source_encoder(batch.source, self.embeddings)
        mask = batch.source.mask
        if self.has_features_encoder:
            if not batch.has_features:
                raise base.ConfigurationError(
                    "Features encoder specified but "
                    "no feature column specified"
                )
            if (
                self.source_encoder.output_size
                != self.features_encoder.output_size
            ):
                raise base.ConfigurationError(
                    "Cannot concatenate source encoding "
                    f"({self.source_encoder.output_size}) and features "
                    f"encoding ({self.features_encoder.output_size})"
                )
            sequence = torch.cat((sequence, batch.features.tensor), dim=1)
            features_encoded = self.features_encoder(
                batch.features, self.embeddings
            )
            encoded = torch.cat((encoded, features_encoded), dim=1)
            mask = torch.cat((mask, batch.features.mask), dim=1)
        elif batch.has_features:
            raise base.ConfigurationError(
                "Features column specified but no feature encoder specified"
            )
        context = self.decoder.get_context(sequence, encoded)
        if self.beam_width > 1:
            return self.beam_decode(context, mask)
        else:
            if self.training or self.validating:
                # This version supports teacher forcing.
                return self.greedy_decode_train_validate(
                    context,
                    mask,
                    batch.target.tensor if self.teacher_forcing else None,
                )
            else:
                return self.greedy_decode_predict_test(context, mask)

    @abc.abstractmethod
    def get_decoder(self) -> modules.RNNDecoder: ...

    def greedy_decode_train_validate(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decodes greedily during training and validation.

        Provide the target for teacher forcing.

        If the target is provided, decoding halts at target length. If the
        target is not provided, decoding halts once each sequence in the
        batch generates END or the maximum target length is reached,
        whichever comes first.

        Args:
            context (torch.Tensor).
            mask (torch.Tensor).
            target (torch.Tensor, optional): target symbols; if provided,
                these are used for teacher forcing.

        Returns:
            torch.Tensor: predictions of B x target_vocab_size x seq_len.
        """
        batch_size = context.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        predictions = []
        if target is None:
            target_length = self.max_target_length
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            target_length = target.size(1)
        for t in range(target_length):
            logits, state = self.decode_step(symbol, context, mask, state)
            predictions.append(logits.squeeze(1))
            if target is None:
                # Student forcing.
                symbol = logits.argmax(dim=2)
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
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes greedily during prediction and testing.

        These are different because teacher forcing is not supported, but
        decoding will halt once each sequence in the batch generates END or
        the maximum target length is reached, whichever comes first.

        Args:
            context (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            torch.Tensor: predictions of B x target_vocab_size x seq_len.
        """
        batch_size = context.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        predictions = []
        final = torch.zeros(batch_size, device=self.device, dtype=bool)
        for _ in range(self.max_target_length):
            logits, state = self.decode_step(symbol, context, mask, state)
            predictions.append(logits.squeeze(1))
            symbol = logits.argmax(dim=2)
            final = torch.logical_or(final, symbol == special.END_IDX)
            if final.all():
                break
        predictions = torch.stack(predictions, dim=2)
        return predictions

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

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class GRUModel(RNNModel):
    """GRU model without attention."""

    def get_decoder(self) -> modules.GRUDecoder:
        return modules.GRUDecoder(
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
        )

    @property
    def name(self) -> str:
        return "GRU"


class LSTMModel(RNNModel):
    """LSTM model without attention.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def get_decoder(self) -> modules.LSTMDecoder:
        return modules.LSTMDecoder(
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
        )

    @property
    def name(self) -> str:
        return "LSTM"


class SoftAttentionGRUModel(GRUModel):
    """GRU model with attention."""

    def get_decoder(self) -> modules.SoftAttentionGRUDecoder:
        return modules.SoftAttentionGRUDecoder(
            attention_input_size=self.decoder_input_size,
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
        )

    @property
    def name(self) -> str:
        return "soft attention GRU"


class SoftAttentionLSTMModel(LSTMModel):
    """LSTM model with attention."""

    def get_decoder(self) -> modules.SoftAttentionLSTMDecoder:
        return modules.SoftAttentionLSTMDecoder(
            attention_input_size=self.decoder_input_size,
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
        )

    @property
    def name(self) -> str:
        return "soft attention LSTM"
