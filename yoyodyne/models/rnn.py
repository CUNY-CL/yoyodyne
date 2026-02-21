"""RNN model classes.

RNNModel is the base class; it has encoder and decoder modules, the classifier,
and the initial decoder hidden state and provides methods for greedy and beam
decoding. Subclassing is used to inject the concrete decoder modules.
"""

import abc
from typing import Optional, Tuple, Union

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
        self.teacher_forcing = teacher_forcing
        self.classifier = nn.Linear(
            self.decoder_hidden_size, self.target_vocab_size
        )
        self.decoder = self.get_decoder()
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                "classifier",
                "decoder",
                "embeddings",
                "features_encoder",
                "source_encoder",
            ]
        )

    def beam_decode(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes with beam search.

        Decoding halts once all sequences in a batch have reached END. It is
        not currently possible to combine this with loss computation or
        teacher forcing.

        The implementation assumes batch size is 1, but both inputs and outputs
        are still assumed to have a leading dimension representing batch size.

        Args:
            context (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions of shape
                B x beam_width x seq_len and log-likelihoods of shape
                B x beam_width.
        """
        # TODO: modify to work with batches larger than 1.
        batch_size = context.size(0)
        if batch_size != 1:
            raise NotImplementedError(
                "Beam search is not supported for batch_size > 1"
            )
        state = self.decoder.initial_state(batch_size)
        # The start symbol is not needed here because the beam puts that in
        # automatically.
        beam = beam_search.Beam(self.beam_width, state)
        for _ in range(self.max_target_length):
            for cell in beam.cells:
                if cell.final:
                    beam.push(cell)
                else:
                    symbol = torch.tensor([[cell.symbol]], device=self.device)
                    logits, state = self.decode_step(
                        symbol, context, mask, state
                    )
                    scores = nn.functional.log_softmax(
                        logits.squeeze(1), dim=0
                    )
                    for new_cell in cell.extensions(state, scores):
                        beam.push(new_cell)
            beam.update()
            if beam.final:
                break
        return beam.predictions(self.device), beam.scores(self.device)

    def decode_step(
        self,
        symbol: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        state: modules.RNNState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step of the decoder.

        Args:
            symbol (torch.Tensor): previously decoded symbol(s) of shape B x 1.
            context (torch.Tensor).
            mask (torch.Tensor).
            state (RNNState).

        Returns:
            Tuple[torch.Tensor, modules.RNNState]: logits and the RNN state.
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
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: beam search
                returns a tuple with a tensor of predictions of shape
                B x beam_width x seq_len and a tensor of B x shape beam_width
                with the likelihood (the unnormalized sum of sequence
                log-probabilities) for each prediction; greedy search returns
                a tensor of predictions of shape
                B x seq_len x target_vocab_size.

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
        target: Optional[torch.Tensor] = None,
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
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.

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
            num_embeddings=self.num_embeddings,
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
            num_embeddings=self.num_embeddings,
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
            num_embeddings=self.num_embeddings,
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
            num_embeddings=self.num_embeddings,
        )

    @property
    def name(self) -> str:
        return "soft attention LSTM"
