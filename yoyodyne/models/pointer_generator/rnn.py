"""Pointer-generator RNN model classes."""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from ... import data, special
from .. import beam_search, defaults, embeddings, modules
from . import base


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
                "classifier",
                "decoder",
                "embeddings",
                "generation_probability",
                "features_encoder",
                "source_encoder",
            ]
        )

    def beam_decode(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes with beam search.

        Implementationally this is almost identical to the method of the same
        name in RNNModel.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            features_encoded (torch.Tensor, optional): encoded feaure symbols.
            features_mask (torch.Tensor, optional): mask for the features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions of shape
                B x beam_width x seq_len and log-likelihoods of shape
                B x beam_width.

        Raises:
            NotImplementedError: Beam search is not implemented for
                batch_size > 1.
        """
        # TODO: modify to work with batches larger than 1.
        batch_size = source_mask.size(0)
        if batch_size != 1:
            raise NotImplementedError(
                "Beam search is not implemented for batch_size > 1"
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
                    prediction, state = self.decode_step(
                        source,
                        source_encoded,
                        source_mask,
                        symbol,
                        cell.state,
                        features_encoded,
                        features_mask,
                    )
                    logits = nn.functional.log_softmax(
                        prediction.squeeze(), dim=0
                    )
                    for new_cell in cell.extensions(state, logits):
                        beam.push(new_cell)
            beam.update()
            if beam.final:
                break
        return beam.predictions(self.device), beam.logits(self.device)

    def decode_step(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        symbol: torch.Tensor,
        state: modules.RNNState,
        *,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single decoder step.

        This predicts a distribution for one symbol.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            symbol (torch.Tensor): next symbol.
            state (modules.RNNState): RNN state.
            features_encoded (torch.Tensor, optional): encoded features
                symbols.
            features_mask (torch.Tensor, optional): mask for the features.

        Returns:
            Tuple[torch.Tensor, modules.RNNState]: predictions for that state
                and the RNN state.
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
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.

        Returns:
            nn.Embedding: embedding layer.
        """
        return embeddings.normal_embedding(num_embeddings, embedding_size)

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
                B x target_vocab_size x seq_len.

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
        target: Optional[torch.Tensor] = None,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
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
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
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
            target (torch.Tensor, optional): target symbols; if provided,
                these are used for teacher forcing.
            target_length (int, optional): maximum target length during
                decoding. If not specified, max_target_length is used.

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
            num_embeddings=self.num_embeddings,
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
            num_embeddings=self.num_embeddings,
        )

    @property
    def name(self) -> str:
        return "pointer-generator LSTM"
