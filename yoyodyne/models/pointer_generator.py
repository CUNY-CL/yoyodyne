"""Pointer-generator model classes."""

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from .. import data, special
from . import base, beam_search, defaults, modules, rnn, transformer


class Error(Exception):
    pass


class PointerGeneratorModel(base.BaseModel):
    """Abstract base class for pointer-generator models."""

    # Constructed inside __init__.
    geneneration_probability: modules.GenerationProbability

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        This overrides the loss function behavior in
        models.base.BaseModel because we need to use NLLLoss in
        order to factor the addition of two separate probability
        distributions. An NLLLoss-compatible implementation of label smoothing
        is also provided when label smoothing is requested.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        if not self.label_smoothing:
            return nn.NLLLoss(ignore_index=special.PAD_IDX)
        else:
            return self._smooth_nllloss

    def _smooth_nllloss(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the NLLLoss with a smoothing factor such that some
        proportion of the output distribution is replaced with a
        uniform distribution.

        After:

            https://github.com/NVIDIA/DeepLearningExamples/blob/
            8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/
            ConvNets/image_classification/smoothing.py#L18

        Args:
            predictions (torch.Tensor): tensor of prediction
                distribution of shape B x target_vocab_size x seq_len.
            target (torch.Tensor): tensor of golds of shape
                B x seq_len.

        Returns:
            torch.Tensor: loss.
        """
        # -> (B * seq_len) x target_vocab_size.
        predictions = predictions.transpose(1, 2).reshape(
            -1, self.target_vocab_size
        )
        # -> (B * seq_len) x 1.
        target = target.view(-1, 1)
        non_pad_mask = target.ne(special.PAD_IDX)
        # Gets the ordinary loss.
        nll_loss = -predictions.gather(dim=-1, index=target)[
            non_pad_mask
        ].mean()
        # Gets the smoothed loss.
        smooth_loss = -predictions.sum(dim=-1, keepdim=True)[
            non_pad_mask
        ].mean()
        smooth_loss = smooth_loss / self.target_vocab_size
        # Combines both according to label smoothing weight.
        loss = (1.0 - self.label_smoothing) * nll_loss
        loss.add_(self.label_smoothing * smooth_loss)
        return loss


class PointerGeneratorRNNModel(PointerGeneratorModel, rnn.RNNModel):
    """Abstract base class for pointer-generator models with RNN backends.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.encoder_layers != self.decoder_layers:
            raise Error(
                "The number of encoder and decoder layers must match "
                f"({self.encoder_layers} != {self.decoder_layers})"
            )
        # Uses the inherited defaults for the source embeddings and encoder.
        if self.has_features_encoder:
            self.features_attention = modules.Attention(
                self.features_encoder.output_size, self.hidden_size
            )
        self.generation_probability = modules.GenerationProbability(
            self.embedding_size,
            self.hidden_size,
            self.decoder_input_size,
        )
        # Overrides inherited classifier.
        self.classifier = nn.Linear(
            self.hidden_size + self.decoder_input_size, self.target_vocab_size
        )

    def _check_layer_sizes(self) -> None:
        """Checks that encoder and decoder layers are the same number.

        Raises:
            Error.
        """
        if self.encoder_layers != self.decoder_layers:
            raise Error(
                "The number of encoder and decoder layers must match "
                f"({self.encoder_layers} != {self.decoder_layers})"
            )

    def beam_decode(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes with beam search.

        Implementationally this is almost identical to the method of the same
        name in RNNModel except that features are passed separately.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            features_encoded (torch.Tensor, optional): encoded feaure symbols.
            features_mask (torch.Tensor, optional): mask for the features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions of shape
                B x beam_width x seq_length and log-likelihoods of shape
                B x beam_width.
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
                    scores = nn.functional.log_softmax(
                        prediction.squeeze(), dim=0
                    )
                    for new_cell in cell.extensions(state, scores):
                        beam.push(new_cell)
            beam.update()
            if beam.final:
                break
        return beam.predictions(self.device), beam.scores(self.device)

    def decode_step(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        symbol: torch.Tensor,
        state: modules.RNNState,
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
        embedded = self.decoder.embed(symbol)
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
            # -> B x 1 x 4*hidden_size.
            context = torch.cat((context, features_context), dim=2)
        _, state = self.decoder.module(
            torch.cat((embedded, context), dim=2), state
        )
        # -> B x 1 x hidden_size.
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
        return torch.log(scaled_output_dist + scaled_pointer_dist), state

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            batch (data.PaddedBatch).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: beam search
                returns a tuple with a tensor of predictions of shape
                B x beam_width x seq_len and a tensor of B x shape beam_width
                with the likelihood (the unnormalized sum of sequence
                log-probabilities) for each prediction; greedy search returns
                a tensor of predictions of shape
                B x seq_len x target_vocab_size.
        """
        source_encoded = self.source_encoder(batch.source)
        if self.has_features_encoder:
            features_encoded = self.features_encoder(batch.features)
            if self.beam_width > 1:
                return self.beam_decode(
                    batch.source.padded,
                    source_encoded,
                    batch.source.mask,
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
            else:
                return self.greedy_decode(
                    batch.source.padded,
                    source_encoded,
                    batch.source.mask,
                    teacher_forcing=(
                        self.teacher_forcing if self.training else False
                    ),
                    target=batch.target.padded if batch.has_target else None,
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
        elif self.beam_width > 1:
            return self.beam_decode(
                batch.source.padded,
                source_encoded,
                batch.source.mask,
            )
        else:
            return self.greedy_decode(
                batch.source.padded,
                source_encoded,
                batch.source.mask,
                teacher_forcing=(
                    self.teacher_forcing if self.training else False
                ),
                target=batch.target.padded if batch.has_target else None,
            )

    def greedy_decode(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        target: Optional[torch.Tensor] = None,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        A hard upper bound on the length of the decoded strings is provided by
        the length of the target strings (more specifically, the length of the
        longest target string) if a target is provided, or `max_target_length`
        if not. Decoding will halt earlier if no target is provided and all
        sequences have reached END.

        Implementationally this is almost identical to the method of the same
        name in RNNModel except that features are passed separately.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            teacher_forcing (bool, optional): whether or not to decode with
                teacher forcing.
            target (torch.Tensor, optional): target symbols; if provided
                decoding continues up until this length is reached.
            features_encoded (torch.Tensor, optional): encoded feaure symbols.
            features_mask (torch.Tensor, optional): mask for the features.

        Returns:
            torch.Tensor: predictions of B x seq_length x target_vocab_size.
        """
        batch_size = source_mask.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        predictions = []
        if target is None:
            max_num_steps = self.max_target_length
            # Tracks when each sequence has decoded an END.
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            max_num_steps = target.size(1)
        for t in range(max_num_steps):
            prediction, state = self.decode_step(
                source,
                source_encoded,
                source_mask,
                symbol,
                state,
                features_encoded,
                features_mask,
            )
            predictions.append(prediction.squeeze(1))
            # With teacher forcing the next input is the gold symbol for this
            # step; with student forcing, it's the top prediction.
            symbol = (
                target[:, t].unsqueeze(1)
                if teacher_forcing
                else torch.argmax(prediction, dim=2)
            )
            if target is None:
                # Updates which sequences have decoded an END>
                final = torch.logical_or(final, symbol == special.END_IDX)
                if final.all():
                    break
        # -> B x seq_length x target_vocab_size.
        predictions = torch.stack(predictions, dim=1)
        return predictions


class PointerGeneratorGRUModel(PointerGeneratorRNNModel):
    """Pointer-generator model with an GRU backend."""

    def get_decoder(self) -> modules.AttentiveGRUDecoder:
        return modules.AttentiveGRUDecoder(
            attention_input_size=self.source_encoder.output_size,
            decoder_input_size=self.decoder_input_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    @property
    def name(self) -> str:
        return "pointer-generator GRU"


class PointerGeneratorLSTMModel(PointerGeneratorRNNModel):
    """Pointer-generator model with an LSTM backend."""

    def get_decoder(self) -> modules.AttentiveLSTMDecoder:
        return modules.AttentiveLSTMDecoder(
            attention_input_size=self.source_encoder.output_size,
            decoder_input_size=self.decoder_input_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    @property
    def name(self) -> str:
        return "pointer-generator LSTM"


class PointerGeneratorTransformerModel(
    PointerGeneratorModel, transformer.TransformerModel
):
    """Pointer-generator model with a transformer backend.

    After:
        Singer, A., and Kann, K. 2020. The NYU-CUBoulder Systems for
        SIGMORPHON 2020 Task 0 and Task 2. In Proceedings of the 17th
        SIGMORPHON Workshop on Computational Research in Phonetics, Phonology,
        and Morphology, pages 90–98.
    """

    # Model arguments.
    features_attention_heads: int

    def __init__(self, *args, features_attention_heads, **kwargs):
        self.features_attention_heads = features_attention_heads
        super().__init__(*args, **kwargs)
        if self.has_features_encoder:
            self.generation_probability = modules.GenerationProbability(
                self.embedding_size,
                self.embedding_size,
                self.embedding_size,
            )
        else:
            self.generation_probability = modules.GenerationProbability(
                self.embedding_size,
                self.embedding_size,
                self.source_encoder.output_size,
            )

    def decode_step(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Single decoder step.

        This will work on any sequence length, and returns output
        probabilities for all targets, meaning we can use this method for
        greedy decoding, wherein only a single new token is decoded at a time,
        or for teacher-forced training, wherein all tokens can be decoded in
        parallel with a diagonal mask.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source_symbols.
            source_mask (torch.Tensor): mask for the source.
            predictions (torch.Tensor): tensor of predictions thus far.

        Returns:
            torch.Tensor: predictions for that state.
        """
        # Uses a dummy mask of all zeros.
        target_mask = torch.zeros_like(predictions, dtype=bool)
        decoded, target_embedded = self.decoder(
            source_encoded, source_mask, predictions, target_mask
        )
        # Outputs from the multi-headed attention from each decoder step to
        # the encoded source. Values have been averaged over each attention
        # head.
        # -> B x target_seq_len x source_seq_len.
        mha_outputs = self.decoder.attention_output[0]
        # Clears the stored attention result.
        self.decoder.attention_output.clear()
        logits = self.classifier(decoded)
        output_dist = nn.functional.softmax(logits, dim=2)
        # -> B x target-seq_len x target_vocab_size.
        pointer_dist = torch.zeros(
            mha_outputs.size(0),
            mha_outputs.size(1),
            self.target_vocab_size,
            device=self.device,
            dtype=mha_outputs.dtype,
        )
        # Repeats the source indices for each target.
        # -> B x target_seq_len x source_seq_len.
        repeated_source = source.unsqueeze(1).repeat(1, mha_outputs.size(1), 1)
        # Scatters the attention weights onto the pointer_dist at their vocab
        # indices in order to get outputs that match the indexing of the
        # generation probability.
        pointer_dist.scatter_add_(2, repeated_source, mha_outputs)
        # A matrix of context vectors from applying attention to the encoder
        # representations w.r.t. each decoder step.
        context = torch.bmm(mha_outputs, source_encoded)
        # Probability of generating from output_dist.
        gen_probs = self.generation_probability(
            context, decoded, target_embedded
        )
        scaled_pointer_dist = pointer_dist * (1 - gen_probs)
        scaled_output_dist = output_dist * gen_probs
        return torch.log(scaled_output_dist + scaled_pointer_dist)

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.

        Raises:
            NotImplementedError: beam search not implemented.
        """
        # TODO(#313): add support for this.
        if self.has_features_encoder:
            raise NotImplementedError(
                "Separate features encoders are not supported by the "
                f"{self.name} model"
            )
        source_encoded = self.source_encoder(batch.source)
        if self.beam_width > 1:
            # Will raise a NotImplementedError.
            return self.beam_decode(source_encoded, batch.source.mask)
        else:
            return self.greedy_decode(
                batch.source.padded,
                source_encoded,
                batch.source.mask,
                batch.target.padded if batch.has_target else None,
            )

    def get_decoder(
        self,
    ) -> modules.TransformerPointerDecoder:
        return modules.TransformerPointerDecoder(
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            features_attention_heads=self.features_attention_heads,
            layers=self.decoder_layers,
            max_source_length=self.max_source_length,
            num_embeddings=self.vocab_size,
            source_attention_heads=self.source_attention_heads,
            separate_features=self.has_features_encoder,
        )

    def greedy_decode(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        A hard upper bound on the length of the decoded strings is provided by
        the length of the target strings (more specifically, the length of the
        longest target string) if a target is provided, or `max_target_length`
        if not. Decoding will halt earlier if no target is provided and all
        sequences have reached END.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            target (torch.Tensor, optional): target symbols; if provided
                decoding continues until this length is reached.

        Returns:
            torch.Tensor: predictions from the decoder.
        """
        batch_size = source_encoded.size(0)
        # The output distributions to be returned.
        outputs = []
        # The predicted symbols at each iteration.
        predictions = [
            torch.tensor([special.START_IDX], device=self.device).repeat(
                batch_size
            )
        ]
        if target is None:
            max_num_steps = self.max_target_length
            # Tracks when each sequence has decoded an END
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            max_num_steps = target.size(1)
        for _ in range(max_num_steps):
            scores = self.decode_step(
                source,
                source_encoded,
                source_mask,
                torch.stack(predictions, dim=1),
            )
            scores = scores[:, -1, :]
            outputs.append(scores)
            symbol = torch.argmax(scores, dim=1)
            predictions.append(symbol)
            if target is None:
                # Updates which sequences have decoded an END.
                final = torch.logical_or(final, (symbol == special.END_IDX))
                if final.all():
                    break
        # -> B x seq_len x target_vocab_size.
        outputs = torch.stack(outputs, dim=1)
        return outputs

    @property
    def name(self) -> str:
        return "pointer-generator transformer"
