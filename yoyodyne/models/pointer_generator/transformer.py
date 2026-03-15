"""Pointer-generator transformer model classes."""

from typing import Optional, Tuple

import torch
from torch import nn

from ... import data, special
from .. import beam_search, defaults, embeddings, modules
from . import base


class PointerGeneratorTransformerModel(base.PointerGeneratorModel):
    """Pointer-generator model with a transformer backend.

    After:
        Singer, A., and Kann, K. 2020. The NYU-CUBoulder Systems for
        SIGMORPHON 2020 Task 0 and Task 2. In _Proceedings of the 17th
        SIGMORPHON Workshop on Computational Research in Phonetics, Phonology,
        and Morphology_, pages 90–98.

    Args:
        *args: passed to the superclass.
        attention_heads (int, optional).
        teacher_forcing (bool, optional).
        **kwargs: passed to the superclass.
    """

    attention_heads: int
    classifier: nn.Linear
    decoder_positional_encoding: modules.BasePositionalEncoding
    generation_probability: modules.GenerationProbability
    teacher_forcing: bool

    def __init__(
        self,
        *args,
        attention_heads: int = defaults.ATTENTION_HEADS,
        decoder_positional_encoding: Optional[
            modules.BasePositionalEncoding
        ] = None,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention_heads = attention_heads
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )
        self.decoder_positional_encoding = decoder_positional_encoding
        self.decoder = self.get_decoder()
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
        self.teacher_forcing = teacher_forcing
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                # Modules,.
                "classifier",
                "decoder",
                "decoder_positional_encoding",
                "embeddings",
                "features_encoder",
                "generation_probability",
                "source_encoder",
                # Options that can change between training and prediction.
                "beam_width",
            ],
        )

    def beam_decode(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes with beam search.

        Decoding halts once all sequences in the beam have reached END. It is
        not currently possible to combine this with loss computation or
        teacher forcing.

        The implementation assumes batch size is 1, but both inputs and outputs
        are still assumed to have a leading dimension representing batch size.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            features_encoded (torch.Tensor, optional): encoded feature symbols.
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
                "Beam search is not supported for batch_size > 1"
            )
        beam = beam_search.Beam(self.beam_width)
        for _ in range(self.max_target_length):
            for cell in beam.cells:
                if cell.final:
                    beam.push(cell)
                else:
                    predictions = torch.tensor(
                        cell.symbols, device=self.device
                    ).unsqueeze(0)
                    log_probs = self.decode_step(
                        source,
                        source_encoded,
                        source_mask,
                        predictions,
                        None,
                        features_encoded=features_encoded,
                        features_mask=features_mask,
                    )
                    # decode_step returns log-probs for all target positions;
                    # we only need the last one.
                    scores = log_probs[:, -1, :].squeeze(0)
                    for new_cell in cell.extensions(scores):
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
        target: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single decoder step.

        This will work on any sequence length, and returns output
        probabilities for all targets, meaning we can use this method for
        student forcing, in which only a single new token is decoded at a time,
        or for teacher forcing, in which all tokens are be decoded in
        parallel using a diagonal mask.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source_symbols.
            source_mask (torch.Tensor): mask for the source.
            target (torch.Tensor): tensor of predictions thus far.
            target_mask (torch.Tensor): mask for the target.
            features_encoded (torch.Tensor, optional): encoded features
                symbols.
            features_mask (torch.Tensor, optional): mask for the features.

        Returns:
            torch.Tensor: predictions for that state.
        """
        self.decoder.attention_output.clear()
        if self.has_features_encoder:
            decoded, target_embedded = self.decoder(
                source_encoded,
                source_mask,
                target,
                target_mask,
                self.embeddings,
                features_encoded=features_encoded,
                features_mask=features_mask,
            )
        else:
            decoded, target_embedded = self.decoder(
                source_encoded,
                source_mask,
                target,
                target_mask,
                self.embeddings,
            )
        # Outputs from the multi-headed attention from each decoder step to
        # the encoded source; values have been averaged over each attention
        # head.
        # -> B x target_seq_len x source_seq_len.
        mha_outputs = self.decoder.attention_output[0]
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

    def get_decoder(self) -> modules.PointerGeneratorTransformerDecoder:
        return modules.PointerGeneratorTransformerDecoder(
            attention_heads=self.attention_heads,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            has_features_encoder=self.has_features_encoder,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_decoder_length,
            positional_encoding=self.decoder_positional_encoding,
        )

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.

        Returns:
            nn.Embedding: embedding layer.
        """
        return embeddings.xavier_embedding(num_embeddings, embedding_size)

    def forward(
        self,
        batch: data.Batch,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            torch.Tensor.

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
            if (self.training or self.validating) and self.teacher_forcing:
                batch_size = len(batch)
                symbol = self.start_symbol(batch_size)
                target = torch.cat((symbol, batch.target.tensor), dim=1)
                target_mask = torch.cat(
                    (
                        torch.zeros_like(symbol, dtype=bool),
                        batch.target.mask,
                    ),
                    dim=1,
                )
                log_probs = self.decode_step(
                    batch.source.tensor,
                    source_encoded,
                    batch.source.mask,
                    target,
                    target_mask,
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
                # Truncates the prediction generated by the END_IDX token,
                # which corresponds to nothing in the target tensor.
                return log_probs[:, :-1, :].transpose(1, 2)
            elif self.beam_width > 1:
                return self.beam_decode(
                    batch.source.tensor,
                    source_encoded,
                    batch.source.mask,
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
            else:
                return self.greedy_decode(
                    batch.source.tensor,
                    source_encoded,
                    batch.source.mask,
                    features_encoded=features_encoded,
                    features_mask=batch.features.mask,
                )
        elif batch.has_features:
            raise base.ConfigurationError(
                "Feature column specified but no feature encoder specified"
            )
        if (self.training or self.validating) and self.teacher_forcing:
            batch_size = len(batch)
            symbol = self.start_symbol(batch_size)
            target = torch.cat((symbol, batch.target.tensor), dim=1)
            target_mask = torch.cat(
                (
                    torch.zeros_like(symbol, dtype=bool),
                    batch.target.mask,
                ),
                dim=1,
            )
            log_probs = self.decode_step(
                batch.source.tensor,
                source_encoded,
                batch.source.mask,
                target,
                target_mask,
            )
            # Truncates the prediction generated by the END_IDX token, which
            # corresponds to nothing in the target tensor.
            return log_probs[:, :-1, :].transpose(1, 2)
        elif self.beam_width > 1:
            return self.beam_decode(
                batch.source.tensor,
                source_encoded,
                batch.source.mask,
            )
        else:
            return self.greedy_decode(
                batch.source.tensor,
                source_encoded,
                batch.source.mask,
            )

    def greedy_decode(
        self,
        source: torch.Tensor,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes greedily during prediction and testing.

        This performs student forcing.

        Decoding halts once each sequence in the batch generates END or the
        maximum target length is reached, whichever comes first.

        Args:
            source (torch.Tensor): source symbols, used to compute pointer
                weights.
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            features_encoded (torch.Tensor, optional): encoded feaure symbols.
            features_mask (torch.Tensor, optional): mask for the features.

        Returns:
            torch.Tensor: predictions of B x target_vocab_size x seq_len.
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
        final = torch.zeros(batch_size, device=self.device, dtype=bool)
        for _ in range(self.max_target_length):
            if self.has_features_encoder:
                assert features_encoded is not None
                assert features_mask is not None
                log_probs = self.decode_step(
                    source,
                    source_encoded,
                    source_mask,
                    torch.stack(predictions, dim=1),
                    None,
                    features_encoded=features_encoded,
                    features_mask=features_mask,
                )
            else:
                log_probs = self.decode_step(
                    source,
                    source_encoded,
                    source_mask,
                    torch.stack(predictions, dim=1),
                    None,
                )
            log_probs = log_probs[:, -1, :]  # From last symbol.
            outputs.append(log_probs)
            symbol = log_probs.argmax(dim=1)
            final = torch.logical_or(final, symbol == special.END_IDX)
            if final.all():
                break
            predictions.append(symbol)
        outputs = torch.stack(outputs, dim=2)
        return outputs

    @property
    def max_decoder_length(self) -> int:
        return self.max_target_length + 1  # Including the START symbol.

    @property
    def name(self) -> str:
        return "pointer-generator transformer"


class RotaryPointerGeneratorTransformerModel(PointerGeneratorTransformerModel):
    """Pointer-generator transformer model using rotary positional encoding.

    RoPE is applied inside attention layers rather than to the token
    embeddings, so this model does not accept a decoder_positional_encoding
    argument. Passing one is a configuration error.

    All encoders used with this model should also use rotary positional
    encoding. Specifically, use RotaryTransformerEncoder or
    RotaryFeatureInvariantTransformerEncoder as the source_encoder and
    features_encoder. Mixing rotary decoders with non-rotary encoders
    is possible but is not recommended.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("decoder_positional_encoding") is not None:
            raise base.ConfigurationError(
                f"{self.__class__.__name__} does not accept "
                "decoder_positional_encoding"
            )
        super().__init__(*args, **kwargs)

    def get_decoder(
        self,
    ) -> modules.RotaryPointerGeneratorTransformerDecoder:
        return modules.RotaryPointerGeneratorTransformerDecoder(
            attention_heads=self.attention_heads,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            has_features_encoder=self.has_features_encoder,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_decoder_length,
        )

    @property
    def name(self) -> str:
        return f"rotary {super().name}"
