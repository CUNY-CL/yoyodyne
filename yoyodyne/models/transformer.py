"""Transformer model classes."""

import torch
from torch import nn

from .. import data, defaults, special
from . import base, beam_search, embeddings, modules


class TransformerModel(base.BaseModel):
    """Vanilla transformer model.

    If features are provided, the encodings are fused by concatenation of the
    features encoding with the source encoding on the sequence length
    dimension.

    This supports optional student forcing during training.

    Args:
        *args: passed to superclass.
        attention_heads (int, optional).
        decoder_positional_encoding (modules.BasePositionalEncoding, optional):
            a positional encoding object for the decoder; if not specified, a
            sinusoidal encoding of the appropriate size will be allocated.
        teacher_forcing (bool, optional).
        **kwargs: passed to superclass.
    """

    # Model arguments.
    attention_heads: int
    classifier: nn.Linear
    decoder_positional_encoding: modules.BasePositionalEncoding
    teacher_forcing: bool

    def __init__(
        self,
        *args,
        attention_heads: int = defaults.ATTENTION_HEADS,
        decoder_positional_encoding: (
            modules.BasePositionalEncoding | None
        ) = None,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention_heads = attention_heads
        if self.has_features_encoder and (
            self.source_encoder.output_size
            != self.features_encoder.output_size
        ):
            raise base.ConfigurationError(
                "Cannot concatenate source encoding "
                f"({self.source_encoder.output_size}) and features "
                f"encoding ({self.features_encoder.output_size})"
            )
        self.decoder = self.get_decoder(decoder_positional_encoding)
        self.teacher_forcing = teacher_forcing
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                # Modules.
                "classifier",
                "decoder",
                "decoder_positional_encoding",
                "embeddings",
                "features_encoder",
                "source_encoder",
                # Options that can change between training and prediction.
                "beam_width",
            ]
        )

    def beam_decode(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes with beam search, supporting arbitrary batch sizes.

        Each item in the batch gets its own independent beam of width
        beam_width. Decoding halts once every beam across every batch item
        has reached END, or max_target_length steps have elapsed.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predictions and
                log-likelihoods.
        """
        batch_size = encoded.size(0)
        batched_beam = beam_search.BatchedBeam(
            self.beam_width, batch_size, [None] * batch_size
        )
        for _ in range(self.max_target_length):
            if batched_beam.final:
                break
            self._beam_decode_step(batched_beam, encoded, mask)
        return (
            batched_beam.predictions(self.device),
            batched_beam.scores(self.device),
        )

    def _beam_decode_step(
        self,
        batched_beam: beam_search.BatchedBeam,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Runs one decode step for all active cells and updates the beam.

        Args:
            batched_beam (beam_search.BatchedBeam).
            encoded (torch.Tensor).
            mask (torch.Tensor).
        """
        sequences, item_indices, index_map = (
            batched_beam.collect_active_sequences(self.device)
        )
        if not index_map:
            return
        expanded_encoded = encoded[item_indices]
        expanded_mask = mask[item_indices]
        logits = self.decode_step(expanded_encoded, expanded_mask, sequences)
        scores = nn.functional.log_softmax(logits, dim=1)
        batched_beam.push_final_cells()
        batched_beam.fan_out_stateless(scores, index_map)
        batched_beam.update()

    def decode_step(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Single decoder step; predicts a distribution for the next symbol.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).
            predictions (torch.Tensor).

        Returns:
            torch.Tensor: logits of shape B x vocab_size.
        """
        decoded, _ = self.decoder(
            encoded, mask, predictions, None, self.embeddings
        )
        return self.classifier(decoded[:, -1, :])

    def forward(self, batch: data.Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            torch.Tensor.

        Raises:
            base.ConfigurationError: Features column specified but no features
                encoder specified.
            base.ConfigurationError: Features encoder specified but no features
                column specified.
        """
        encoded = self.source_encoder(
            batch.source, self.embeddings, is_source=True
        )
        mask = batch.source.mask
        if batch.has_features and not self.has_features_encoder:
            raise base.ConfigurationError(
                "Features column provided but no features encoder specified"
            )
        if self.has_features_encoder:
            if not batch.has_features:
                raise base.ConfigurationError(
                    "Features encoder specified but "
                    "no features column specified"
                )
            features_encoded = self.features_encoder(
                batch.features,
                self.embeddings,
                is_source=False,
            )
            encoded = torch.cat((encoded, features_encoded), dim=1)
            mask = torch.cat((mask, batch.features.mask), dim=1)
        if self.teacher_forcing and (self.training or self.validating):
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
            decoded, _ = self.decoder(
                encoded, mask, target, target_mask, self.embeddings
            )
            logits = self.classifier(decoded).transpose(1, 2)
            return logits[:, :, :-1]  # Ignores END.
        elif self.beam_width > 1:
            return self.beam_decode(encoded, mask)
        else:
            return self.greedy_decode(encoded, mask)

    def get_decoder(
        self,
        positional_encoding: modules.BasePositionalEncoding | None = None,
    ) -> modules.TransformerDecoder:
        return modules.TransformerDecoder(
            attention_heads=self.attention_heads,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_decoder_length,
            positional_encoding=positional_encoding,
        )

    def greedy_decode(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        This performs student forcing.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            torch.Tensor: logits from the decoder.
        """
        batch_size = mask.size(0)
        outputs = []
        predictions = [self.start_symbol(batch_size).squeeze(1)]
        final = torch.zeros(batch_size, device=self.device, dtype=bool)
        for _ in range(self.max_target_length):
            logits = self.decode_step(
                encoded,
                mask,
                torch.stack(predictions, dim=1),
            )
            outputs.append(logits)
            symbol = torch.argmax(logits, dim=1)
            predictions.append(symbol)
            final = torch.logical_or(final, symbol == special.END_IDX)
            if final.all():
                break
        # -> B x target_vocab_size x seq_len.
        outputs = torch.stack(outputs, dim=2)
        return outputs

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int).
            embedding_size (int).

        Returns:
            nn.Embedding: embedding layer.
        """
        return embeddings.xavier_embedding(num_embeddings, embedding_size)

    @property
    def max_length(self) -> int:
        if self.has_features_encoder:
            return self.max_source_length + self.max_features_length
        else:
            return self.max_source_length

    @property
    def max_decoder_length(self) -> int:
        return self.max_target_length + 1  # Including the START symbol.

    @property
    def name(self) -> str:
        return "transformer"


class RotaryTransformerModel(TransformerModel):
    """Transformer model with rotary positional encodings.

    For consistency, the source encoder (and features encoder, if used) should
    also use a rotary variant; see the class docstring note below.

    This model does not enforce that the source or features encoders also use
    RoPE, but mixing rotary and sinusoidal encodings within the same model is
    not recommended; one should pair this with RotaryTransformerEncoder
    source and/or features encoders, or use
    RotaryFeatureInvariantTransformerEncoder as the source encoder.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("decoder_positional_encoding") is not None:
            raise base.ConfigurationError(
                f"{self.__class__.__name__} does not accept "
                "decoder_positiona_encoding"
            )
        super().__init__(*args, **kwargs)

    def get_decoder(
        self, positional_encoding=None
    ) -> modules.RotaryTransformerDecoder:
        return modules.RotaryTransformerDecoder(
            attention_heads=self.attention_heads,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_decoder_length,
        )

    @property
    def name(self) -> str:
        return f"rotary {super().name}"


class CausalTransformerModel(base.BaseModel):
    """Causal transformer model.

    This implements the decoder-only ("prefix LM") transformer architecture in
    which there is no separate encoder, and cross-attention is replaced by
    full attention across the concatenated source and target sequence, with
    masking so the "prefix" (the source and features) are fully self-visible,
    but the target is causally masked (i.e., can only see the prefix and
    earlier target symbols).

    This is not compatible with source or features encoders.

    After:
        Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M.,
        ..., and Liu, P. J. 2020. Exploring the limits of transfer learning
        with a unified text-to-text transformer. Journal of Machine Learning
        Research 21: 1-67.

    Args:
        *args: passed to superclass.
        attention_heads (int, optional).
        positional_encoding (modules.BasePositionalEncoding, optional):
            a positional encoding object; if not specified, a sinusoidal
            encoding of the appropriate size will be allocated.
        teacher_forcing (bool, optional): should teacher (rather than student)
            forcing be used?
        **kwargs: passed to superclass.
    """

    attention_heads: int
    positional_encoding: modules.BasePositionalEncoding
    teacher_forcing: bool
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        attention_heads: int = defaults.ATTENTION_HEADS,
        positional_encoding: modules.BasePositionalEncoding | None = None,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        if kwargs.get("source_encoder") is not None:
            raise base.ConfigurationError(
                f"{self.__class__.__name__} does not support source encoders"
            )
        if kwargs.get("features_encoder", False) is not False:
            raise base.ConfigurationError(
                f"{self.__class__.__name__} does not support features encoders"
            )
        super().__init__(
            source_encoder=None,
            features_encoder=False,
            *args,
            **kwargs,
        )
        self.attention_heads = attention_heads
        self.decoder = self.get_decoder(positional_encoding)
        self.teacher_forcing = teacher_forcing
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                # Modules.
                "classifier",
                "decoder",
                "embeddings",
                "positional_encoding",
                # Options that can change between training and prediction.
                "beam_width",
            ]
        )

    def beam_decode(
        self,
        prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decodes with beam search, supporting arbitrary batch sizes.

        Each item in the batch gets its own independent beam of width
        beam_width. Decoding halts once every beam across every batch item
        has reached END, or max_target_length steps have elapsed.

        Args:
            prefix (torch.Tensor).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predictions and
                log-likelihoods.
        """
        batch_size = prefix.size(0)
        batched_beam = beam_search.BatchedBeam(
            self.beam_width, batch_size, [None] * batch_size
        )
        for _ in range(self.max_target_length):
            if batched_beam.final:
                break
            self._beam_decode_step(batched_beam, prefix)
        return (
            batched_beam.predictions(self.device),
            batched_beam.scores(self.device),
        )

    def _beam_decode_step(
        self,
        batched_beam: beam_search.BatchedBeam,
        prefix: torch.Tensor,
    ) -> None:
        """Runs one decode step for all active cells and updates the beam.

        Args:
            batched_beam (beam_search.BatchedBeam): beam to update in place.
            prefix (torch.Tensor).
        """
        sequences, item_indices, index_map = (
            batched_beam.collect_active_sequences(self.device)
        )
        if not index_map:
            return
        expanded_prefix = prefix[item_indices]
        logits = self.decode_step(expanded_prefix, sequences)
        scores = nn.functional.log_softmax(logits, dim=1)
        batched_beam.push_final_cells()
        batched_beam.fan_out_stateless(scores, index_map)
        batched_beam.update()

    def decode_step(
        self,
        prefix: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Single decoder step; predicts a distribution for the next symbol.

        Args:
            prefix (torch.Tensor).
            predictions (torch.Tensor).

        Returns:
            torch.Tensor: logits of shape B x vocab_size.
        """
        sequence = torch.cat((prefix, predictions), dim=1)
        tensor = data.PaddedTensor.from_tensor(sequence)
        prefix_length = prefix.size(1)
        target_length = predictions.size(1)
        attn_mask = self._get_prefix_mask(prefix_length, target_length)
        decoded = self.decoder(tensor, self.embeddings, mask=attn_mask)
        return self.classifier(decoded[:, -1, :])

    def forward(self, batch: data.Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            torch.Tensor.
        """
        batch_size = len(batch)
        prefix = batch.source.tensor
        if batch.has_features:
            prefix = torch.cat((prefix, batch.features.tensor), dim=1)
        if (self.training or self.validating) and self.teacher_forcing:
            symbol = self.start_symbol(batch_size)
            sequence = torch.cat((prefix, symbol, batch.target.tensor), dim=1)
            tensor = data.PaddedTensor.from_tensor(sequence)
            prefix_length = prefix.size(1)
            target_length = batch.target.tensor.size(1) + 1
            attn_mask = self._get_prefix_mask(prefix_length, target_length)
            decoded = self.decoder(tensor, self.embeddings, mask=attn_mask)
            # We don't need to run the classifier on the prefix symbols.
            decoded = decoded[:, prefix_length:, :]
            logits = self.classifier(decoded).transpose(1, 2)
            return logits[:, :, :-1]  # Ignores END.
        elif self.beam_width > 1:
            return self.beam_decode(prefix)
        else:
            return self.greedy_decode(prefix)

    def get_decoder(
        self,
        positional_encoding: modules.BasePositionalEncoding | None = None,
    ) -> modules.CausalTransformerDecoder:
        return modules.CausalTransformerDecoder(
            attention_heads=self.attention_heads,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_length,
            positional_encoding=positional_encoding,
        )

    def greedy_decode(self, prefix: torch.Tensor) -> torch.Tensor:
        """Decodes the output sequence greedily.

        This performs student forcing.

        Since this is a single stack without a separate encoder, we must
        feed the growing sequence (prefix + predicted target) into the model
        at each step.

        Args:
            prefix (torch.Tensor).

        Returns:
            torch.Tensor: logits.
        """
        batch_size = prefix.size(0)
        outputs = []
        predictions = [self.start_symbol(batch_size).squeeze(1)]
        final = torch.zeros(batch_size, device=self.device, dtype=bool)
        for _ in range(self.max_target_length):
            logits = self.decode_step(
                prefix,
                torch.stack(predictions, dim=1),
            )
            outputs.append(logits)
            symbol = torch.argmax(logits, dim=1)
            predictions.append(symbol)
            final = torch.logical_or(final, (symbol == special.END_IDX))
            if final.all():
                break
        outputs = torch.stack(outputs, dim=2)
        return outputs

    def _get_prefix_mask(
        self, prefix_length: int, target_length: int
    ) -> torch.Tensor:
        """Generates the prefix LM attention mask.

        Mask shape is L x L where L = prefix_length + target_length.

        * Prefix tokens can attend to any prefix token but not to
          target tokens.
        * Target tokens can attend to any prefix token but only to
          earlier target tokens ("causal masking").

        Args:
            prefix_length (int).
            target_length (int).

        Returns:
            torch.Tensor: float mask.
        """
        total_length = prefix_length + target_length
        mask = torch.zeros(
            (total_length, total_length), device=self.device, dtype=torch.float
        )
        # Prevents prefix from attending to target.
        mask[:prefix_length, prefix_length:] = defaults.NEG_INF
        # Causally masks target.
        mask[prefix_length:, prefix_length:] = (
            nn.Transformer.generate_square_subsequent_mask(
                target_length, device=self.device
            )
        )
        return mask

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int
    ) -> nn.Embedding:
        return embeddings.xavier_embedding(num_embeddings, embedding_size)

    @property
    def max_length(self) -> int:
        # Source (plus START/END) plus features plus target (plus START/END).
        # The positional encoding may be oversized if there are no features,
        # but I believe this is harmless.
        return (
            self.max_source_length
            + 2
            + self.max_features_length
            + self.max_target_length
            + 2
        )

    @property
    def name(self) -> str:
        return "causal transformer"


class RotaryCausalTransformerModel(CausalTransformerModel):
    """Causal transformer model with rotary positional encodings.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("positional_encoding") is not None:
            raise base.ConfigurationError(
                f"{self.__class__.__name__} does not accept "
                "positional_encoding"
            )
        super().__init__(*args, **kwargs)

    def get_decoder(
        self, positional_encoding=None
    ) -> modules.RotaryCausalTransformerDecoder:
        return modules.RotaryCausalTransformerDecoder(
            attention_heads=self.attention_heads,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_length,
        )

    @property
    def name(self) -> str:
        return f"rotary {super().name}"
