"""Transformer model classes."""

from typing import Optional

import torch
from torch import nn

from .. import data, defaults, special
from . import base, embeddings, modules


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
        if self.has_features_encoder and (
            self.source_encoder.output_size
            != self.features_encoder.output_size
        ):
            raise base.ConfigurationError(
                "Cannot concatenate source encoding "
                f"({self.source_encoder.output_size}) and features "
                f"encoding ({self.features_encoder.output_size})"
            )
        self.decoder = self.get_decoder()
        self.teacher_forcing = teacher_forcing
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                "classifier",
                "decoder",
                # This lives in the decoder.
                "decoder_positional_encoding",
                "embeddings",
                "features_encoder",
                "source_encoder",
            ]
        )

    def decode_step(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Single decoder step.

        This predicts a distribution for one symbol.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).
            predictions (torch.Tensor): tensor of predictions thus far.

        Returns:
            torch.Tensor: logits.
        """
        decoded, _ = self.decoder(
            encoded, mask, predictions, None, self.embeddings
        )
        # We only need the logits for the most recent time step.
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
            # Truncates the prediction generated by the END_IDX token, which
            # corresponds to nothing in the target tensor.
            return logits[:, :, :-1]
        else:
            return self.greedy_decode(encoded, mask)

    def get_decoder(self) -> modules.TransformerDecoder:
        return modules.TransformerDecoder(
            attention_heads=self.attention_heads,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_decoder_length,
            num_embeddings=self.num_embeddings,
            positional_encoding=self.decoder_positional_encoding,
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
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.

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

    def get_decoder(self) -> modules.RotaryTransformerDecoder:
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
        positional_encoding: Optional[modules.BasePositionalEncoding] = None,
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
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )
        self.positional_encoding = positional_encoding
        self.teacher_forcing = teacher_forcing
        self.decoder = self.get_decoder()
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                "classifier",
                "decoder",
                "embeddings",
                # This lives in the decoder.
                "positional_encoding",
            ]
        )

    def decode_step(
        self,
        prefix: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Single decoder step.

        Args:
            prefix (torch.Tensor).
            predictions (torch.Tensor): tensor of predictions thus far.

        Returns:
            torch.Tensor: logits.
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
            logits = self.classifier(decoded[:, prefix_length:, :]).transpose(
                1, 2
            )
            return logits[:, :, :-1]  # Ignores END.
        else:
            return self.greedy_decode(prefix)

    def get_decoder(self) -> modules.CausalTransformerDecoder:
        return modules.CausalTransformerDecoder(
            attention_heads=self.attention_heads,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_length,
            positional_encoding=self.positional_encoding,
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

    def get_decoder(self) -> modules.RotaryCausalTransformerDecoder:
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
