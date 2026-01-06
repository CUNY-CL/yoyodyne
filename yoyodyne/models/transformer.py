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

    After:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
        A. N., Łukasz, K., and Polosukhin, I. 2017. Attention is all you need.
        In Advances in Neural Information Processing Systems 30, 5998-6008.

    Args:
        *args: passed to superclass.
        attention_heads (int, optional).
        teacher_forcing (bool, optional): should teacher (rather than student)
            forcing be used?
        **kwargs: passed to superclass.
    """

    # Model arguments.
    attention_heads: int
    teacher_forcing: bool
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        attention_heads: int = defaults.ATTENTION_HEADS,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        self.attention_heads = attention_heads
        super().__init__(*args, **kwargs)
        self.teacher_forcing = teacher_forcing
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )
        if (
            self.has_features_encoder
            and self.source_encoder.output_size
            != self.features_encoder.output_size
        ):
            raise base.ConfigurationError(
                "Cannot concatenate source encoding "
                f"({self.source_encoder.output_size}) and features "
                f"encoding ({self.features_encoder.output_size})"
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
        # FIXME am I running too much classifier? It seems like I only need
        # to run it over the last decoded symbol.
        logits = self.classifier(decoded)
        return logits[:, -1, :]

    def forward(self, batch: data.Batch) -> torch.Tensor:
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
            base.ConfigurationError: Teacher forcing requested but no target
                provided.
        """
        encoded = self.source_encoder(
            batch.source, self.embeddings, is_source=True
        )
        mask = batch.source.mask
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
            encoded = torch.cat((encoded, features_encoded), dim=1)
            mask = torch.cat((mask, batch.features.mask), dim=1)
        if self.training and self.teacher_forcing:
            if not batch.has_target:
                raise base.ConfigurationError(
                    "Teacher forcing requested but no target provided"
                )
            batch_size = len(batch)
            symbol = self.start_symbol(batch_size)
            target = torch.cat((symbol, batch.target.padded), dim=1)
            target_mask = torch.cat(
                (
                    torch.ones_like(symbol, dtype=bool),
                    batch.target.mask,
                ),
                dim=1,
            )
            decoded, _ = self.decoder(
                encoded, mask, target, target_mask, self.embeddings
            )
            logits = self.classifier(decoded).transpose(1, 2)
            return logits[:, :, :-1]
        else:
            return self.greedy_decode(
                encoded,
                mask,
                batch.target.padded if batch.has_target else None,
            )

    def get_decoder(self) -> modules.TransformerDecoder:
        return modules.TransformerDecoder(
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            max_length=self.target_max_length,
            num_embeddings=self.num_embeddings,
            attention_heads=self.attention_heads,
        )

    def greedy_decode(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).
            target (torch.Tensor, optional): target symbols; if provided
                decoding continues until this length is reached.

        Returns:
            torch.Tensor: logits from the decoder.
        """
        batch_size = mask.size(0)
        # The output distributions to be returned.
        outputs = []
        # The predicted symbols at each iteration.
        predictions = [self.start_symbol(batch_size).squeeze(1)]
        if target is None:
            max_num_steps = self.max_target_length
            # Tracks when each sequence has decoded an END.
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            max_num_steps = target.size(1)
        for _ in range(max_num_steps):
            logits = self.decode_step(
                encoded,
                mask,
                torch.stack(predictions, dim=1),
            )
            outputs.append(logits)
            symbol = torch.argmax(logits, dim=1)
            predictions.append(symbol)
            if target is None:
                # Updates which sequences have decoded an END.
                final = torch.logical_or(final, (symbol == special.END_IDX))
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
    def name(self) -> str:
        return "transformer"


class DecoderOnlyTransformerModel(base.BaseModel):
    """Decoder-only transformer model.

    This implements the "prefix LM" architecture in which there is no separate
    encoder, and cross-attention is replaced by full attention across the
    concatenated source and target sequence, with a mask such that the "prefix"
    (the source and features) are fully visible to itself, and the target is
    causally masked (i.e., can only see the prefix and earlier target symbols).
    Since this is a decoder-only architecture, it is not compatible with source
    or features encoders. Given that the source and features contribute to
    max_target_length, one is strongly encouraged to increase this value.

    After:
        Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M.,
        ..., and Liu, P. J. 2020. Exploring the limits of transfer learning
        with a unified text-to-text transformer. Journal of Machine Learning
        Research 21: 1-67.

    Args:
        *args: passed to superclass.
        attention_heads (int, optional).
        teacher_forcing (bool, optional): should teacher (rather than student)
            forcing be used?
        **kwargs: passed to superclass.
    """

    attention_heads: int
    teacher_forcing: bool
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        attention_heads: int = defaults.ATTENTION_HEADS,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        self.attention_heads = attention_heads
        super().__init__(
            source_encoder=None,
            features_encoder=False,
            *args,
            **kwargs,
        )
        self.teacher_forcing = teacher_forcing
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )

    def decode_step(
        self,
        prefix_padded: torch.Tensor,
        prefix_mask: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Single decoder step for Prefix LM.

        Args:
            prefix_padded (torch.Tensor).
            prefix_mask (torch.Tensor).
            predictions (torch.Tensor): tensor of predictions thus far.

        Returns:
            torch.Tensor: logits.
        """
        prefix_len = prefix_padded.size(1)
        target_len = predictions.size(1)
        sequence = torch.cat((prefix_padded, predictions), dim=1)
        target_mask = torch.ones_like(predictions, dtype=bool)
        padding_mask = torch.cat((prefix_mask, target_mask), dim=1)
        attn_mask = self._get_prefix_mask(prefix_len, target_len)
        # We wrap the concatenated sequence in a PaddedTensor.
        tensor = data.PaddedTensor(sequence, padding_mask)
        decoded = self.decoder(tensor, self.embeddings, mask=attn_mask)
        logits = self.classifier(decoded)
        return logits[:, :, :-1]  # Ignores END.

    def forward(self, batch: data.Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            torch.Tensor.
        """
        batch_size = len(batch)
        prefix_padded = batch.source.padded
        prefix_mask = batch.source.mask
        if batch.has_features:
            prefix_padded = torch.cat(
                (prefix_padded, batch.features.padded), dim=1
            )
            prefix_mask = torch.cat((prefix_mask, batch.features.mask), dim=1)
        if self.training and self.teacher_forcing:
            if not batch.has_target:
                raise base.ConfigurationError(
                    "Teacher forcing requested but no target provided"
                )
            symbol = self.start_symbol(batch_size)
            target_padded = torch.cat((symbol, batch.target.padded), dim=1)
            target_mask = torch.cat(
                (
                    torch.ones_like(symbol, dtype=bool),
                    batch.target.mask,
                ),
                dim=1,
            )
            sequence = torch.cat((prefix_padded, target_padded), dim=1)
            padding_mask = torch.cat((prefix_mask, target_mask), dim=1)
            prefix_len = prefix_padded.size(1)
            target_len = target_padded.size(1)
            attn_mask = self._get_prefix_mask(prefix_len, target_len)
            # We wrap the concatenated sequence in a PaddedTensor.
            tensor = data.PaddedTensor(sequence, padding_mask)
            # We only need the prediction portion.
            decoded = self.decoder(tensor, self.embeddings, mask=attn_mask)[
                :, prefix_len:, :
            ]
            logits = self.classifier(decoded).transpose(1, 2)
            return logits[:, :, :-1]  # Ignores END.
        else:
            return self.greedy_decode(
                prefix_padded,
                prefix_mask,
                batch.target.padded if batch.has_target else None,
            )

    def get_decoder(self) -> modules.TrasnformerEncoder:
        # We use modules.TransformerEncoder as the underlying stack because
        # it has the necesssary self-attention layers only.
        return modules.TransformerEncoder(
            attention_heads=self.attention_heads,
            dropout=self.decoder_dropout,
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
            # Source and features now count towards "target length", so we
            # incorporate them here.
            #max_length=self.max_length,
            # FIXME I want this to be bigger.
            max_length = self.max_target_length,
        )

    def greedy_decode(
        self,
        prefix_padded: torch.Tensor,
        prefix_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        Since this is a single stack without a separate encoder, we must
        feed the growing sequence (prefix + predicted target) into the model
        at each step.

        Args:
            prefix_padded (torch.Tensor).
            prefix_mask (torch.Tensor).
            target (torch.Tensor, optional): target symbols.

        Returns:
            torch.Tensor: logits.
        """
        batch_size = prefix_padded.size(0)
        # The output distributions to be returned.
        outputs = []
        # The predicted symbols at each iteration.
        predictions = [self.start_symbol(batch_size).squeeze(1)]
        if target is None:
            max_num_steps = self.max_target_length
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            max_num_steps = target.size(1)
        for _ in range(max_num_steps):
            logits = self.decode_step(
                prefix_padded,
                prefix_mask,
                torch.stack(predictions, dim=1),
            )
            outputs.append(logits)
            symbol = torch.argmax(logits, dim=1)
            predictions.append(symbol)
            if target is None:
                final = torch.logical_or(final, (symbol == special.END_IDX))
                if final.all():
                    break
        outputs = torch.stack(outputs, dim=2)
        return outputs

    def _get_prefix_mask(
        self, prefix_len: int, target_len: int
    ) -> torch.Tensor:
        """Generates the prefix LM attention mask.

        Mask shape is L x L where L = prefix_len + target_len.

        * Prefix tokens can attend to any prefix token but to no
          target tokens.
        * Target tokens can attend to any prefix token but to only
          earlier target tokens ("causal masking").

        Args:
            prefix_len (int).
            target_len (int).

        Returns:
            torch.Tensor: float mask (0.0 for allowed, -inf for blocked).
        """
        total_len = prefix_len + target_len
        mask = torch.zeros(
            (total_len, total_len), device=self.device, dtype=torch.float
        )
        # Prevents prefix from attending to target.
        mask[:prefix_len, prefix_len:] = defaults.NEG_INF
        # Causally masks target.
        causal = nn.Transformer.generate_square_subsequent_mask(
            target_len, device=self.device
        )
        mask[prefix_len:, prefix_len:] = causal
        return mask

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int
    ) -> nn.Embedding:
        return embeddings.xavier_embedding(num_embeddings, embedding_size)

    @property
    def max_length(self) -> int:
        # Both source and features now count towards "target length", so we
        # incorporate them here.
        if self.has_features_encoder:
            return (
                self.max_source_length
                + self.max_features_length
                + self.max_target_length
            )
        else:
            return self.max_source_length + self.max_target_length

    @property
    def name(self) -> str:
        return "decoder-only transformer"
