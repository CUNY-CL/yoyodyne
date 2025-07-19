"""Transformer model classes."""

from typing import Optional

import torch
from torch import nn

from .. import data, defaults, special
from . import base, embeddings, modules


class Error(Exception):
    pass


class TransformerModel(base.BaseModel):
    """Vanilla transformer model.

    If features are provided, the encodings are fused by concatenation of the
    features encoding with the source encoding on the sequence length
    dimension.

    Args:
        attention_heads (int).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    # Model arguments.
    attention_heads: int
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        attention_heads=defaults.ATTENTION_HEADS,
        **kwargs,
    ):
        self.attention_heads = attention_heads
        super().__init__(
            *args,
            attention_heads=attention_heads,
            **kwargs,
        )
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
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

    def beam_decode(self, *args, **kwargs):
        raise NotImplementedError(
            f"Beam search is not supported by {self.name} model"
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
            source_encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for the source.
            predictions (torch.Tensor): tensor of predictions thus far.

        Returns:
            torch.Tensor: logits.
        """
        # Uses a dummy mask of all zeros.
        target_mask = torch.zeros_like(predictions, dtype=bool)
        decoded, _ = self.decoder(encoded, mask, predictions, target_mask)
        logits = self.classifier(decoded)
        logits = logits[:, -1, :]  # Ignores END.
        return logits

    def forward(self, batch: data.Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            torch.Tensor.
        """
        encoded = self.source_encoder(batch.source)
        mask = batch.source.mask
        if self.has_features_encoder:
            if (
                self.source_encoder.output_size
                != self.features_encoder.output_size
            ):
                raise Error(
                    "Cannot concatenate source and features encoding "
                    f"({self.source_encoder.output_size} != "
                    f"{self.features_encoder.output_size})"
                )
            features_encoded = self.features_encoder(batch.features)
            encoded = torch.cat((encoded, features_encoded), dim=1)
            mask = torch.cat((mask, batch.features.mask), dim=1)
        if self.training and self.teacher_forcing:
            assert (
                batch.has_target
            ), "Teacher forcing requested but no target provided"
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
                encoded,
                mask,
                target,
                target_mask,
            )
            # -> B x target_vocab_size x seq_len.
            logits = self.classifier(decoded).transpose(1, 2)
            return logits[:, :, :-1]  # Ignores END.
        else:
            if self.beam_width > 1:
                # Will raise a NotImplementedError.
                return self.beam_decode(encoded, mask, self.beam_width)
            else:
                return self.greedy_decode(
                    encoded,
                    mask,
                    batch.target.padded if batch.has_target else None,
                )

    def get_decoder(self) -> modules.TransformerDecoder:
        return modules.TransformerDecoder(
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            max_length=self.max_length,
            num_embeddings=self.vocab_size,
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
        predictions = [
            torch.tensor([special.START_IDX], device=self.device).repeat(
                batch_size
            )
        ]
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

    @property
    def max_length(self) -> int:
        if self.has_features_encoder:
            return self.max_source_length + self.max_features_length
        else:
            return self.max_source_length

    @property
    def name(self) -> str:
        return "transformer"
