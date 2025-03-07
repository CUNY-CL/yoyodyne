"""Transformer model classes."""

import argparse
from typing import Optional

import torch
from torch import nn

from .. import data, defaults, special
from . import base, embeddings, modules


class TransformerModel(base.BaseModel):
    """Vanilla transformer model.

    Args:
        source_attention_heads (int).
        *args: passed to superclass.
        max_source_length (int).
        **kwargs: passed to superclass.
    """

    # Model arguments.
    source_attention_heads: int  # Constructed inside __init__.
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        source_attention_heads=defaults.SOURCE_ATTENTION_HEADS,
        **kwargs,
    ):
        self.source_attention_heads = source_attention_heads
        super().__init__(
            *args,
            source_attention_heads=source_attention_heads,
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

    def forward(self, batch: data.PaddedBatch) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.

        Raises:
            NotImplementedError: separate features encoders are not supported.
        """
        # TODO(#313): add support for this.
        if self.has_features_encoder:
            raise NotImplementedError(
                "Separate features encoders are not supported by "
                "{self.name} model"
            )
        if self.training and self.teacher_forcing:
            assert (
                batch.has_target
            ), "Teacher forcing requested but no target provided"
            # Initializes the start symbol for decoding.
            batch_size = batch.source.mask.size(0)
            symbol = self.start_symbol(batch_size)
            target_padded = torch.cat((symbol, batch.target.padded), dim=1)
            target_mask = torch.cat(
                (symbol == special.PAD_IDX, batch.target.mask), dim=1
            )
            encoded = self.source_encoder(batch.source)
            decoded = self.decoder(
                encoded,
                batch.source.mask,
                target_padded,
                target_mask,
            )
            logits = self.classifier(decoded)
            logits = logits[:, :-1, :]  # Ignores END.
            return logits
        else:
            encoded = self.source_encoder(batch.source)
            if self.beam_width > 1:
                # Will raise a NotImplementedError.
                return self.beam_decode(
                    encoded,
                    batch.source.mask,
                    self.beam_width,
                )
            else:
                # -> B x seq_len x output_size.
                return self.greedy_decode(
                    encoded,
                    batch.source.mask,
                    batch.target.padded if batch.has_target else None,
                )

    def get_decoder(self) -> modules.transformer.TransformerDecoder:
        return modules.transformer.TransformerDecoder(
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            max_source_length=self.max_source_length,
            num_embeddings=self.vocab_size,
            source_attention_heads=self.source_attention_heads,
        )

    def greedy_decode(
        self,
        encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        Args:
            encoded (torch.Tensor): batch of encoded source symbols.
            source_mask (torch.Tensor): mask.
            target (torch.Tensor, optional): target symbols; if provided
                decoding continues until this length is reached.

        Returns:
            torch.Tensor: predictions from the decoder.
        """
        batch_size = source_mask.size(0)
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
            target_tensor = torch.stack(predictions, dim=1)
            # Uses a dummy mask of all zeros.
            target_mask = torch.zeros_like(target_tensor, dtype=bool)
            decoded = self.decoder(
                encoded,
                source_mask,
                target_tensor,
                target_mask,
            )
            logits = self.classifier(decoded)
            logits = logits[:, -1, :]  # Ignores END.
            outputs.append(logits)
            # -> B.
            symbol = torch.argmax(logits, dim=1)
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
        return "transformer"


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds transformer configuration options to the argument parser.

    These are only needed at training time.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--source_attention_heads",
        type=int,
        default=defaults.SOURCE_ATTENTION_HEADS,
        help="Number of attention heads "
        "(transformer-backed architectures only. Default: %(default)s.",
    )
    parser.add_argument(
        "--features_attention_heads",
        type=int,
        default=defaults.FEATURES_ATTENTION_HEADS,
        help="Number of features attention heads "
        "(transformer-backed pointer-generator only). "
        "Default: %(default)s.",
    )
