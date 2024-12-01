"""Transformer model classes."""

import argparse
from typing import Optional

import torch
from torch import nn

from .. import data, defaults, special
from . import base, embeddings, modules


class TransformerModel(base.BaseModel):
    """Base class for transformer models.

    Args:
        source_attention_heads (int).
        max_source_length (int).
        **kwargs: passed to superclass.
    """

    # Model arguments.
    source_attention_heads: int
    # Constructed inside __init__.
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

    def beam_decode(self, *args, **kwargs):
        raise NotImplementedError(
            f"Beam search not implemented for {self.name} model"
        )

    def greedy_decode(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        Args:
            encoder_hidden (torch.Tensor): hidden states from the encoder.
            source_mask (torch.Tensor): mask for the encoded source tokens.
            targets (torch.Tensor, optional): the optional target tokens,
                which is only used for early stopping during validation
                if the decoder has predicted END for every sequence in
                the batch.

        Returns:
            torch.Tensor: predictions from the decoder.
        """
        # The output distributions to be returned.
        outputs = []
        batch_size = encoder_hidden.size(0)
        # The predicted symbols at each iteration.
        predictions = [
            torch.tensor(
                [special.START_IDX for _ in range(encoder_hidden.size(0))],
                device=self.device,
            )
        ]
        # Tracking when each sequence has decoded an END.
        finished = torch.zeros(batch_size, device=self.device)
        for _ in range(self.max_target_length):
            target_tensor = torch.stack(predictions, dim=1)
            # Uses a dummy mask of all ones.
            target_mask = torch.ones_like(target_tensor, dtype=torch.float)
            target_mask = target_mask == 0
            decoder_output = self.decoder(
                encoder_hidden,
                source_mask,
                target_tensor,
                target_mask,
            ).output
            logits = self.classifier(decoder_output)
            last_output = logits[:, -1, :]  # Ignores END.
            outputs.append(last_output)
            # -> B x 1 x 1
            _, pred = torch.max(last_output, dim=1)
            predictions.append(pred)
            # Updates to track which sequences have decoded an END.
            finished = torch.logical_or(
                finished, (predictions[-1] == special.END_IDX)
            )
            # Breaks when all sequences have predicted an END symbol. If we
            # have a target (and are thus computing loss), we only break when
            # we have decoded at least the the same number of steps as the
            # target length.
            if finished.all():
                if targets is None or len(outputs) >= targets.size(-1):
                    break
        # -> B x seq_len x target_vocab_size.
        return torch.stack(outputs).transpose(0, 1)

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.
        """
        if self.training and self.teacher_forcing:
            assert (
                batch.target.padded is not None
            ), "Teacher forcing requested but no target provided"
            # Initializes the start symbol for decoding.
            starts = (
                torch.tensor(
                    [special.START_IDX],
                    device=self.device,
                )
                .repeat(batch.target.padded.size(0))
                .unsqueeze(1)
            )
            target_padded = torch.cat((starts, batch.target.padded), dim=1)
            target_mask = torch.cat(
                (starts == special.PAD_IDX, batch.target.mask), dim=1
            )
            encoder_output = self.source_encoder(batch.source).output
            decoder_output = self.decoder(
                encoder_output,
                batch.source.mask,
                target_padded,
                target_mask,
            ).output
            logits = self.classifier(decoder_output)
            return logits[:, :-1, :]  # Ignores END.
        else:
            encoder_output = self.source_encoder(batch.source).output
            if self.beam_width > 1:
                # Will raise a NotImplementedError.
                return self.beam_decode(
                    encoder_output,
                    batch.source.mask,
                    self.beam_width,
                )
            else:
                # -> B x seq_len x output_size.
                return self.greedy_decode(
                    encoder_output,
                    batch.source.mask,
                    batch.target.padded if batch.target else None,
                )

    @property
    def name(self) -> str:
        return "transformer"

    @staticmethod
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
