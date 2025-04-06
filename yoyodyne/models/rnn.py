"""RNN model classes.

RNNModel is the base class; it has encoder and decoder modules, the classifier,
and the initial decoder hidden state and provides methods for greedy and beam
decoding. Subclassing is used to inject the concrete decoder modules.
"""

import abc
import argparse
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .. import data, defaults, special
from . import base, beam_search, embeddings, modules


class RNNModel(base.BaseModel):
    """Abstract base class for RNN models.

    The implementation of `get_decoder` in the subclasses determines what kind
    of RNN is used (i.e., GRU or LSTM), and this determines whether the model
    is "inattative" or attentive.
    """

    # Constructed inside __init__.
    classifier: nn.Linear

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.hidden_size, self.target_vocab_size)

    def beam_decode(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes with beam search.

        Decoding halts once all sequences in a batch have reached END. It is
        not currently possible to combine this with loss computation or
        teacher forcing.

        The implementation assumes batch size is 1, but both inputs and outputs
        are still assumed to have a leading dimension representing batch size.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions of shape
                B x beam_width x seq_len and log-likelihoods of shape
                B x beam_width.
        """
        # TODO: modify to work with batches larger than 1.
        batch_size = mask.size(0)
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
                        encoded, mask, symbol, cell.state
                    )
                    scores = nn.functional.log_softmax(logits.squeeze(), dim=0)
                    for new_cell in cell.extensions(state, scores):
                        beam.push(new_cell)
            beam.update()
            if beam.final:
                break
        return beam.predictions(self.device), beam.scores(self.device)

    def decode_step(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        symbol: torch.Tensor,
        state: modules.RNNState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step of the decoder.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).
            symbol (torch.Tensor): next symbol.
            state (modules.RNNState): RNN state.

        Returns:
            Tuple[torch.Tensor, modules.RNNState]: logits and the RNN state.
        """
        decoded, state = self.decoder(encoded, mask, symbol, state)
        logits = self.classifier(decoded)
        return logits, state

    @property
    def decoder_input_size(self) -> int:
        if self.has_features_encoder:
            return (
                self.source_encoder.output_size
                + self.features_encoder.output_size
            )
        else:
            return self.source_encoder.output_size

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

        Raises:
            NotImplementedError: separate features encoders are not supported.
        """
        encoded = self.source_encoder(batch.source)
        if self.has_features_encoder:
            features_encoded = self.features_encoder(batch.features)
            # Feature information is averaged across all positions, broadcast
            # across the length of the source, and then concatenated with the
            # source encoding along the encoding dimension.
            features_encoded = features_encoded.mean(dim=1, keepdim=True)
            features_encoded = features_encoded.expand(-1, encoded.size(1), -1)
            encoded = torch.cat((encoded, features_encoded), dim=2)
        if self.beam_width > 1:
            return self.beam_decode(encoded, batch.source.mask)
        else:
            return self.greedy_decode(
                encoded,
                batch.source.mask,
                self.teacher_forcing if self.training else False,
                batch.target.padded if batch.has_target else None,
            )

    @abc.abstractmethod
    def get_decoder(self) -> modules.BaseModule: ...

    def greedy_decode(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes greedily.

        A hard upper bound on the length of the decoded strings is provided by
        the length of the target strings (more specifically, the length of the
        longest target string) if a target is provided, or `max_target_length`
        if not. Decoding will halt earlier if no target is provided and all
        sequences have reached END.

        Args:
            encoded (torch.Tensor).
            mask (torch.Tensor).
            teacher_forcing (bool, optional): whether or not to decode with
                teacher forcing.
            target (torch.Tensor, optional): target symbols; if provided
                decoding continues until this length is reached.

        Returns:
            torch.Tensor: predictions of B x target_vocab_size x seq_len.
        """
        batch_size = mask.size(0)
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
            logits, state = self.decode_step(encoded, mask, symbol, state)
            predictions.append(logits.squeeze(1))
            # With teacher forcing the next input is the gold symbol for this
            # step; with student forcing, it's the top prediction.
            symbol = (
                target[:, t].unsqueeze(1)
                if teacher_forcing
                else torch.argmax(logits, dim=2)
            )
            if target is None:
                # Updates which sequences have decoded an END.
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
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))

    def get_decoder(self) -> modules.LSTMDecoder:
        return modules.LSTMDecoder(
            decoder_input_size=self.decoder_input_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    def init_state(self, batch_size: int) -> modules.RNNState:
        return modules.RNNState(
            self._init_input(batch_size),
            self._init_hiddens(batch_size),
            self._init_cell(batch_size),
        )

    def _init_cell(self, batch_size: int) -> torch.Tensor:
        return self.c0.repeat(self.decoder_layers, batch_size, 1)

    @property
    def name(self) -> str:
        return "LSTM"


class AttentiveGRUModel(GRUModel):
    """GRU model with attention."""

    def get_decoder(self) -> modules.AttentiveGRUDecoder:
        return modules.AttentiveGRUDecoder(
            attention_input_size=self.decoder_input_size,
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
        return "attentive GRU"


class AttentiveLSTMModel(LSTMModel):
    """LSTM model with attention."""

    def get_decoder(self) -> modules.AttentiveLSTMDecoder:
        return modules.AttentiveLSTMDecoder(
            attention_input_size=self.decoder_input_size,
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
        return "attentive LSTM"


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds RNN configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=defaults.BIDIRECTIONAL,
        help="Uses a bidirectional encoder (RNN-backed architectures "
        "only. Default: enabled.",
    )
    parser.add_argument(
        "--no_bidirectional",
        action="store_false",
        dest="bidirectional",
    )
