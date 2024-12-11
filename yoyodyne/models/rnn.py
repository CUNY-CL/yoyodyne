"""RNN model classes."""

import abc
import argparse
import heapq
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .. import data, defaults, special
from . import base, embeddings, modules


class RNNModel(base.BaseModel):
    """Abstract base class for RNN models.

    The implementation of `get_decoder` in the derived classes determines not
    just what kind of RNN is used (i.e., GRU or LSTM), but also whether this
    is an this is "inattentive"---in which case last (non-padding) hidden state
    of the encoder is the input to the decoder---or attentive.
    """

    # Constructed inside __init__.
    classifier: nn.Linear
    h0: nn.Parameter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.hidden_size, self.target_vocab_size)
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))

    def beam_decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes with a beam.

        Args:
            encoder_out (torch.Tensor): batch of encoded input symbols.
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols.

        Decoding halts once all sequences in a batch have reached END. It is
        not currently possible to combine this with loss computation or
        teacher forcing.

        The implementation assumes batch size is 1, but both inputs and outputs
        are assumed to have a leading dimension representing batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions of shape
                B x beam_size x seq_length and log-likelihoods of shape
                B x beam_size.
        """
        # TODO: modify to work with batches larger than 1.
        batch_size = encoder_mask.size(0)
        if batch_size != 1:
            raise NotImplementedError(
                "Beam search is not implemented for batch_size > 1"
            )
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(batch_size)
        # Log likelihood, last decoded idx, hidden state tensor.
        histories = [[0.0, [special.START_IDX], decoder_hiddens]]
        for t in range(self.max_target_length):
            # List that stores the heap of the top beam_width elements from all
            # beam_width x target_vocab_size possibilities
            likelihoods = []
            hypotheses = []
            # First accumulates all beam_width predictions.
            for (
                beam_likelihood,
                beam_idxs,
                decoder_hiddens,
            ) in histories:
                # Does not keep decoding a path that has hit END.
                if len(beam_idxs) > 1 and beam_idxs[-1] == special.END_IDX:
                    fields = [
                        beam_likelihood,
                        beam_idxs,
                        decoder_hiddens,
                    ]
                    # TODO: Replace heapq with torch.max or similar?
                    heapq.heappush(hypotheses, fields)
                    continue
                # Feeds in the first decoder input, as a start tag.
                # -> batch_size x 1
                decoder_input = torch.tensor(
                    [beam_idxs[-1]],
                    device=self.device,
                ).unsqueeze(1)
                decoded = self.decoder(
                    decoder_input,
                    decoder_hiddens,
                    encoder_out,
                    encoder_mask,
                )
                logits = self.classifier(decoded.output)
                likelihoods.append(
                    (
                        logits,
                        beam_likelihood,
                        beam_idxs,
                        decoded.hiddens,
                    )
                )
            # Constrains the next step to beamsize.
            for (
                logits,
                beam_loglikelihood,
                beam_idxs,
                decoder_hiddens,
            ) in likelihoods:
                # This is 1 x 1 x target_vocab_size since we fixed batch size
                # to 1. We squeeze off the first 2 dimensions to get a tensor
                # of target_vocab_size.
                logits = logits.squeeze((0, 1))
                # Obtain the log-probabilities of the logits.
                predictions = nn.functional.log_softmax(logits, dim=0).cpu()
                for j, logprob in enumerate(predictions):
                    if len(hypotheses) < self.beam_width:
                        fields = [
                            beam_loglikelihood + logprob,
                            beam_idxs + [j],
                            decoder_hiddens,
                        ]
                        heapq.heappush(hypotheses, fields)
                    else:
                        fields = [
                            beam_loglikelihood + logprob,
                            beam_idxs + [j],
                            decoder_hiddens,
                        ]
                        heapq.heappushpop(hypotheses, fields)
            # Sorts hypotheses and reverse to have the min log_likelihood at
            # first index. We think that this is faster than heapq.nlargest().
            hypotheses.sort(reverse=True)
            # It not necessary to make a deep copy beacuse hypotheses is going
            # to be defined again at the start of the loop.
            histories = hypotheses
            # If the top n hypotheses are full sequences, break.
            if all([h[1][-1] == special.END_IDX for h in histories]):
                break
        # Sometimes path lengths does not match so it is neccesary to pad it
        # all to same length to create a tensor.
        max_len = max(len(h[1]) for h in histories)
        # -> B x beam_size x seq_len.
        predictions = torch.tensor(
            [
                h[1] + [special.PAD_IDX] * (max_len - len(h[1]))
                for h in histories
            ],
            device=self.device,
        ).unsqueeze(0)
        # -> B x beam_size.
        scores = torch.tensor([h[0] for h in histories]).unsqueeze(0)
        return predictions, scores

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Runs the encoder-decoder model.

        Args:
            batch (data.PaddedBatch).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: beam
                search returns a tuple with a tensor of predictions of shape
                B x beam_width x seq_len and a tensor of B x shape beam_width
                with the likelihood (the unnormalized sum of sequence
                log-probabilities) for each prediction; greedy search returns
                a tensor of predictions of shape
                B x seq_len x target_vocab_size.
        """
        encoder_out = self.source_encoder(batch.source).output
        # This function has a polymorphic return because beam search needs to
        # return two tensors. For greedy, the return has not been modified to
        # match the Tuple[torch.Tensor, torch.Tensor] type because the
        # training and validation functions depend on it.
        if self.beam_width > 1:
            return self.beam_decode(encoder_out, batch.source.mask)
        else:
            return self.greedy_decode(
                encoder_out,
                batch.source.mask,
                self.teacher_forcing if self.training else False,
                batch.target.padded if batch.target else None,
            )

    @abc.abstractmethod
    def get_decoder(self) -> modules.BaseModule: ...

    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes greedily.

        A hard upper bound on the length of the decoded strings is provided by
        the length of the target strings (more specifically, the length of the
        longest target string) if a target is provided, or `max_target_length`
        if not. Decoding will halt earlier if no target is provided and all
        sequences have reached END.

        Args:
            encoder_out (torch.Tensor): batch of encoded input symbols.
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols.
            teacher_forcing (bool): whether or not to decode with teacher
                forcing.
            target (torch.Tensor, optional): target symbols; if provided this
                decoding continues until this length is reached.

        Returns:
            torch.Tensor: predictions of B x seq_length x target_vocab_size.
        """
        batch_size = encoder_mask.size(0)
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(batch_size)
        decoder_input = (
            torch.tensor([special.START_IDX], device=self.device)
            .repeat(batch_size)
            .unsqueeze(1)
        )
        predictions = []
        if target is None:
            max_num_steps = self.max_target_length
            # Tracks when each sequence has decoded an END.
            finished = torch.zeros(batch_size, device=self.device)
        else:
            max_num_steps = target.size(1)
        for t in range(max_num_steps):
            # pred: B x 1 x output_size.
            decoded = self.decoder(
                decoder_input,
                decoder_hiddens,
                encoder_out,
                encoder_mask,
            )
            decoder_output, decoder_hiddens = (
                decoded.output,
                decoded.hiddens,
            )
            logits = self.classifier(decoder_output)
            predictions.append(logits.squeeze(1))
            if teacher_forcing:
                # Under teacher forcing the next input is the gold symbol for
                # this step.
                decoder_input = target[:, t].unsqueeze(1)
            else:
                # Otherwise, under student forcing, the next input is the top
                # prediction.
                decoder_input = self._get_predicted(logits)
            if target is None:
                # Updates which sequences have decoded an END.
                finished = torch.logical_or(
                    finished, (decoder_input == special.END_IDX)
                )
                if finished.all():
                    break
        # -> B x seq_len x target_vocab_size.
        predictions = torch.stack(predictions, dim=1)
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

    @abc.abstractmethod
    def init_hiddens(
        self, batch_size: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class GRUModel(RNNModel):
    """GRU model without attention."""

    def get_decoder(self) -> modules.GRUDecoder:
        return modules.GRUDecoder(
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    def init_hiddens(self, batch_size: int) -> torch.Tensor:
        """Initializes the hidden state to pass to the RNN.

        We treat the initial value as a model parameter.

        Args:
            batch_size (int).

        Returns:
            torch.Tensor: hidden state for initialization.
        """
        return self.h0.repeat(self.decoder_layers, batch_size, 1)

    @property
    def name(self) -> str:
        return "GRU"


class LSTMModel(RNNModel):
    """LSTM model without attention.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    # This also needs an initial cell state parameter.
    c0: nn.Parameter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))

    def get_decoder(self) -> modules.LSTMDecoder:
        return modules.LSTMDecoder(
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    def init_hiddens(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state to pass to the RNN.

        We treat the initial value as a model parameter.

        Args:
            batch_size (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        return (
            self.h0.repeat(self.decoder_layers, batch_size, 1),
            self.c0.repeat(self.decoder_layers, batch_size, 1),
        )

    @property
    def name(self) -> str:
        return "LSTM"


class AttentiveGRUModel(GRUModel):
    """GRU model with attention."""

    def get_decoder(self) -> modules.AttentiveGRUDecoder:
        return modules.AttentiveGRUDecoder(
            attention_input_size=self.source_encoder.output_size,
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
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
            attention_input_size=self.source_encoder.output_size,
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
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
