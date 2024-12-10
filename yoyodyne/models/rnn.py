"""RNN model classes."""

import argparse
import heapq
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .. import data, defaults, special
from . import base, embeddings, modules


class RNNModel(base.BaseModel):
    """Base class for RNN models.

    In lieu of attention, we concatenate the last (non-padding) hidden state of
    the encoder to the decoder hidden state.
    """

    # Constructed inside __init__.
    classifier: nn.Linear
    h0: nn.Parameter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.hidden_size, self.target_vocab_size)
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))

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

    def beam_decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Overrides `beam_decode` in `BaseEncoderDecoder`.

        This method implements the LSTM-specific beam search version. Note
        that we assume batch size is 1.
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
                    decoder_input, decoder_hiddens, encoder_out, encoder_mask
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

    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached END up to a
        specified length depending on the `target` args.

        Args:
            encoder_out (torch.Tensor): batch of encoded input symbols.
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols.
            teacher_forcing (bool): Whether or not to decode
                with teacher forcing.
            target (torch.Tensor, optional): target symbols;  we
                decode up to `len(target)` symbols. If None, we decode up to
                `self.max_target_length` symbols.

        Returns:
            torch.Tensor: tensor of predictions of shape seq_len x
                batch_size x target_vocab_size.
        """
        batch_size = encoder_mask.size(0)
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(batch_size)
        # Feed in the first decoder input, as a start tag.
        # -> B x 1.
        decoder_input = (
            torch.tensor([special.START_IDX], device=self.device)
            .repeat(batch_size)
            .unsqueeze(1)
        )
        predictions = []
        num_steps = (
            target.size(1) if target is not None else self.max_target_length
        )
        # Tracks when each sequence has decoded an END.
        finished = torch.zeros(batch_size, device=self.device)
        for t in range(num_steps):
            # pred: B x 1 x output_size.
            decoded = self.decoder(
                decoder_input, decoder_hiddens, encoder_out, encoder_mask
            )
            decoder_output, decoder_hiddens = decoded.output, decoded.hiddens
            logits = self.classifier(decoder_output)
            predictions.append(logits.squeeze(1))
            # In teacher forcing mode the next input is the gold symbol
            # for this step.
            if teacher_forcing:
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise we pass the top pred to the next timestep
            # (i.e., student forcing, greedy decoding).
            else:
                decoder_input = self._get_predicted(logits)
                # Updates to track which sequences have decoded an END.
                finished = torch.logical_or(
                    finished, (decoder_input == special.END_IDX)
                )
                # Breaks when all sequences have predicted an END symbol. If we
                # have a target (and are thus computing loss), we only break
                # when we have decoded at least the the same number of steps as
                # the target length.
                if finished.all():
                    if target is None or decoder_input.size(-1) >= target.size(
                        -1
                    ):
                        break
        # -> B x seq_len x target_vocab_size.
        predictions = torch.stack(predictions, dim=1)
        return predictions

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
                B x beam_width x seq_len and a tensor of shape beam_width with
                the likelihood (the unnormalized sum of sequence
                log-probabilities) for each prediction; greedy search returns
                a tensor of predictions of shape
                B x target_vocab_size x seq_len.
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

    @staticmethod
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

    def get_decoder(self):
        raise NotImplementedError

    def init_hiddens(self, batch_size: int):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class GRUModel(RNNModel):
    """GRU encoder-decoder without attention."""

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
    """LSTM encoder-decoder without attention.

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

    def init_hiddens(self, batch_size: int) -> torch.Tensor:
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
    """GRU encoder-decoder with attention."""

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
    """LSTM encoder-decoder with attention."""

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
