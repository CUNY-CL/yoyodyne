"""LSTM model classes."""

import argparse
import heapq
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .. import data, defaults
from . import base, modules


class LSTMEncoderDecoder(base.BaseEncoderDecoder):
    """LSTM encoder-decoder without attention.

    # TODO: Evaluate if this blurb is still correct.
    We achieve this by concatenating the last (non-padding) hidden state of
    the encoder to the decoder hidden state."""

    # Constructed inside __init__.
    h0: nn.Parameter
    c0: nn.Parameter

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            *args: passed to superclass.
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        # Initial hidden state whose parameters are shared across all examples.
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))
        self.classifier = nn.Linear(self.hidden_size, self.target_vocab_size)

    def get_decoder(self) -> modules.lstm.LSTMDecoder:
        return modules.lstm.LSTMDecoder(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            decoder_input_size=self.source_encoder.output_size,
            num_embeddings=self.target_vocab_size,
            dropout=self.dropout,
            bidirectional=False,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
        )

    def init_hiddens(
        self, batch_size: int, num_layers: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state to pass to the LSTM.

        Note that we learn the initial state h0 as a parameter of the model.

        Args:
            batch_size (int).
            num_layers (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hidden cells for LSTM
                initialization.
        """
        return (
            self.h0.repeat(num_layers, batch_size, 1),
            self.c0.repeat(num_layers, batch_size, 1),
        )

    def decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
            encoder_out (torch.Tensor): batch of encoded input symbols.
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols.
            teacher_forcing (bool): Whether or not to decode
                with teacher forcing.
            target (torch.Tensor, optional): target symbols;  we
                decode up to `len(target)` symbols. If it is None, then we
                decode up to `self.max_target_length` symbols.

        Returns:
            predictions (torch.Tensor): tensor of predictions of shape
                seq_len x batch_size x target_vocab_size.
        """
        batch_size = encoder_mask.shape[0]
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(batch_size, self.decoder_layers)
        # Feed in the first decoder input, as a start tag.
        # -> B x 1.
        decoder_input = (
            torch.tensor(
                [self.start_idx], device=self.device, dtype=torch.long
            )
            .repeat(batch_size)
            .unsqueeze(1)
        )
        predictions = []
        num_steps = (
            target.size(1) if target is not None else self.max_target_length
        )
        # Tracks when each sequence has decoded an EOS.
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
                # Updates to track which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all sequences have predicted an EOS symbol. If we
                # have a target (and are thus computing loss), we only break
                # when we have decoded at least the the same number of steps as
                # the target length.
                if finished.all():
                    if target is None or decoder_input.size(-1) >= target.size(
                        -1
                    ):
                        break
        predictions = torch.stack(predictions)
        return predictions

    def beam_decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        beam_width: int,
        n: int = 1,
        return_confidences: bool = False,
    ) -> Union[Tuple[List, List], List]:
        """Beam search with beam_width.

        Note that we assume batch size is 1.

        Args:
            encoder_out (torch.Tensor): encoded inputs.
            encoder_mask (torch.Tensor).
            beam_width (int): size of the beam.
            n (int): number of hypotheses to return.
            return_confidences (bool, optional): additionally return the
                likelihood of each hypothesis.

        Returns:
            Union[Tuple[List, List], List]: _description_
        """
        # TODO: only implemented for batch size of 1. Implement batch mode.
        batch_size = encoder_mask.shape[0]
        if batch_size != 1:
            raise NotImplementedError(
                "Beam search is not implemented for batch_size > 1"
            )
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(
            encoder_out.size(0), self.decoder_layers
        )
        # log likelihood, last decoded idx, all likelihoods,  hiddens tensor.
        histories = [[0.0, [self.start_idx], [0.0], decoder_hiddens]]
        for t in range(self.max_target_length):
            # List that stores the heap of the top beam_width elements from all
            # beam_width x target_vocab_size possibilities
            likelihoods = []
            hypotheses = []
            # First accumulates all beam_width predictions.
            for (
                beam_likelihood,
                beam_idxs,
                char_likelihoods,
                decoder_hiddens,
            ) in histories:
                # Does not keep decoding a path that has hit EOS.
                if len(beam_idxs) > 1 and beam_idxs[-1] == self.end_idx:
                    fields = [
                        beam_likelihood,
                        beam_idxs,
                        char_likelihoods,
                        decoder_hiddens,
                    ]
                    # TODO: Beam search with beam_width.
                    # TODO: Replace heapq with torch.max or similar?
                    heapq.heappush(hypotheses, fields)
                    continue
                # Feeds in the first decoder input, as a start tag.
                # -> batch_size x 1
                decoder_input = torch.tensor(
                    [beam_idxs[-1]], device=self.device, dtype=torch.long
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
                        char_likelihoods,
                        decoded.hiddens,
                    )
                )
            # Constrains the next step to beamsize.
            for (
                predictions,
                beam_likelihood,
                beam_idxs,
                char_likelihoods,
                decoder_hiddens,
            ) in likelihoods:
                # This is 1 x 1 x target_vocab_size since we fixed batch size
                # to 1. We squeeze off the first 2 dimensions to get a tensor
                # of target_vocab_size.
                predictions = predictions.squeeze(0).squeeze(0)
                for j, prob in enumerate(predictions):
                    if return_confidences:
                        cl = char_likelihoods + [prob]
                    else:
                        cl = char_likelihoods
                    if len(hypotheses) < beam_width:
                        fields = [
                            beam_likelihood + prob,
                            beam_idxs + [j],
                            cl,
                            decoder_hiddens,
                        ]
                        heapq.heappush(hypotheses, fields)
                    else:
                        fields = [
                            beam_likelihood + prob,
                            beam_idxs + [j],
                            cl,
                            decoder_hiddens,
                        ]
                        heapq.heappushpop(hypotheses, fields)
            # Takes the top beam hypotheses from the heap.
            histories = heapq.nlargest(beam_width, hypotheses)
            # If the top n hypotheses are full sequences, break.
            if all([h[1][-1] == self.end_idx for h in histories]):
                break
        # Returns the top-n hypotheses.
        histories = heapq.nlargest(n, hypotheses)
        predictions = torch.tensor([h[1] for h in histories], self.device)
        # Converts shape to that of `decode`: seq_len x B x target_vocab_size.
        predictions = predictions.unsqueeze(0).transpose(0, 2)
        if return_confidences:
            return (predictions, torch.tensor([h[2] for h in histories]))
        else:
            return predictions

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder model.

        Args:
            batch (data.PaddedBatch).

        Returns:
            predictions (torch.Tensor): tensor of predictions of shape
                (seq_len, batch_size, target_vocab_size).
        """
        encoder_out = self.source_encoder(batch.source).output
        if self.beam_width is not None and self.beam_width > 1:
            predictions = self.beam_decode(
                encoder_out,
                batch.source.mask,
                beam_width=self.beam_width,
            )
        else:
            predictions = self.decode(
                encoder_out,
                batch.source.mask,
                self.teacher_forcing if self.training else False,
                batch.target.padded if batch.target else None,
            )
        # -> B x seq_len x target_vocab_size.
        predictions = predictions.transpose(0, 1)
        return predictions

    @property
    def name(self) -> str:
        return "LSTM"

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds LSTM configuration options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--bidirectional",
            action="store_true",
            default=defaults.BIDIRECTIONAL,
            help="Uses a bidirectional encoder "
            "(LSTM-backed architectures only. Default: %(default)s.",
        )
        parser.add_argument(
            "--no_bidirectional",
            action="store_false",
            dest="bidirectional",
        )


class AttentiveLSTMEncoderDecoder(LSTMEncoderDecoder):
    """LSTM encoder-decoder with attention."""

    def get_decoder(self):
        return modules.lstm.LSTMAttentiveDecoder(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            decoder_input_size=self.source_encoder.output_size,
            num_embeddings=self.target_vocab_size,
            dropout=self.dropout,
            bidirectional=False,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
            attention_input_size=self.source_encoder.output_size,
        )

    @property
    def name(self) -> str:
        return "attentive LSTM"
