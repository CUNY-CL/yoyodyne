"""LSTM model classes."""

import argparse
import heapq
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .. import batches, defaults
from . import attention, base


class LSTMEncoderDecoder(base.BaseEncoderDecoder):
    """LSTM encoder-decoder without attention.

    We achieve this by concatenating the last (non-padding) hidden state of
    the encoder to the decoder hidden state."""

    # Model arguments.
    bidirectional: bool
    # Constructed inside __init__.
    source_embeddings: nn.Embedding
    target_embeddings: nn.Embedding
    encoder: nn.LSTM
    h0: nn.Parameter
    c0: nn.Parameter
    decoder: nn.LSTM
    classifier: nn.Linear
    log_softmax: nn.LogSoftmax

    def __init__(
        self,
        *args,
        bidirectional=defaults.BIDIRECTIONAL,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            *args: passed to superclass.
            bidirectional (bool).
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.source_embeddings = self.init_embeddings(
            self.vocab_size, self.embedding_size, self.pad_idx
        )
        self.target_embeddings = self.init_embeddings(
            self.output_size, self.embedding_size, self.pad_idx
        )
        self.encoder = nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.encoder_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        encoder_size = self.hidden_size * self.num_directions
        # Initial hidden state whose parameters are shared across all examples.
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))
        self.decoder = nn.LSTM(
            encoder_size + self.embedding_size,
            self.hidden_size,
            dropout=self.dropout,
            num_layers=self.decoder_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(self.hidden_size, self.output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    @property
    def num_directions(self):
        return 2 if self.bidirectional else 1

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        return self._normal_embedding_initialization(
            num_embeddings, embedding_size, pad_idx
        )

    def encode(
        self, source: batches.PaddedTensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes the input.

        Args:
            source (batches.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                encoded timesteps, and the LSTM h0 and c0 cells.
        """
        embedded = self.source_embeddings(source.padded)
        embedded = self.dropout_layer(embedded)
        # Packs embedded source symbols into a PackedSequence.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source.lengths(), batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim, (h0, c0).
        packed_outs, (H, C) = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        return encoded, (H, C)

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes one step.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (torch.Tensor): last hidden states from the decoder
                of shape (1 x B x decoder_dim, 1 x B x decoder_dim).
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of scores and hiddens.
        """
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h0, last_c0 = last_hiddens
        # Get the index of the last unmasked tensor.
        # -> B.
        last_encoder_out_idxs = (~encoder_mask).sum(dim=1) - 1
        # -> B x 1 x 1.
        last_encoder_out_idxs = last_encoder_out_idxs.view(
            [encoder_out.size(0)] + [1, 1]
        )
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_encoder_out_idxs = last_encoder_out_idxs.expand(
            [-1, -1, encoder_out.size(-1)]
        )
        # -> B x 1 x encoder_dim.
        last_encoder_out = torch.gather(encoder_out, 1, last_encoder_out_idxs)
        # The input to decoder LSTM is the embedding concatenated to the
        # weighted, encoded, inputs.
        output, hiddens = self.decoder(
            torch.cat((embedded, last_encoder_out), 2), (last_h0, last_c0)
        )
        output = self.dropout_layer(output)
        # Classifies into output vocab.
        # -> B x 1 x output_size.
        output = self.classifier(output)
        # Computes log_softmax scores for NLLLoss.
        output = self.log_softmax(output)
        return output, hiddens

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
        batch_size: int,
        encoder_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        This initializes an <EOS> tensor, and decodes using teacher forcing
        when training, or else greedily.

        Args:
            batch_size (int).
            encoder_mask (torch.Tensor): mask for the batch of encoded inputs.
            encoder_out (torch.Tensor): batch of encoded inputs.
            target (torch.Tensor, optional): target symbols; if None, then we
                decode greedily with 'student forcing'.

        Returns:
            predictions (torch.Tensor): tensor of predictions of shape
                seq_len x batch_size x output_size.
        """
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
            output, decoder_hiddens = self.decode_step(
                decoder_input, decoder_hiddens, encoder_out, encoder_mask
            )
            predictions.append(output.squeeze(1))
            # If we have a target (training) then the next input is the gold
            # symbol for this step (i.e., teacher forcing).
            if target is not None:
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise, it must be inference time and we pass the top pred to
            # the next next timestep (i.e., student forcing, greedy decoding).
            else:
                decoder_input = self._get_predicted(output)
                # Updates to track which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all batches predicted an EOS symbol.
                if finished.all():
                    break
        predictions = torch.stack(predictions)
        return predictions

    def beam_decode(
        self,
        batch_size: int,
        encoder_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        beam_width: int,
        n: int = 1,
        return_confidences: bool = False,
    ) -> Union[Tuple[List, List], List]:
        """Beam search with beam_width.

        Note that we assume batch size is 1.

        Args:
            batch_size (int).
            encoder_mask (torch.Tensor).
            encoder_out (torch.Tensor): encoded inputs.
            beam_width (int): size of the beam.
            n (int): number of hypotheses to return.
            return_confidences (bool, optional): additionally return the
                likelihood of each hypothesis.

        Returns:
            Union[Tuple[List, List], List]: _description_
        """
        # TODO: only implemented for batch size of 1. Implement batch mode.
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
            # beam_width x output_size possibilities
            likelihoods = []
            hypotheses = []
            # First accumulates all beam_width softmaxes.
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
                predictions, decoder_hiddens = self.decode_step(
                    decoder_input, decoder_hiddens, encoder_out, encoder_mask
                )
                likelihoods.append(
                    (
                        predictions,
                        beam_likelihood,
                        beam_idxs,
                        char_likelihoods,
                        decoder_hiddens,
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
                # This is 1 x 1 x output_size since we fixed batch size to 1.
                # We squeeze off the fist 2 dimensions to get a tensor of
                # output_size.
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
        # Converts shape to that of `decode`: seq_len x B x output_size.
        predictions = predictions.unsqueeze(0).transpose(0, 2)
        if return_confidences:
            return (predictions, torch.tensor([h[2] for h in histories]))
        else:
            return predictions

    def forward(self, batch: batches.PaddedBatch) -> torch.Tensor:
        """Runs the encoder-decoder model.

        Args:
            batch (batches.PaddedBatch).

        Returns:
            predictions (torch.Tensor): tensor of predictions of shape
                (seq_len, batch_size, output_size).
        """
        encoder_out, _ = self.encode(batch.source)
        if self.beam_width is not None and self.beam_width > 1:
            predictions = self.beam_decode(
                len(batch),
                batch.source.mask,
                encoder_out,
                beam_width=self.beam_width,
            )
        else:
            predictions = self.decode(
                len(batch), batch.source.mask, encoder_out, batch.target.padded
            )
        # -> B x seq_len x output_size.
        predictions = predictions.transpose(0, 1)
        return predictions

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

    attention: attention.Attention

    def __init__(self, *args, **kwargs):
        """Initializes the encoder-decoder with attention."""
        super().__init__(*args, **kwargs)
        encoder_size = self.hidden_size * self.num_directions
        self.attention = attention.Attention(encoder_size, self.hidden_size)

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes one step.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): last hidden
                states from the decoder of shape
                (1 x B x decoder_dim, 1 x B x decoder_dim).
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: softmax scores over all outputs,
                and the previous hidden states from the decoder LSTM.
        """
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h0, last_c0 = last_hiddens
        context, _ = self.attention(
            last_h0.transpose(0, 1), encoder_out, encoder_mask
        )
        output, hiddens = self.decoder(
            torch.cat((embedded, context), 2), (last_h0, last_c0)
        )
        output = self.dropout_layer(output)
        # Classifies into output vocab.
        # -> B x 1 x output_size
        output = self.classifier(output)
        output = self.log_softmax(output)
        return output, hiddens
