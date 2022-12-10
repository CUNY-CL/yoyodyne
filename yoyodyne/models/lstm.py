"""LSTM model classes."""

import heapq
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .. import evaluators
from . import attention, base


class Error(Exception):
    pass


class LSTMEncoderDecoder(base.BaseEncoderDecoder):
    """LSTM encoder-decoder without attention.

    We achieve this by concatenating the last (non-padding) hidden state of
    the encoder to the decoder hidden state."""

    vocab_size: int
    enc_layers: int
    dec_layers: int
    pad_idx: int
    optim_name: str
    beta1: float
    beta2: float
    warmup_steps: int
    lr: int
    evaluator: evaluators.Evaluator
    scheduler: str
    start_idx: int
    end_idx: int
    embedding_size: int
    bidirectional: bool
    num_directions: int
    dropout: float
    dropout_layer: nn.Dropout
    hidden_size: int
    output_size: int
    source_embeddings: nn.Embedding
    target_embeddings: nn.Embedding
    encoder: nn.LSTM
    h0: nn.Parameter
    c0: nn.Parameter
    max_decode_len: int
    decoder: nn.LSTM
    classifier: nn.Linear
    log_softmax: nn.LogSoftmax
    label_smoothing: Optional[float]
    beam_width: Optional[int]

    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        output_size,
        pad_idx,
        start_idx,
        end_idx,
        optim,
        beta1,
        beta2,
        warmup_steps,
        lr,
        scheduler,
        evaluator,
        max_decode_len,
        dropout=0.1,
        enc_layers=2,
        dec_layers=2,
        bidirectional=True,
        label_smoothing=None,
        beam_width=None,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            vocab_size (int).
            embedding_size (int).
            hidden_size (int).
            output_size (int).
            pad_idx (int).
            start_idx (int).
            end_idx (int).
            optim (str).
            beta1 (float).
            beta2 (float).
            warmup_steps (int).
            lr (float).
            evaluator (evaluators.Evaluator).
            scheduler (str).
            max_decode_len (int).
            dropout (float, optional).
            enc_layers (int, optional).
            dec_layers (int, optional).
            bidirectional (bool, optional).
            label_smoothing (float, optional).
            beam_width (int, optional): if specified, beam search is used
                during decoding.
            **kwargs: ignored.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.pad_idx = pad_idx
        self.optim_name = optim
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.source_embeddings = self.init_embeddings(
            self.vocab_size, self.embedding_size, self.pad_idx
        )
        self.target_embeddings = self.init_embeddings(
            self.output_size, self.embedding_size, self.pad_idx
        )
        self.encoder = nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers=enc_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        enc_size = self.hidden_size * self.num_directions
        # Initial hidden state whose parameters are shared across all examples.
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))
        self.max_decode_len = max_decode_len
        self.decoder = nn.LSTM(
            enc_size + self.embedding_size,
            self.hidden_size,
            dropout=self.dropout,
            num_layers=self.dec_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(self.hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.label_smoothing = label_smoothing
        self.beam_width = beam_width
        self.loss_func = self.get_loss_func("mean")
        # Saves hyperparameters for PL checkpointing.
        self.save_hyperparameters()

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
        self, source: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes a batch of inputs.

        Args:
            source (torch.Tensor): input symbols of shape B x seq_len x 1
            mask (torch.Tensor): mask for the input symbols of shape
                B x seq_len x 1

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                encoded timesteps, and the LSTM h0 and c0 cells.
        """
        embedded = self.source_embeddings(source)
        embedded = self.dropout_layer(embedded)
        # Packs embedded source symbols into a PackedSequence.
        lens = (mask == 0).sum(dim=1).to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lens, batch_first=True, enforce_sorted=False
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
        enc_out: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes one step.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (torch.Tensor): last hidden states from the decoder
                of shape (1 x B x decoder_dim, 1 x B x decoder_dim).
            enc_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            enc_mask (torch.Tensor): mask for the encoded input batch of shape
                B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of scores and hiddens.
        """
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h0, last_c0 = last_hiddens
        # Get the index of the last unmasked tensor.
        # -> B.
        last_enc_out_idxs = (~enc_mask).sum(dim=1) - 1
        # -> B x 1 x 1.
        last_enc_out_idxs = last_enc_out_idxs.view([enc_out.size(0)] + [1, 1])
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_enc_out_idxs = last_enc_out_idxs.expand(
            [-1, -1, enc_out.size(-1)]
        )
        # -> B x 1 x encoder_dim.
        last_enc_out = torch.gather(enc_out, 1, last_enc_out_idxs)
        # The input to decoder LSTM is the embedding concatenated to the
        # weighted, encoded, inputs.
        output, hiddens = self.decoder(
            torch.cat((embedded, last_enc_out), 2), (last_h0, last_c0)
        )
        output = self.dropout_layer(output)
        # Classifies into output vocab.
        # -> B x 1 x output_size.
        output = self.classifier(output)
        # Computes log_softmax scores for NLLLoss.
        output = self.log_softmax(output)
        return output, hiddens

    def init_hiddens(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state to pass to the LSTM.

        Note that we learn the initial state h0 as a parameter of the model.

        Args:
            batch_size (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hidden cells for LSTM
                initialization.
        """
        return (
            self.h0.repeat(self.enc_layers, batch_size, 1),
            self.c0.repeat(self.enc_layers, batch_size, 1),
        )

    def decode(
        self,
        batch_size: int,
        enc_mask: torch.Tensor,
        enc_out: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        This initializes an <EOS> tensor, and decodes using teacher forcing
        when training, or else greedily.

        Args:
            batch_size (int).
            enc_mask (torch.Tensor): mask for the batch of encoded inputs.
            enc_out (torch.Tensor): batch of encoded inputs.
            target (torch.Tensor, optional): target symbols. If None, then we
                decode greedily with 'student forcing'.

        Returns:
            preds (torch.Tensor): tensor of predictions of shape
                (sequence_length, batch_size, output_size)
        """
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(batch_size)
        # Feed in the first decoder input, as a start tag.
        # -> B x 1.
        decoder_input = (
            torch.LongTensor([self.start_idx])
            .to(self.device)
            .repeat(batch_size)
            .unsqueeze(1)
        )
        preds = []
        num_steps = (
            target.size(1) if target is not None else self.max_decode_len
        )
        # Tracks when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size).to(self.device)
        for t in range(num_steps):
            # pred: B x 1 x output_size.
            output, decoder_hiddens = self.decode_step(
                decoder_input, decoder_hiddens, enc_out, enc_mask
            )
            preds.append(output.squeeze(1))
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
        preds = torch.stack(preds)
        return preds

    def beam_decode(
        self,
        batch_size: int,
        enc_mask: torch.Tensor,
        enc_out: torch.Tensor,
        beam_width: int,
        n: int = 1,
        return_confidences: bool = False,
    ) -> Union[Tuple[List, List], List]:
        """Beam search with beam_width.

        Note that we assume batch size is 1.

        Args:
            batch_size (int).
            enc_mask (torch.Tensor).
            enc_out (torch.Tensor): encoded inputs.
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
        decoder_hiddens = self.init_hiddens(enc_out.size(0))
        # log likelihood, last decoded idx, all likelihoods,  hiddens tensor.
        histories = [[0.0, [self.start_idx], [0.0], decoder_hiddens]]
        for t in range(self.max_decode_len):
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
                decoder_input = (
                    torch.LongTensor([beam_idxs[-1]])
                    .to(self.device)
                    .unsqueeze(1)
                )
                preds, decoder_hiddens = self.decode_step(
                    decoder_input, decoder_hiddens, enc_out, enc_mask
                )
                likelihoods.append(
                    (
                        preds,
                        beam_likelihood,
                        beam_idxs,
                        char_likelihoods,
                        decoder_hiddens,
                    )
                )
            # Constrains the next step to beamsize.
            for (
                preds,
                beam_likelihood,
                beam_idxs,
                char_likelihoods,
                decoder_hiddens,
            ) in likelihoods:
                # -> B x seq_len x outputs.
                # This is 1 x 1 x outputs since we fixed batch size to 1.
                # We squeeze off the fist 2 dimensions to get a tensor of
                # output_size.
                preds = preds.squeeze(0).squeeze(0)
                for j, prob in enumerate(preds):
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
        preds = torch.tensor([h[1] for h in histories])
        # Converts shape to that of `decode`: seq_len x B x output_size.
        preds = preds.unsqueeze(0).transpose(0, 2)
        if return_confidences:
            return (preds, torch.tensor([h[2] for h in histories]))
        else:
            return preds

    def forward(self, batch: base.Batch) -> torch.Tensor:
        """Runs the encoder-decoder model.

        Args:
            batch (base.Batch): batch of the input, input mask, output, and
                output mask.

        Returns:
            preds (torch.Tensor): tensor of predictions of shape
                (sequence_length, batch_size, output_size).
        """
        if len(batch) == 4:
            source, source_mask, target, target_mask = batch
        elif len(batch) == 2:
            source, source_mask = batch
            target = None
        else:
            raise Error(f"Batch of {len(batch)} elements is invalid")
        batch_size = source.size(0)
        enc_out, _ = self.encode(source, source_mask)
        if self.beam_width is not None and self.beam_width > 1:
            preds = self.beam_decode(
                batch_size, source_mask, enc_out, beam_width=self.beam_width
            )
        else:
            preds = self.decode(batch_size, source_mask, enc_out, target)
        # -> B x output_size x seq_len.
        preds = preds.transpose(0, 1).transpose(1, 2)
        return preds


class LSTMEncoderDecoderAttention(LSTMEncoderDecoder):
    """LSTM encoder-decoder with attention."""

    attention: attention.Attention

    def __init__(self, *args, **kwargs):
        """Initializes the encoder-decoder with attention."""
        super().__init__(*args, **kwargs)
        enc_size = self.hidden_size * self.num_directions
        self.attention = attention.Attention(enc_size, self.hidden_size)

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        enc_out: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes one step.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): last hidden
                states from the decoder of shape
                (1 x B x decoder_dim, 1 x B x decoder_dim).
            enc_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            enc_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: softmax scores over all outputs,
                and the previous hidden states from the decoder LSTM.
        """
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h0, last_c0 = last_hiddens
        context, _ = self.attention(last_h0.transpose(0, 1), enc_out, enc_mask)
        output, hiddens = self.decoder(
            torch.cat((embedded, context), 2), (last_h0, last_c0)
        )
        output = self.dropout_layer(output)
        # Classifies into output vocab.
        # -> B x 1 x output_size
        output = self.classifier(output)
        output = self.log_softmax(output)
        return output, hiddens
