"""Hard monotonic neural HMM classes."""

import argparse
from typing import Callable, Dict, Optional, Union

import numpy
import torch
from torch import nn

from .. import data, defaults
from . import lstm, modules


class HardAttentionHmm(lstm.LSTMEncoderDecoder):
    """Hard Attention Transducer.

    Learns probability distribution of target string by modeling transduction
    of source string to target string as Markov process. Assumes each character
    produced is conditioned by state transitions over each source character.

    Default model assumes independence between state and non-monotonic
    progression over source string. `enforce_monotonic` enforces monotonic
    state transition (model progresses over each source character), and
    `attention_context` allows conditioning of state transition over previous
    _n_ states. 
    
    After:
        Wu, S. and Cotterell, R. 2019. Exact hard monotonic attention for
        character-level transduction. In _Proceedings of the 57th Annual
        Meeting of the Association for Computational Linguistics_, pages
        1530-1537.

    Original implementation:
        https://github.com/shijie-wu/neural-transducer
    """

    enforce_monotonic: bool
    attention_context: int

    def __init__(
        self, *args, enforce_monotonic=False, attention_context=0, **kwargs
    ):
        """Initializes the encoder-decoder.

        Args:
            *args: passed to superclass.
            enforce_monotonic [bool, optional]: Enforce monotonic state
                transition in decoding.
            attention_context [int, optional]: Size of context window for
            conditioning state transition. If 0, state transitions are
                independent.
            **kwargs: passed to superclass.
        """
        self.enforce_monotonic = enforce_monotonic
        self.attention_context = attention_context
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(
            self.decoder.output_size, self.target_vocab_size
        )

    def init_decoding(self, encoder_out, encoder_mask):
        """Initializes hiddens and initial decoding output.

        Simply feeds a BOS string to decoder to provide initial probability.
        """
        batch_size, _ = encoder_mask.shape
        decoder_hiddens = self.init_hiddens(batch_size, self.decoder_layers)
        bos = (
            torch.tensor(
                [self.start_idx], device=self.device, dtype=torch.long
            )
            .repeat(batch_size)
            .unsqueeze(-1)
        )
        return self.decode_step(
            bos, decoder_hiddens, encoder_out, encoder_mask
        )

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder model.

        The emission tensor returned is of shape 
        (tgt_len x batch_size x src_len x vocab_size), and the
        transition tensor returned is of shape
        (tgt_len x batch_size x src_len x src_len).

        Args:
            batch (data.PaddedBatch).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tensosr of emission
                 and transition probabilities for each transition
                 state (i.e., target symbol).
        """
        encoder_out = self.source_encoder(batch.source).output
        if self.has_features_encoder:
            encoder_features_out = self.features_encoder(batch.features).output
            # Averages to flatten embedding.
            encoder_features_out = encoder_features_out.sum(
                dim=1, keepdim=True
            )
            # Expands length of source.
            encoder_features_out = encoder_features_out.expand(
                -1, encoder_out.shape[1], -1
            )
            # Concatenates with average.
            encoder_out = torch.cat(
                [encoder_out, encoder_features_out], dim=-1
            )
        if self.training:
            output = self.decode(
                encoder_out,
                batch.source.mask,
                batch.target.padded,
            )
        else:
            output = self.greedy_decode(
                encoder_out,
                batch.source.mask,
            )
        return output

    def decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        length of `target` args.

        Args:
                encoder_out (torch.Tensor): batch of encoded input symbols.
                    Shape: (batch_size x src_len x encoder_hidden*num_dir).
                encoder_mask (torch.Tensor): mask for the batch of encoded
                    input symbols. Shape: (batch_size x src_len).
                target (torch.Tensor): target symbols. Decodes up to
                    `len(target)` symbols.

        Returns:
                all_log_probs: (torch.Tensor): tensor of emission
                    probabilities for each transition state (target symb).
                    Shape: (tgt_len x batch_size x src_len x vocab_size)
                all_alignements: (torch.Tensor): tensor of transition
                    probabilities for each transition state (target symb).
                    Shape: (tgt_len x batch_size x src_len x src_len)
        """
        log_probs, alignment_matrix, decoder_hiddens = self.init_decoding(
            encoder_out, encoder_mask
        )
        all_log_probs, all_alignments = [log_probs], [alignment_matrix]
        for t in range(target.size(1)):
            tgt_symbol = target[:, t]
            log_probs, alignment_matrix, decoder_hiddens = self.decode_step(
                tgt_symbol.unsqueeze(-1),
                decoder_hiddens,
                encoder_out,
                encoder_mask,
            )
            all_log_probs.append(log_probs), all_alignments.append(
                alignment_matrix
            )
        return torch.stack(all_log_probs), torch.stack(all_alignments)

    def decode_step(
        self, tgt_symbol, decoder_hiddens, encoder_out, encoder_mask
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Performs a single decoding step for current state of decoder.

        Args:
                tgt_symbol (torch.Tensor): tgt symbol for current state.
                encoder_out (torch.Tensor): batch of encoded input symbols.
                    Shape: (batch_size x src_len x encoder_hidden*num_dir).
                encoder_mask (torch.Tensor): mask for the batch of encoded
                        input symbols of shape (batch_size x src_len).

        Returns:
                all_log_probs: (torch.Tensor): tensor of emission
                    probabilities for each transition state (target symb).
                all_alignements: (torch.Tensor): tensor of transition
                    probabilities for each transition state (target symb).
        """
        decoded = self.decoder(
            tgt_symbol,
            decoder_hiddens,
            encoder_out,
            encoder_mask,
        )
        output, alignment_matrix, decoder_hiddens = (
            decoded.output,
            decoded.embeddings,
            decoded.hiddens,
        )
        logits = self.classifier(output)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Expands matrix for all time steps.
        if self.enforce_monotonic:
            alignment_matrix = self._apply_mono_mask(alignment_matrix)
        return log_probs, alignment_matrix, decoder_hiddens

    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
                encoder_out (torch.Tensor): batch of encoded input symbols.
                    Shape: (batch_size x src_len x encoder_hidden*num_dir).
                encoder_mask (torch.Tensor): mask for the batch of encoded
                        input symbols. Shape: (batch_size x src_len).

        Returns:
                predictions (torch.Tensor): tensor of predictions of shape
                        (batch_size x pred_seq_len).
                likelihood (torch.Tensor): tensor of final likelihoods per
                    prediction step. Shape (batch_size x 1 x src_len).
        """
        batch_size, _ = encoder_mask.shape
        log_probs, alignment_matrix, decoder_hiddens = self.init_decoding(
            encoder_out, encoder_mask
        )
        predictions, likelihood = self.greedy_step(
            log_probs, alignment_matrix[:, 0].unsqueeze(1)
        )
        finished = (
            torch.zeros(batch_size, device=self.device).bool().unsqueeze(-1)
        )
        for t in range(self.max_target_length):
            symbol = predictions[:, t]
            log_probs, alignment_matrix, decoder_hiddens = self.decode_step(
                symbol.unsqueeze(-1),
                decoder_hiddens,
                encoder_out,
                encoder_mask,
            )
            likelihood = likelihood + alignment_matrix.transpose(1, 2)
            likelihood = likelihood.logsumexp(dim=-1, keepdim=True).transpose(
                1, 2
            )
            pred, likelihood = self.greedy_step(log_probs, likelihood)
            finished = finished | (pred == self.end_idx)
            if finished.all().item():
                break
            # If finished decoding, pads.
            pred = torch.where(
                ~finished, pred, torch.tensor(self.end_idx, device=self.device)
            )
            predictions = torch.cat((predictions, pred), dim=-1)
            # Updates likelihood emissions.
            likelihood = likelihood + self._gather_at_idx(log_probs, pred)
        return predictions, likelihood

    def greedy_step(self, log_probs, likelihood):
        # Updates current likelihood with greedy probability prediction.
        tgt_prob = likelihood + log_probs.transpose(1, 2)
        tgt_prob = tgt_prob.logsumexp(dim=-1)
        tgt = torch.max(tgt_prob, dim=-1)[1]
        return tgt.unsqueeze(-1), likelihood

    def _gather_at_idx(self, prob, tgt):
        # Collects probability of tgt index across all states in prob.
        batch_size, src_seq_len, vocab_size = prob.shape
        idx = tgt.view(-1, 1).expand(batch_size, src_seq_len).unsqueeze(-1)
        output = torch.gather(prob, -1, idx).view(batch_size, 1, src_seq_len)
        idx = idx.view(batch_size, 1, src_seq_len)
        pad_mask = (idx != self.pad_idx).float()
        return output * pad_mask

    def _apply_mono_mask(self, alignment):
        # Applies mask to enforce monotonic alignment.
        align_mask = torch.ones_like(alignment[0]).triu().unsqueeze(0)
        align_mask = (align_mask - 1) * -numpy.log(1e-7)
        alignment = alignment + align_mask
        alignment = alignment - alignment.logsumexp(-1, keepdim=True)
        return alignment

    def training_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
                batch (data.PaddedBatch)
                batch_idx (int).

        Returns:
                torch.Tensor: loss.
        """
        # Forward pass produces loss by default.
        log_probs, alignments = self(batch)
        loss = self.loss_func(batch.target.padded, log_probs, alignments)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        predictions, likelihood = self(batch)
        # Processes for accuracy calculation.
        predictions = self.evaluator.finalize_predictions(
            predictions, self.end_idx, self.pad_idx
        )
        val_eval_item = self.evaluator.get_eval_item(
            predictions, batch.target.padded, self.pad_idx
        )
        return {"val_eval_item": val_eval_item, "val_loss": -likelihood.mean()}

    def predict_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        predictions, _ = self(batch)
        # Processes for accuracy calculation.
        predictions = self.evaluator.finalize_predictions(
            predictions, self.end_idx, self.pad_idx
        )
        return predictions

    def get_decoder(self) -> modules.lstm.HMMLSTMDecoder:
        if self.attention_context > 0:
            return modules.lstm.ContextHMMLSTMDecoder(
                pad_idx=self.pad_idx,
                start_idx=self.start_idx,
                end_idx=self.end_idx,
                attention_context=self.attention_context,
                decoder_input_size=(
                    self.source_encoder.output_size
                    + self.features_encoder.output_size
                    if self.has_features_encoder
                    else self.source_encoder.output_size
                ),
                num_embeddings=self.target_vocab_size,
                dropout=self.dropout,
                bidirectional=False,
                embedding_size=self.embedding_size,
                layers=self.decoder_layers,
                hidden_size=self.hidden_size,
            )
        else:
            return modules.lstm.HMMLSTMDecoder(
                pad_idx=self.pad_idx,
                start_idx=self.start_idx,
                end_idx=self.end_idx,
                decoder_input_size=(
                    self.source_encoder.output_size
                    + self.features_encoder.output_size
                    if self.has_features_encoder
                    else self.source_encoder.output_size
                ),
                num_embeddings=self.target_vocab_size,
                dropout=self.dropout,
                bidirectional=False,
                embedding_size=self.embedding_size,
                layers=self.decoder_layers,
                hidden_size=self.hidden_size,
            )

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        # TODO: Make the loss update continuous during training to avoid.
        # Also enable better reflection of val loss.
        """Returns the actual function used to compute loss.

        Returns:
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                    configured loss function.
        """
        return self.loss

    def loss(self, target, log_probs, alignments):
        fwd = alignments[0, :, 0].unsqueeze(1) + self._gather_at_idx(
            log_probs[0], target[:, 0]
        )
        for t in range(1, target.shape[1]):
            fwd = fwd + alignments[t].transpose(1, 2)
            fwd = fwd.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
            fwd = fwd + self._gather_at_idx(log_probs[t], target[:, t])
        return -torch.logsumexp(fwd, dim=-1).mean() / target.shape[1]

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds HMM configuration options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--enforce_monotonic",
            action="store_true",
            default=defaults.HMM_MONOTONIC,
            help="Enforce monotonic condition "
            "(HMM architectures only). Default: %(default)s.",
        )
        parser.add_argument(
            "--hmm_context",
            type=int,
            default=defaults.HMM_CONTEXT,
            help="Width of attention context "
            "(HMM architectures only). Default: %(default)s.",
        )
