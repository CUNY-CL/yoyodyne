"""Hard monotonic neural HMM classes."""

import argparse
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn

from .. import data, defaults
from . import lstm, modules


class HardAttentionLSTM(lstm.LSTMEncoderDecoder):
    """Hard attention transducer.

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
        self,
        *args,
        enforce_monotonic=defaults.ENFORCE_MONOTONIC,
        attention_context=defaults.ATTENTION_CONTEXT,
        **kwargs,
    ):
        """Initializes the encoder-decoder.

        Args:
            *args: passed to superclass.
            enforce_monotonic (bool, optional): enforces monotonic state
                transition in decoding.
            attention_context (int, optional): size of context window for
            conditioning state transition; if 0, state transitions are
                independent.
            **kwargs: passed to superclass.
        """
        self.enforce_monotonic = enforce_monotonic
        self.attention_context = attention_context
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(
            self.decoder.output_size, self.target_vocab_size
        )
        assert (
            self.teacher_forcing
        ), "Teacher forcing disabled but required by this model"

    def init_decoding(
        self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initializes hiddens and initial decoding output.

        This feeds the BOS string to the decoder to provide an initial
        probability.

        Args:
            encoder_out (torch.Tensor).
            encoder_mask (torch.Tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor,
                torch.Tensor]].
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the encoder-decoder model.

        Args:
            batch (data.PaddedBatch).

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: emission probabilities for
                each transition state of shape tgt_len x batch_size x src_len
                x vocab_size, and transition probabilities for each transition
                state of shape batch_size x src_len x src_len.
        """
        encoder_out = self.source_encoder(batch.source).output
        if self.has_features_encoder:
            encoder_features_out = self.features_encoder(batch.features).output
            # Averages to flatten embedding.
            encoder_features_out = encoder_features_out.sum(
                dim=1, keepdim=True
            )
            # Sums to flatten embedding; this is done as an alternative to the
            # linear projection used in the original paper.
            encoder_features_out = encoder_features_out.expand(
                -1, encoder_out.shape[1], -1
            )
            # Concatenates with the average.
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        length of `target` args.

        Args:
            encoder_out (torch.Tensor): batch of encoded input symbols
                of shape batch_size x src_len x (encoder_hidden *
                num_directions).
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols of shape batch_size x src_len.
            target (torch.Tensor): target symbols, decodes up to
                `len(target)` symbols.

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: emission probabilities
                for each state (target symbol) of shape tgt_len x
                batch_size x src_len x vocab_size, and transition
                probabilities for each state (target symbol) of shape
                tgt_len x batch_size x src_len x src_len.
        """
        log_probs, transition_probs, decoder_hiddens = self.init_decoding(
            encoder_out, encoder_mask
        )
        all_log_probs = [log_probs]
        all_transition_probs = [transition_probs]
        for tgt_char_idx in range(target.size(1)):
            tgt_symbol = target[:, tgt_char_idx]
            log_probs, transition_probs, decoder_hiddens = self.decode_step(
                tgt_symbol.unsqueeze(-1),
                decoder_hiddens,
                encoder_out,
                encoder_mask,
            )
            all_log_probs.append(log_probs)
            all_transition_probs.append(transition_probs)
        return torch.stack(all_log_probs), torch.stack(all_transition_probs)

    def decode_step(
        self,
        tgt_symbol: torch.Tensor,
        decoder_hiddens: Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Performs a single decoding step for current state of decoder.

        Args:
            tgt_symbol (torch.Tensor): tgt symbol for current state.
            decoder_hiddens (Tuple[torch.Tensor, torch.Tensor]): the last
                hidden states from the decoder of shapes 1 x B x decoder_dim
                and 1 x B x decoder_dim.
            encoder_out (torch.Tensor): batch of encoded input symbols
                of shape batch_size x src_len x (encoder_hidden *
                num_directions).
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols of shape batch_size x src_len.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor,
                torch.Tensor]: emission probabilities for each transition
                state (target symbol), transition probabilities for each
                transition state (target symbol), and the last hidden states
                from the decoder, of shape (1 x B x decoder_dim, 1 x B x
                decoder_dim).
        """
        decoded = self.decoder(
            tgt_symbol,
            decoder_hiddens,
            encoder_out,
            encoder_mask,
        )
        output, transition_prob, decoder_hiddens = (
            decoded.output,
            decoded.embeddings,
            decoded.hiddens,
        )
        logits = self.classifier(output)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Expands matrix for all time steps.
        if self.enforce_monotonic:
            transition_prob = self._apply_mono_mask(transition_prob)
        return log_probs, transition_prob, decoder_hiddens

    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
            encoder_out (torch.Tensor): batch of encoded input symbols
                of shape batch_size x src_len x (encoder_hidden *
                num_directions).
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols of shape batch_size x src_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions of shape batch_size
                x pred_seq_len, final likelihoods per prediction step of shape
                batch_size x 1 x src_len.
        """
        batch_size, _ = encoder_mask.shape
        log_probs, transition_prob, decoder_hiddens = self.init_decoding(
            encoder_out, encoder_mask
        )
        predictions, likelihood = self.greedy_step(
            log_probs, transition_prob[:, 0].unsqueeze(1)
        )
        finished = (
            torch.zeros(batch_size, device=self.device).bool().unsqueeze(-1)
        )
        for tgt_char_idx in range(self.max_target_length):
            symbol = predictions[:, tgt_char_idx]
            log_probs, transition_prob, decoder_hiddens = self.decode_step(
                symbol.unsqueeze(-1),
                decoder_hiddens,
                encoder_out,
                encoder_mask,
            )
            likelihood = likelihood + transition_prob.transpose(1, 2)
            likelihood = likelihood.logsumexp(dim=-1, keepdim=True).transpose(
                1, 2
            )
            pred, likelihood = self.greedy_step(log_probs, likelihood)
            finished = finished | (pred == self.end_idx)
            if finished.all().item():
                break
            # Pads if finished decoding.
            pred = torch.where(
                ~finished, pred, torch.tensor(self.end_idx, device=self.device)
            )
            predictions = torch.cat((predictions, pred), dim=-1)
            # Updates likelihood emissions.
            likelihood = likelihood + self._gather_at_idx(
                log_probs, pred, self.pad_idx
            )
        return predictions, likelihood

    @staticmethod
    def greedy_step(
        log_probs: torch.Tensor, likelihood: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding of current timestep.

        Args:
            log_probs (torch.Tensor): vocabulary probabilities at current
                time step.
            likelihood (torch.Tensor): accumulative likelihood of decoded
                character sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: greedily decoded character
                for current timestep and the current likelihood of the
                decoded character sequence.
        """
        tgt_prob = likelihood + log_probs.transpose(1, 2)
        tgt_prob = tgt_prob.logsumexp(dim=-1)
        tgt_char = torch.max(tgt_prob, dim=-1)[1]
        return tgt_char.unsqueeze(-1), likelihood

    @staticmethod
    def _gather_at_idx(
        prob: torch.Tensor, tgt: torch.Tensor, pad_idx=None
    ) -> torch.Tensor:
        """Collects probability of tgt index across all states in prob.

        To calculate the final emission probability, the pseudo-HMM
        graph needs to aggregate the final emission probabilities of
        tgt char across all potential hidden states in prob.

        Args:
            prob (torch.Tensor): log probabilities of emission states of shape
                batch_size x src_len x vocab_size.
            tgt (torch.Tensor): tgt symbol to poll probabilities for
                shape batch_size.

        Returns:
            torch.Tensor: emission probabilities of tgt symbol for each hidden
                state of size batch_size 1 x src_len.
        """
        batch_size, src_seq_len, _ = prob.shape
        idx = tgt.view(-1, 1).expand(batch_size, src_seq_len).unsqueeze(-1)
        output = torch.gather(prob, -1, idx).view(batch_size, 1, src_seq_len)
        idx = idx.view(batch_size, 1, src_seq_len)
        if pad_idx:
            pad_mask = (idx != pad_idx).float()
        return output * pad_mask

    @staticmethod
    def _apply_mono_mask(transition_prob: torch.Tensor) -> torch.Tensor:
        """Applies monotonic attention mask to transition probabilities.

        Enforces a 0 log-probability values for all non-monotonic relations
        in the transition_prob tensor (i.e., all values i < j per row j).

        Args:
            transition_prob (torch.Tensor): transition probabilities between
                all hidden states (source sequence) of shape batch_size x
                src_len x src_len.

        Returns:
            torch.Tensor: masked transition probabilities of shape batch_size
                x src_len x src_len.
        """
        mask = torch.ones_like(transition_prob[0]).triu().unsqueeze(0)
        # Using 0 log-probability value for masking; this is borrowed from the
        # original implementation.
        mask = (mask - 1) * defaults.NEG_LOG_EPSILON
        transition_prob = transition_prob + mask
        transition_prob = transition_prob - transition_prob.logsumexp(
            -1, keepdim=True
        )
        return transition_prob

    def training_step(
        self, batch: data.PaddedBatch, batch_idx: int
    ) -> torch.Tensor:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        # Forward pass produces loss by default.
        log_probs, transition_probs = self(batch)
        loss = self.loss_func(batch.target.padded, log_probs, transition_probs)
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
        val_eval_items_dict = {}
        for evaluator in self.evaluators:
            predictions = evaluator.finalize_predictions(
                predictions, self.end_idx, self.pad_idx
            )
            golds = evaluator.finalize_golds(
                batch.target.padded, self.end_idx, self.pad_idx
            )
            val_eval_items_dict[evaluator.name] = evaluator.get_eval_item(
                predictions, golds, self.pad_idx
            )
        val_eval_items_dict.update({"val_loss": -likelihood.mean()})
        return val_eval_items_dict

    def predict_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        predictions, _ = self(batch)
        return predictions

    def get_decoder(self) -> modules.lstm.HardAttentionLSTMDecoder:
        if self.attention_context > 0:
            return modules.lstm.ContextHardAttentionLSTMDecoder(
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
            return modules.lstm.HardAttentionLSTMDecoder(
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
        """Returns the actual function used to compute loss.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        return self._loss

    def _loss(
        self,
        target: torch.Tensor,
        log_probs: torch.Tensor,
        transition_probs: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: Currently we're storing a concatenation of loss tensors for
        # each time step. This is costly. Revisit this calculation and see if
        # we can use DP to simplify.
        fwd = transition_probs[0, :, 0].unsqueeze(1) + self._gather_at_idx(
            log_probs[0], target[:, 0], self.pad_idx
        )
        for tgt_char_idx in range(1, target.shape[1]):
            fwd = fwd + transition_probs[tgt_char_idx].transpose(1, 2)
            fwd = fwd.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
            fwd = fwd + self._gather_at_idx(
                log_probs[tgt_char_idx], target[:, tgt_char_idx], self.pad_idx
            )
        loss = -torch.logsumexp(fwd, dim=-1).mean() / target.shape[1]
        return loss

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds HMM configuration options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--enforce_monotonic",
            action="store_true",
            default=defaults.ENFORCE_MONOTONIC,
            help="Enforce monotonicity "
            "(hard attention architectures only). Default: %(default)s.",
        )
        parser.add_argument(
            "--no_enforce_monotonic",
            action="store_false",
            dest="enforce_monotonic",
        )
        parser.add_argument(
            "--attention_context",
            type=int,
            default=defaults.ATTENTION_CONTEXT,
            help="Width of attention context "
            "(hard attention architectures only). Default: %(default)s.",
        )
