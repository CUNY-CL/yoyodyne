"""Transducer model class."""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy
import torch
from maxwell import actions
from torch import nn

from . import expert, lstm
from .. import batches


class ActionError(Exception):
    pass


class TransducerNoFeatures(lstm.LSTMEncoderDecoder):
    """Transducer model with an LSTM backend and no features.

    After:
        Makarov, P., and Clematide, S. 2018. Imitation learning for neural
        morphological string transduction. In Proceedings of the 2018
        Conference on Empirical Methods in Natural Language Processing, pages
        2877–2882.

    This uses a trained oracle for imitation learning edits.
    """

    expert: expert.Expert
    # Constructed inside __init__.
    target_embeddings: nn.Embedding

    def __init__(
        self,
        expert,
        *args,
        **kwargs,
    ):
        """Initializes transducer model.

        Args:
            expert (expert.Expert): oracle that guides training for transducer.
            *args: passed to superclass.
            **kwargs: passed to superclass.
        """
        # Alternate outputs than dataset targets.
        kwargs["output_size"] = len(expert.actions)
        super().__init__(*args, **kwargs)
        # Model specific variables.
        self.expert = expert  # Oracle to train model.
        self.actions = self.expert.actions
        self.substitutions = self.actions.substitutions
        self.insertions = self.actions.insertions
        # Target embeddings use alternate padding.
        self.target_embeddings = self.init_embeddings(
            num_embeddings=self.output_size,
            embedding_size=self.embedding_size,
            pad_idx=self.actions.end_idx,
        )

    def forward(
        self,
        batch: batches.PaddedBatch,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Runs the encoder-decoder model.

        Args:
            batch (batches.PaddedBatch).

        Returns:
            Tuple[List[List[int]], torch.Tensor] of encoded prediction values
                and loss tensor; due to transducer setup, prediction is
                performed during training, so these are returned.
        """
        encoder_out, _ = self.encode(batch.source)
        # Ignores start symbol.
        encoder_out = encoder_out[:, 1:, :]
        source_padded = batch.source.padded[:, 1:]
        source_mask = batch.source.mask[:, 1:]
        prediction, loss = self.decode(
            encoder_out,
            source_padded,
            source_mask,
            teacher_forcing=self.teacher_forcing if self.training else False,
            target=batch.target.padded if batch.target else None,
            target_mask=batch.target.mask,
        )
        return prediction, loss

    def decode(
        self,
        encoder_out: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Decodes a sequence given the encoded input.

        This essentially serves as a wrapper for looping decode_step.

        Args:
            encoder_out (torch.Tensor): input symbols of shape
                B x seq_len x emb_size.
            source (torch.Tensor): encoded source input.
            source_mask (torch.Tensor): mask for source input.
            teacher_forcing (bool): Whether or not to decode
                with teacher forcing. Determines whether or not to rollout
                optimal actions.
            target (torch.Tensor, optional): encoded target input.
            target_mask (torch.Tensor, optional): mask for target input.

        Returns:
            Tuple[List[List[int]], torch.Tensor]: encoded prediction values
                and loss tensor; due to transducer setup, prediction is
                performed during training, so these are returned.
        """
        batch_size = source_mask.size(dim=0)
        input_length = (~source_mask).sum(dim=1)
        # Initializing values.
        alignment = torch.zeros(
            batch_size, device=self.device, dtype=torch.int64
        )
        action_count = torch.zeros(
            batch_size, device=self.device, dtype=torch.int64
        )
        last_action = torch.full(
            (batch_size,), self.actions.beg_idx, device=self.device
        )
        loss = torch.zeros(batch_size, device=self.device)
        prediction = [[] for _ in range(batch_size)]
        # Converting encodings for prediction.
        if target is not None:
            # Target and source need to be integers for SED values.
            # Clips EOW (idx = -1) for source and target.
            source = [
                s[~smask].tolist()[:-1]
                for s, smask in zip(source, source_mask)
            ]
            target = [
                t[~tmask].tolist()[:-1]
                for t, tmask in zip(target, target_mask)
            ]
        # Start of decoding.
        last_hiddens = self.init_hiddens(batch_size, self.decoder_layers)
        for _ in range(self.max_target_length):
            # Checks if completed all sequences.
            not_complete = last_action != self.actions.end_idx
            if not any(not_complete):
                break
            # Proceeds to make new edit; new action for all current decoding.
            action_count = torch.where(
                not_complete.to(self.device), action_count + 1, action_count
            )
            # Decoding.
            logits, last_hiddens = self.decode_step(
                encoder_out,
                last_action,
                last_hiddens,
                alignment,
            )
            # If given targets, asks expert for optimal actions.
            optim_action = (
                self.batch_expert_rollout(
                    source, target, alignment, prediction, not_complete
                )
                if target is not None
                else None
            )
            last_action = self.decode_action_step(
                logits,
                alignment,
                input_length,
                not_complete,
                optim_action=optim_action if teacher_forcing else None,
            )
            alignment = self.update_prediction(
                last_action, source, alignment, prediction
            )
            # If target, validation or training step loss required.
            if target is not None:
                log_sum_loss = self.log_sum_softmax_loss(logits, optim_action)
                loss = torch.where(not_complete, log_sum_loss + loss, loss)
        avg_loss = torch.mean(loss / action_count)
        return prediction, -avg_loss

    def decode_step(
        self,
        encoder_out: torch.Tensor,
        last_action: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        alignment: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Performs decoding step.

        Per item in batch: chooses single symbol in encoder_out using
        alignment. The symbol is concatenated with last_action and then
        decoded to compute logits.

        Args:
            encoder_out (torch.Tensor): output from encoder of shape
                B x seq_len x emb_size.
            last_action (torch.Tensor): edit action from previous decode_step
                of shape B x seq_len x emb_size.
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): previous hidden
                states from the decoder, both of shape 1 x B x decoder_dim.
            alignment (torch.Tensor): index of encoding symbols for decoding,
                per item in batch of shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: tuple of
                (logits, hidden_state) values.
        """
        # B x seq_len -> emb_size x 1 x B -> B x 1 x emb_size.
        alignment_expand = alignment.expand(
            encoder_out.size(-1), 1, -1
        ).transpose(0, -1)
        char_encoder_out = torch.gather(encoder_out, 1, alignment_expand)
        previous_action_embedding = self.target_embeddings(
            last_action
        ).unsqueeze(dim=1)
        decoder_input = torch.cat(
            (char_encoder_out, previous_action_embedding), dim=2
        )
        decoder_output, (h1, c1) = self.decoder(decoder_input, last_hiddens)
        logits = self.classifier(decoder_output).squeeze(dim=1)
        return logits, (h1, c1)

    def decode_action_step(
        self,
        logits: torch.Tensor,
        alignment: torch.Tensor,
        input_length: torch.Tensor,
        not_complete: torch.Tensor,
        optim_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes logits to find edit action.

        Finds possible actions given remaining size of source input and masks
        logits for edit action decoding.

        Args:
            logits (torch.Tensor): logit values from decode_step of shape
                B x num_actions.
            alignment (torch.Tensor): index of encoding symbols for decoding,
                per item in batch of shape B x seq_len.
            input_length (torch.Tensor): length of each item in batch.
            not_complete (torch.Tensor): boolean values designating which items
                have not terminated edits.
            optim_actions (List[List[int]], optional): optimal actions
                determined by expert, present when loss is being calculated.

        Returns:
            torch.Tensor: chosen edit action.
        """
        # Finds valid actions given remaining input length.
        end_of_input = (input_length - alignment) <= 1  # 1 -> Last char.
        valid_actions = [
            self.compute_valid_actions(eoi) if nc else [self.actions.end_idx]
            for eoi, nc in zip(end_of_input, not_complete)
        ]
        # Masks invalid actions.
        logits = self.action_probability_mask(logits, valid_actions)
        return self.choose_action(logits, not_complete, optim_action)

    def compute_valid_actions(self, end_of_input: bool) -> List[int]:
        """Gives all possible actions for remaining length of edits.

        Args:
            end_of_input (bool): indicates if this is the last input from
                string; if true, only insertions are available.

        Returns:
            List[actions.Edit]: actions known by transducer.
        """
        valid_actions = [self.actions.end_idx]
        valid_actions.extend(self.insertions)
        if not end_of_input:
            valid_actions.extend([self.actions.copy_idx, self.actions.del_idx])
            valid_actions.extend(self.substitutions)
        return valid_actions

    def action_probability_mask(
        self, logits: torch.Tensor, valid_actions: List[int]
    ) -> torch.Tensor:
        """Masks non-valid actions in logits."""
        with torch.no_grad():
            mask = torch.full(logits.shape, -math.inf, device=self.device)
            for row, action in zip(mask, valid_actions):
                row[action] = 0.0
            logits = mask + logits
        return logits

    def choose_action(
        self,
        logits: torch.Tensor,
        not_complete: torch.Tensor,
        optim_actions: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """Chooses transducer action from log_prob distribution.

        If training, uses dynamic oracle for selection.

        Args:
            log_probs (torch.Tensor): probability distribution of actions.
            not_complete (torch.Tensor): boolean tensor of batch length to
                indicate if each item in batch is complete.
            optim_actions (Optional[List[List[int]]]): optional encoded actions
                to use for action selection.

        Returns:
            torch.Tensor: action encodings.
        """
        # TODO: Merge logic into PyTorch methods.
        log_probs = nn.functional.log_softmax(logits, dim=1)
        if not self.training:
            # Argmax decoding.
            next_action = [
                torch.argmax(probs, dim=0) if nc else self.actions.end_idx
                for probs, nc in zip(log_probs, not_complete)
            ]
        else:
            # Training with dynamic oracle; chooses from optimal actions.
            with torch.no_grad():
                if self.expert.explore():
                    # Action is picked by random exploration.
                    next_action = [
                        self.sample(probs) if nc else self.actions.end_idx
                        for probs, nc in zip(log_probs, not_complete)
                    ]
                else:
                    # Action is picked from optim_actions.
                    next_action = []
                    for action, probs, nc in zip(
                        optim_actions, log_probs, not_complete
                    ):
                        if nc:
                            optim_logs = probs[action]
                            idx = int(torch.argmax(optim_logs, 0))
                            next_action.append(action[idx])
                        else:  # Already complete, so skip.
                            next_action.append(self.actions.end_idx)
        return torch.tensor(next_action, device=self.device, dtype=torch.int)

    # TODO: Merge action classes to remove need for this method.
    @staticmethod
    def remap_actions(
        action_scores: Dict[actions.Edit, float]
    ) -> Dict[actions.Edit, float]:
        """Maps generative oracle's edit to conditional counterpart.

        Oracle edits are a distinct subclass from edits learned from samples.

        This will eventually be removed.

        Args:
            action_scores (Dict[actions.Edit, float]): weights for each action.

        Returns:
            Dict[actions.Edit, float]: edit action-weight pairs.
        """
        remapped_action_scores = {}
        for action, score in action_scores.items():
            if isinstance(action, actions.GenerativeEdit):
                remapped_action = action.conditional_counterpart()
            elif isinstance(action, actions.Edit):
                remapped_action = action
            else:
                raise ActionError(
                    f"Unknown action: {action}, {score}, "
                    f"action_scores: {action_scores}"
                )
            remapped_action_scores[remapped_action] = score
        return remapped_action_scores

    def expert_rollout(
        self,
        source: List[int],
        target: List[int],
        alignment: int,
        prediction: List[int],
    ) -> List[int]:
        """Rolls out with optimal expert policy.

        Args:
            source (List[int]): input string.
            target (List[int]): target string.
            alignment (int): position in source to edit.
            prediction (List[str]): current prediction.

        Returns:
            List[int]: optimal action encodings.
        """
        raw_action_scores = self.expert.score(
            source,
            target,
            alignment,
            prediction,
            max_action_seq_len=self.max_target_length,
        )
        action_scores = self.remap_actions(raw_action_scores)
        optimal_value = min(action_scores.values())
        optimal_action = sorted(
            [
                self.actions.encode_unseen_action(action)
                for action, value in action_scores.items()
                if value == optimal_value
            ]
        )
        return optimal_action

    def batch_expert_rollout(
        self,
        source: List[List[int]],
        target: List[List[int]],
        alignment: torch.Tensor,
        prediction: List[List[int]],
        not_complete: torch.Tensor,
    ) -> List[List[int]]:
        """Performs expert rollout over batch."""
        return [
            self.expert_rollout(s, t, align, pred)
            if nc
            else self.actions.end_idx
            for s, t, align, pred, nc in zip(
                source, target, alignment, prediction, not_complete
            )
        ]

    def update_prediction(
        self,
        action: List[actions.Edit],
        source: List[int],
        alignment: torch.Tensor,
        prediction: List[List[str]],
    ) -> torch.Tensor:
        """Batch updates prediction and alignment information given actions.

        Args:
           action (List[actions.Edit]): valid actions, one per item in batch.
           source (List[int]): source strings, one per item in batch.
           alignment (torch.Tensor): index of current symbol for each item in
               batch.
           prediction (List[List[str]]): current predictions for each item in
               batch, one list of symbols per item.

        Return:
            torch.Tensor: new alignments for transduction.
        """
        alignment_update = torch.zeros(
            alignment.shape, device=self.device, dtype=torch.int64
        )
        for i in range(len(source)):
            a = self.actions.decode(action[i])
            if isinstance(a, actions.ConditionalCopy):
                symb = source[i][alignment[i]]
                prediction[i].append(symb)
                alignment_update[i] += 1
            elif isinstance(a, actions.ConditionalDel):
                alignment_update[i] += 1
            elif isinstance(a, actions.ConditionalIns):
                prediction[i].append(a.new)
            elif isinstance(a, actions.ConditionalSub):
                alignment_update[i] += 1
                prediction[i].append(a.new)
            elif isinstance(a, actions.End):
                prediction[i].append(self.end_idx)
            else:
                raise ActionError(f"Unknown action: {action[i]}")
        return alignment + alignment_update

    @staticmethod
    def log_sum_softmax_loss(
        logits: torch.Tensor, optimal_actions: List[int]
    ) -> torch.Tensor:
        """Computes log loss.

        After:
            Riezler, S. Prescher, D., Kuhn, J. and Johnson, M. 2000.
            Lexicalized stochastic modeling of constraint-based grammars using
            log-linear measures and EM training. In Proceedings of the 38th
            Annual Meeting of the Association for Computational
            Linguistics, pages 480–487.
        """
        opt_act = [
            log[actions] for log, actions in zip(logits, optimal_actions)
        ]
        log_sum_exp_terms = torch.stack(
            [torch.logsumexp(act, dim=-1) for act in opt_act]
        )
        normalization_term = torch.logsumexp(logits, -1)
        return log_sum_exp_terms - normalization_term

    def _get_loss_func(
        self, reduction: str
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Prevents base construction of unused loss function.
        return None

    def training_step(
        self, batch: batches.PaddedBatch, batch_idx: int
    ) -> Dict:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (batches.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        # Forward pass produces loss by default.
        _, loss = self(batch)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self, batch: batches.PaddedBatch, batch_idx: int
    ) -> Dict:
        predictions, loss = self.forward(batch)
        # Evaluation requires prediction as a tensor.
        predictions = self.convert_prediction(predictions)
        # Processes for accuracy calculation.
        predictions = self.evaluator.finalize_predictions(
            predictions, self.end_idx, self.pad_idx
        )
        return {
            "val_accuracy": self.evaluator.accuracy(
                predictions, batch.target.padded, self.pad_idx
            ),
            "val_loss": loss,
        }

    def predict_step(self, batch: Tuple[torch.tensor], batch_idx: int) -> Dict:
        predictions, _ = self.forward(
            batch,
        )
        # Evaluation requires prediction tensor.
        return self.convert_prediction(predictions)

    def convert_prediction(self, prediction: List[List[int]]) -> torch.Tensor:
        """Converts prediction values to tensor for evaluator compatibility."""
        max_len = len(max(prediction, key=len))
        for i, pred in enumerate(prediction):
            pad = [self.actions.end_idx] * (max_len - len(pred))
            pred.extend(pad)
            prediction[i] = torch.tensor(pred, dtype=torch.int)
        return torch.stack(prediction)

    def on_train_epoch_start(self) -> None:
        """Scheduler for oracle."""
        self.expert.roll_in_schedule(self.current_epoch)

    @staticmethod
    def sample(log_probs: torch.Tensor) -> torch.Tensor:
        """Samples an action from a log-probability distribution."""
        dist = torch.exp(log_probs)
        rand = numpy.random.rand()
        for action, p in enumerate(dist):
            rand -= p
            if rand <= 0:
                break
        return action


class TransducerFeatures(TransducerNoFeatures):
    """Transducer model with an LSTM backend."""

    feature_embeddings: nn.Embedding

    def __init__(self, *args, **kwargs):
        """Initializes transducer model.

        Functions equivalently to TransducerNoFeatures except concatenates
        n-hot encoding of feature values to encoded tensor.

        Args:
            features_idx (int): index marking the start of feature encodings.
            features_vocab_size (int): size of features vocab.
            *args: passed to superclass.
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        self.feature_embeddings = self.init_embeddings(
            self.features_vocab_size, self.embedding_size, self.pad_idx
        )
        # Overrides decoder to accomodate features.
        self.decoder = nn.LSTM(
            self.hidden_size * self.num_directions
            + self.embedding_size  # Hidden cells.
            + self.embedding_size,  # Features.
            self.hidden_size,
            dropout=self.dropout,
            num_layers=self.decoder_layers,
            batch_first=True,
        )

    def forward(
        self,
        batch: batches.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder model.

        Args:
            batch (batches.PaddedBatch).

        Returns:
            Tuple[List[List[int]], torch.Tensor]: encoded prediction values
                and loss tensor; due to transducer setup, prediction is
                performed during training, so these are returned.
        """
        encoder_out, _ = self.encode(batch.source)
        # Ignores start symbol.
        encoder_out = encoder_out[:, 1:, :]
        source_padded = batch.source.padded[:, 1:]
        source_mask = batch.source.mask[:, 1:]
        # Prepares feature embeddings.
        features_out = self.feature_embeddings(batch.features.padded)
        features_out = torch.sum(features_out, dim=1)
        denom = torch.sum(~batch.features.mask, dim=1, keepdim=True)
        denom = denom.expand(-1, features_out.size(1))
        features_out = features_out / denom
        # Concatenates output with source encoding.
        features_out = features_out.unsqueeze(dim=1)
        features_out = features_out.expand(-1, source_padded.size(1), -1)
        encoder_out_feat = torch.cat((encoder_out, features_out), dim=2)
        prediction, loss = self.decode(
            encoder_out_feat,
            source_padded,
            source_mask,
            teacher_forcing=self.teacher_forcing if self.training else False,
            target=batch.target.padded if batch.target else None,
            target_mask=batch.target.mask,
        )
        return prediction, loss


# TODO: Implement beam decoding.
