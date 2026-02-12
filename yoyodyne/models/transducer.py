"""Transducer model class."""

import abc
from typing import Callable, Dict, List, Optional, Tuple

from maxwell import actions, sed
import numpy
import torch
from torch import nn

from .. import data, defaults, special, util
from . import base, embeddings, expert, modules


class TransducerRNNModel(base.BaseModel):
    """Abstract base class for transducer models.

    Transducer models are essentially inattentive RNN models which
    predict edits trained using a learned oracle.

    If features are provided, the encodings are fused by concatenation of the
    source encoding with the features encoding, averaged across the length
    dimension and then scattered along the source length dimension, on the
    encoding dimension.

    After:
        Makarov, P., and Clematide, S. 2018. Imitation learning for neural
        morphological string transduction. In Proceedings of the 2018
        Conference on Empirical Methods in Natural Language Processing, pages
        2877–2882.

     Args:
        sed_path (str): path to SED parameters .pkl.
        *args: passed to superclass.
        index (data.Index, optional): index for mapping symbols to indices.
        oracle_factor (int, optional): a scaling factor for scheduling
            predictions during transducer training.
        teacher_forcing (bool, optional): should teacher (rather than student)
            forcing be used?
        **kwargs: passed to superclass.
    """

    expert: expert.Expert
    teacher_forcing: bool
    classifer: nn.Linear

    def __init__(
        self,
        sed_path: str,
        *args,
        index: Optional[data.Index] = None,  # Dummy value filled in via link.
        oracle_factor: int = defaults.ORACLE_FACTOR,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        self.actions = expert.ActionVocabulary(index)
        self.insertions = self.actions.insertions
        self.substitutions = self.actions.substitutions
        aligner = sed.StochasticEditDistance(
            sed.ParamDict.read_params(sed_path)
        )
        self.expert = expert.Expert(actions, aligner, oracle_factor)
        # The vocabularies are defined in a radically different way here.
        self.vocab_offset = index.vocab_size
        kwargs["target_vocab_size"] = len(self.actions)
        kwargs["vocab_size"] = index.vocab_size + len(self.actions)
        super().__init__(*args, **kwargs)
        self.teacher_forcing = teacher_forcing
        self.classifier = nn.Linear(
            self.decoder_hidden_size, self.target_vocab_size
        )

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Prevents base construction of unused loss function.
        return None

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
        batch: data.Batch,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            Tuple[List[List[int]], torch.Tensor]: encoded prediction values
                and loss tensor; due to transducer setup, prediction is
                performed during training, so these are returned.

        Raises:
            base.ConfigurationError: Features encoder specified but no feature
                column specified.
            base.ConfigurationError: Features column specified but no feature
                encoder specified.
        """
        encoded = self.source_encoder(
            batch.source, self.embeddings, is_source=True
        )
        # Ignores start symbol.
        encoded = encoded[:, 1:, :]
        source = batch.source.tensor[:, 1:]
        source_mask = batch.source.mask[:, 1:]
        if self.has_features_encoder:
            if not batch.has_features:
                raise base.ConfigurationError(
                    "Features encoder specified but "
                    "no feature column specified"
                )
            features_encoded = self.features_encoder(
                batch.features,
                self.embeddings,
                is_source=False,
            )
            features_encoded = features_encoded.mean(dim=1, keepdim=True)
            features_encoded = features_encoded.expand(-1, encoded.size(1), -1)
            encoded = torch.cat((encoded, features_encoded), dim=2)
        elif batch.has_features:
            raise base.ConfigurationError(
                "Feature column specified but no feature encoder specified"
            )
        return self.greedy_decode(
            source,
            encoded,
            source_mask,
            teacher_forcing=self.teacher_forcing if self.training else False,
            target=batch.target.tensor if batch.has_target else None,
            target_mask=batch.target.mask if batch.has_target else None,
        )

    def greedy_decode(
        self,
        source: torch.Tensor,
        encoded: torch.Tensor,
        source_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Decodes a sequence given the encoded input.

        This essentially serves as a wrapper for looping decode_step.

        Args:
            source (torch.Tensor): source symbols.
            encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for source input.
            teacher_forcing (bool): whether or not to decode
                with teacher forcing; determines whether or not to rollout
                optimal actions.
            target (torch.Tensor, optional): encoded target input.
            target_mask (torch.Tensor, optional): mask for target input.

        Returns:
            Tuple[List[List[int]], torch.Tensor]: encoded prediction values
                and loss tensor; due to transducer setup, prediction is
                performed during training, so these are returned.
        """
        batch_size = source_mask.size(0)
        lengths = (~source_mask).sum(dim=1)
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
        state = self.decoder.initial_state(batch_size)
        # Converting encodings for prediction.
        if target is not None:
            # Target and source need to be integers for SED values.
            # Clips EOW (idx = -1) for source and target.
            source_list = [
                s[~smask].tolist()[:-1]
                for s, smask in zip(source, source_mask)
            ]
            target_list = [
                t[~tmask].tolist()[:-1]
                for t, tmask in zip(target, target_mask)
            ]
        for _ in range(self.max_target_length):
            # Checks if completed all sequences.
            not_complete = last_action != self.actions.end_idx
            if not any(not_complete):
                break
            # Proceeds to make new edit; new action for all current decoding.
            action_count = torch.where(
                not_complete.to(self.device),
                action_count + 1,
                action_count,
            )
            # Decoding.
            # We offset the action idx by the symbol vocab size so that we
            # can index into the shared embeddings matrix.
            decoded, state = self.decoder(
                source,
                encoded,
                # Accomodates RNNDecoder; see encoder_mask behavior.
                ~(alignment.unsqueeze(1) + 1),
                last_action.unsqueeze(dim=1) + self.vocab_offset,
                state,
                self.embeddings,
            )
            logits = self.classifier(decoded).squeeze(1)
            # If given targets, asks expert for optimal actions.
            optim_actions = (
                self._batch_expert_rollout(
                    source_list,
                    target_list,
                    alignment,
                    prediction,
                    not_complete,
                )
                if target is not None
                else None
            )
            last_action = self._decode_action_step(
                logits,
                alignment,
                lengths,
                not_complete,
                optim_actions=optim_actions if teacher_forcing else None,
            )
            alignment = self._update_prediction(
                last_action, source_list, alignment, prediction
            )
            # If target, validation or training step loss required.
            if target is not None:
                log_sum_loss = self._log_sum_softmax_loss(
                    logits, optim_actions
                )
                loss = torch.where(not_complete, log_sum_loss + loss, loss)
        avg_loss = torch.mean(loss / action_count)
        return prediction, -avg_loss

    def _batch_expert_rollout(
        self,
        source: List[List[int]],
        target: List[List[int]],
        alignment: torch.Tensor,
        prediction: List[List[int]],
        not_complete: torch.Tensor,
    ) -> List[List[int]]:
        """Performs expert rollout over batch."""
        return [
            (
                self._expert_rollout(s, t, align, pred)
                if nc
                else self.actions.end_idx
            )
            for s, t, align, pred, nc in zip(
                source, target, alignment, prediction, not_complete
            )
        ]

    def _expert_rollout(
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
        action_scores = self._remap_actions(raw_action_scores)
        optimal_value = min(action_scores.values())
        optimal_action = sorted(
            [
                self.actions.encode_unseen_action(action)
                for action, value in action_scores.items()
                if value == optimal_value
            ]
        )
        return optimal_action

    def _decode_action_step(
        self,
        logits: torch.Tensor,
        alignment: torch.Tensor,
        lengths: torch.Tensor,
        not_complete: torch.Tensor,
        optim_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes logits to find edit action.

        Finds possible actions given remaining size of source input and masks
        logits for edit action decoding.

        Args:
            logits (torch.Tensor): logit values from decode_step of shape
                B x num_actions.
            alignment (torch.Tensor): index of encoding symbols for decoding,
                per item in batch of shape B x seq_len.
            lengths (torch.Tensor): length of each item in batch.
            not_complete (torch.Tensor): boolean values designating which items
                have not terminated edits.
            optim_actions (List[List[int]], optional): optimal actions
                determined by expert, present when loss is being calculated.

        Returns:
            torch.Tensor: chosen edit action.
        """
        # Finds valid actions given remaining input length.
        end_of_input = (lengths - alignment) <= 1  # 1 -> Last char.
        valid_actions = [
            (
                self._compute_valid_actions(eoi)
                if nc
                else [self.actions.end_idx]
            )
            for eoi, nc in zip(end_of_input, not_complete)
        ]
        # Masks invalid actions.
        logits = self._action_probability_mask(logits, valid_actions)
        return self._choose_action(logits, not_complete, optim_actions)

    def _compute_valid_actions(self, end_of_input: bool) -> List[int]:
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

    def _action_probability_mask(
        self, logits: torch.Tensor, valid_actions: List[int]
    ) -> torch.Tensor:
        """Masks non-valid actions in logits."""
        with torch.no_grad():
            mask = torch.full(
                logits.shape, defaults.NEG_INF, device=self.device
            )
            for row, action in zip(mask, valid_actions):
                row[action] = 0.0
            logits = mask + logits
        return logits

    def _choose_action(
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
        if optim_actions is None:
            # Argmax decoding.
            next_action = [
                (torch.argmax(probs, dim=0) if nc else self.actions.end_idx)
                for probs, nc in zip(log_probs, not_complete)
            ]
        else:
            # Training with dynamic oracle; chooses from optimal actions.
            with torch.no_grad():
                if self.expert.explore():
                    # Action is picked by random exploration.
                    next_action = [
                        (self._sample(probs) if nc else self.actions.end_idx)
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

    @staticmethod
    def _sample(log_probs: torch.Tensor) -> torch.Tensor:
        """Samples an action from a log-probability distribution."""
        dist = torch.exp(log_probs)
        rand = numpy.random.rand()
        for action, p in enumerate(dist):
            rand -= p
            if rand <= 0:
                break
        return action

    # TODO: Merge action classes to remove need for this method.
    @staticmethod
    def _remap_actions(
        action_scores: Dict[actions.Edit, float],
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
                raise expert.ActionError(
                    f"Unknown action: {action}, {score}, "
                    f"action_scores: {action_scores}"
                )
            remapped_action_scores[remapped_action] = score
        return remapped_action_scores

    def _update_prediction(
        self,
        action: List[actions.Edit],
        source: List[int],
        alignment: torch.Tensor,
        prediction: List[List[str]],
    ) -> torch.Tensor:
        """Batch updates prediction and alignment information from actions.

        Args:
           action (List[actions.Edit]): valid actions, one per item in
                batch.
           source (List[int]): source strings, one per item in batch.
           alignment (torch.Tensor): index of current symbol for each item
                in batch.
           prediction (List[List[str]]): current predictions for each item
                in batch, one list of symbols per item.

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
                prediction[i].append(special.END_IDX)
            else:
                raise expert.ActionError(f"Unknown action: {action[i]}")
        return alignment + alignment_update

    @staticmethod
    def _log_sum_softmax_loss(
        logits: torch.Tensor, optimal_actions: List[int]
    ) -> torch.Tensor:
        """Computes log loss.

        After:
            Riezler, S., Prescher, D., Kuhn, J., and Johnson, M. 2000.
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

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        predictions, _ = self(batch)
        length = max(len(prediction) for prediction in predictions)
        # Pads; truncation cannot occur by construction.
        return self._convert_predictions(predictions, length)

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        predictions, _ = self(batch)
        self._update_metrics(
            self._convert_predictions(predictions), batch.target.tensor
        )

    def on_train_epoch_start(self) -> None:
        self.expert.roll_in_schedule(self.current_epoch)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        """Runs one step of training.

        Training loss is tracked.

        Args:
            batch (data.Batch)
            batch_idx (int).

        Returns:
            torch.Tensor: training loss.
        """
        # Forward pass produces loss.
        _, loss = self(batch)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: data.Batch, batch_idx: int) -> None:
        predictions, loss = self(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        # This needs to conform to target size for evaluation.
        length = batch.target.tensor.size(1)
        self._update_metrics(
            self._convert_predictions(predictions, length),
            batch.target.tensor,
        )

    def _convert_predictions(
        self, predictions: List[List[int]], length: int
    ) -> torch.Tensor:
        """Converts a batch of predictions to the proper form.

        This repeatedly calls `_resize_prediction`, stacks, and then converts
        redundant END to PAD.

        Args:
            predictions (list[list[int][): lists of prediction indices.
            length (int): desired length.

        Returns:
            torch.Tensor.
        """
        return util.pad_tensor_after_end(
            torch.stack(
                [
                    self._resize_prediction(prediction, length)
                    for prediction in predictions
                ]
            )
        )

    def _resize_prediction(
        self, prediction: List[int], length: int
    ) -> torch.Tensor:
        """Resizes the prediction and converts to tensor.

        If the prediction matches the desired length it is just converted to
        tensor. If the prediction is longer than the desired length, it is
        first truncated. If the prediction is shorter than the desired length,
        it is padded using END.

        Args:
            predictions (list[int]): prediction indices.
            length (int): desired length.

        Returns:
            torch.Tensor.
        """
        if len(prediction) == length:
            # Just converts to tensor.
            return torch.tensor(prediction, device=self.device)
        elif len(prediction) < length:
            # Pads.
            padding = length - len(prediction)
            return nn.functional.pad(
                torch.tensor(prediction, device=self.device),
                (0, padding),
                "constant",
                self.actions.end_idx,
            )
        else:
            # Truncates; this is never used during the prediction step hence
            # its late ordering.
            return torch.tensor(prediction[:length], device=self.device)

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


class TransducerGRUModel(TransducerRNNModel):
    """Transducer with GRU backend."""

    def get_decoder(self) -> modules.GRUDecoder:
        return modules.GRUDecoder(
            bidirectional=False,
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.decoder_hidden_size,
            num_embeddings=self.num_embeddings,
        )

    @property
    def name(self) -> str:
        return "transducer GRU"


class TransducerLSTMModel(TransducerRNNModel):
    """Transducer with LSTM backend."""

    def get_decoder(self) -> modules.LSTMDecoder:
        return modules.LSTMDecoder(
            bidirectional=False,
            decoder_input_size=self.decoder_input_size,
            dropout=self.decoder_dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.decoder_hidden_size,
            num_embeddings=self.num_embeddings,
        )

    @property
    def name(self) -> str:
        return "transducer LSTM"
