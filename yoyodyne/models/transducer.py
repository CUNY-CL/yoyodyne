"""Transducer model classes."""

import abc
from typing import Callable, Dict, List, Optional, Tuple

import torch
from maxwell import actions, sed
from torch import nn

from .. import data, defaults, special, util
from . import base, embeddings, expert, modules


class TransducerRNNModel(base.BaseModel):
    """Abstract base class for transducer models.

    Transducer models are essentially inattentive RNN models which predict
    edits trained using a learned oracle.

    If features are provided, the encodings are fused by concatenation of the
    source encoding with the features encoding, averaged across the length
    dimension and then scattered along the source length dimension, on the
    encoding dimension.

    As designed this model needs to engage in substantial accelerator-to-CPU
    transfer. To make these as cheap as possible, they are applied as bulk
    operations rather than piecemeal/as needed.

    After:
        Makarov, P., and Clematide, S. 2018. Imitation learning for neural
        morphological string transduction. In _Proceedings of the 2018
        Conference on Empirical Methods in Natural Language Processing_, pages
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

    classifier: nn.Linear
    expert: expert.Expert
    teacher_forcing: bool

    def __init__(
        self,
        sed_path: str,
        *args,
        index: Optional[data.Index] = None,  # Dummy value filled in via link.
        oracle_factor: int = defaults.ORACLE_FACTOR,
        teacher_forcing: bool = defaults.TEACHER_FORCING,
        **kwargs,
    ):
        actions = expert.ActionVocabulary(index)
        aligner = sed.StochasticEditDistance(
            sed.ParamDict.read_params(sed_path)
        )
        # The vocabularies are defined in a radically different way here.
        vocab_offset = index.vocab_size
        kwargs["target_vocab_size"] = len(actions)
        kwargs["vocab_size"] = index.vocab_size + len(actions)
        super().__init__(*args, **kwargs)
        self.actions = actions
        self.expert = expert.Expert(self.actions, aligner, oracle_factor)
        self.vocab_offset = vocab_offset
        self.teacher_forcing = teacher_forcing
        # These are optimizations to avoid extra dereferences.
        self.insertions = self.actions.insertions
        self.substitutions = self.actions.substitutions
        self.classifier = nn.Linear(
            self.decoder_hidden_size, self.target_vocab_size
        )
        self.decoder = self.get_decoder()
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                # Modules.
                "classifier",
                "decoder",
                "embeddings",
                "features_encoder",
                "source_encoder",
            ]
        )

    def _get_loss_func(
        self,
    ) -> Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        # Prevents base model from constructing a loss function we don't need;
        # the transducer computes loss as it goes.
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
                performed during training, so these are returned together.

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
            target=batch.target.tensor if batch.has_target else None,
            target_mask=batch.target.mask if batch.has_target else None,
        )

    def greedy_decode(
        self,
        source: torch.Tensor,
        encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Decodes a sequence given the encoded input.

        Prediction is performed as a side effect of training, so we also
        return the predictions.

        Args:
            source (torch.Tensor): source symbols.
            encoded (torch.Tensor): encoded source symbols.
            source_mask (torch.Tensor): mask for source input.
            target (torch.Tensor, optional): encoded target input.
            target_mask (torch.Tensor, optional): mask for target input.

        Returns:
            Tuple[List[List[int]], torch.Tensor]: encoded prediction values
                and loss tensor.
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
        prediction: List[List[int]] = [[] for _ in range(batch_size)]
        state = self.decoder.initial_state(batch_size)
        context = self.decoder.get_context(source, encoded)
        # Converting encodings for prediction.
        if target is not None:
            # Target and source need to be integers for SED values.
            # Removes the trailing end-of-sequence token from each sequence.
            source_list = [
                s[~smask].tolist()[:-1]
                for s, smask in zip(source, source_mask)
            ]
            target_list = [
                t[~tmask].tolist()[:-1]
                for t, tmask in zip(target, target_mask)
            ]
        teacher_forcing = (
            self.teacher_forcing if self.training or self.validating else False
        )
        for _ in range(self.max_target_length):
            nonfinal = last_action != self.actions.end_idx
            if not nonfinal.any():
                break
            action_count = torch.where(
                nonfinal,
                action_count + 1,
                action_count,
            )
            # Decoding.
            # We offset the action idx by the symbol vocab size so that we
            # can index into the shared embeddings matrix.
            decoded, state = self.decoder(
                last_action.unsqueeze(dim=1) + self.vocab_offset,
                self.embeddings,
                context,
                None,  # Mask, but is ignored.
                state,
            )
            logits = self.classifier(decoded).squeeze(1)
            # If given targets, asks expert for optimal actions.
            optim_actions = (
                self._batch_expert_rollout(
                    source_list,
                    target_list,
                    alignment,
                    prediction,
                    nonfinal,
                )
                if target is not None
                else None
            )
            last_action = self._decode_action_step(
                logits,
                alignment,
                lengths,
                nonfinal,
                optim_actions=optim_actions if teacher_forcing else None,
            )
            alignment = self._update_prediction(
                last_action, source_list, alignment, prediction
            )
            if target is not None:
                # Computes loss if target is present. optim_actions is always
                # computed when target is not None, so it is available here
                # regardless of whether teacher forcing was used for action
                # selection. This is intentional: the expert always provides
                # the loss signal even when student forcing.
                nll = self._log_sum_softmax_loss(logits, optim_actions)
                loss = torch.where(nonfinal, nll + loss, loss)
        # Guards against division by zero.
        safe_count = action_count.clamp(min=1)
        avg_loss = torch.mean(loss / safe_count)
        return prediction, avg_loss

    def _batch_expert_rollout(
        self,
        source: List[List[int]],
        target: List[List[int]],
        alignment: torch.Tensor,
        prediction: List[List[int]],
        nonfinal: torch.Tensor,
    ) -> List[List[int]]:
        """Performs expert rollout over batch.

        Args:
            source (List[List[int]]): source token-id sequences.
            target (List[List[int]]): target token-id sequences.
            alignment (torch.Tensor): current alignment positions of shape B.
            prediction (List[List[int]]): current predictions per item.
            nonfinal (torch.Tensor): completion flags of shape B.

        Returns:
            List[List[int]]: optimal action index lists, one per batch item.
                Completed items have [self.actions.end_idx].
        """
        # Bulk conversions to CPU lists.
        alignment_list = alignment.tolist()
        nonfinal_list = nonfinal.tolist()
        return [
            (
                self._expert_rollout(s, t, align, pred)
                if nc
                else [self.actions.end_idx]
            )
            for s, t, align, pred, nc in zip(
                source, target, alignment_list, prediction, nonfinal_list
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
            source (List[int]): input token-id sequence.
            target (List[int]): target token-id sequence.
            alignment (int): position in source to edit.
            prediction (List[int]): current prediction token ids.

        Returns:
            List[int]: optimal action encodings (may be empty if no actions
                are valid, which should not occur in normal operation).
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
        optimal_actions = sorted(
            self.actions.encode_unseen_action(action)
            for action, value in action_scores.items()
            if value == optimal_value
        )
        return optimal_actions

    def _decode_action_step(
        self,
        logits: torch.Tensor,
        alignment: torch.Tensor,
        lengths: torch.Tensor,
        nonfinal: torch.Tensor,
        optim_actions: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """Decodes logits to find edit action.

        Finds possible actions given remaining size of source input and masks
        logits for edit action decoding.

        Args:
            logits (torch.Tensor): logit values from decode_step of shape
                B x num_actions.
            alignment (torch.Tensor): index of encoding symbols for decoding,
                per item in batch of shape B.
            lengths (torch.Tensor): length of each item in batch of shape B.
            nonfinal (torch.Tensor): boolean values designating which items
                have not terminated edits of shape B.
            optim_actions (List[List[int]], optional): optimal actions
                determined by expert, present when loss is being calculated.

        Returns:
            torch.Tensor: chosen edit action indices of shape B.
        """
        # Bulk conversion to CPU list.
        end_of_input = ((lengths - alignment) <= 1).tolist()
        nonfinal_list = nonfinal.tolist()
        valid_actions = [
            (
                self._compute_valid_actions(eoi)
                if nc
                else [self.actions.end_idx]
            )
            for eoi, nc in zip(end_of_input, nonfinal_list)
        ]
        # Masks invalid actions.
        logits = self._action_probability_mask(logits, valid_actions)
        return self._choose_action(logits, nonfinal_list, optim_actions)

    def _compute_valid_actions(self, end_of_input: bool) -> List[int]:
        """Gives all possible actions for remaining length of edits.

        Args:
            end_of_input (bool): indicates if this is the last input symbol;
                if true, only insertions and end are available.

        Returns:
            List[int]: valid action indices.
        """
        valid_actions = [self.actions.end_idx]
        valid_actions.extend(self.insertions)
        if not end_of_input:
            valid_actions.extend([self.actions.copy_idx, self.actions.del_idx])
            valid_actions.extend(self.substitutions)
        return valid_actions

    def _action_probability_mask(
        self, logits: torch.Tensor, valid_actions: List[List[int]]
    ) -> torch.Tensor:
        """Masks non-valid actions in logits.

        Args:
            logits (torch.Tensor): raw logits of shape B x num_actions.
            valid_actions (List[List[int]]): per-item lists of valid action
                indices.

        Returns:
            torch.Tensor: masked logits of shape B x num_actions.
        """
        # Builds the full mask once.
        with torch.no_grad():
            mask = torch.full(
                logits.shape, defaults.NEG_INF, device=self.device
            )
            for row_idx, action_indices in enumerate(valid_actions):
                mask[row_idx, action_indices] = 0.0
            logits = mask + logits
        return logits

    def _choose_action(
        self,
        logits: torch.Tensor,
        nonfinal: List[bool],
        optim_actions: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """Chooses transducer action from log-probability distribution.

        If training with teacher forcing, uses the dynamic oracle for
        selection. Otherwise performs argmax (greedy) decoding.

        Args:
            logits (torch.Tensor): masked logit values of shape
                B x num_actions.
            nonfinal (List[bool]): per-item completion flags.
            optim_actions (List[List[int]]], optional): encoded actions to use
                for action selection during teacher-forced training.

        Returns:
            torch.Tensor: action indices of shape B.
        """
        log_probs = nn.functional.log_softmax(logits, dim=1)
        end_idx = self.actions.end_idx
        if optim_actions is None:
            # Picks the highest-probability valid action.
            next_action = torch.argmax(log_probs, dim=1).tolist()
            next_action = [
                a if nc else end_idx for a, nc in zip(next_action, nonfinal)
            ]
        else:
            # Training with dynamic oracle.
            if self.expert.explore():
                # Sample from the model's own distribution.
                sampled = (
                    torch.multinomial(log_probs.exp(), 1).squeeze(1).tolist()
                )
                next_action = [
                    a if nc else end_idx for a, nc in zip(sampled, nonfinal)
                ]
            else:
                # Pick the highest-probability action among optimal actions.
                with torch.no_grad():
                    log_probs_cpu = log_probs.cpu()
                    next_action = []
                    for action_indices, log_probs_row, nc in zip(
                        optim_actions, log_probs_cpu, nonfinal
                    ):
                        if nc:
                            optim_logs = log_probs_row[action_indices]
                            best = action_indices[
                                int(torch.argmax(optim_logs, dim=0))
                            ]
                            next_action.append(best)
                        else:
                            next_action.append(end_idx)
        return torch.tensor(next_action, device=self.device, dtype=torch.int64)

    # TODO: Merge action classes in Maxwell to remove the need for this method.
    @staticmethod
    def _remap_actions(
        action_scores: Dict[actions.Edit, float],
    ) -> Dict[actions.Edit, float]:
        """Maps generative oracle edits to their conditional counterparts.

        The Maxwell expert emits generative edit actions while the transducer
        trains on conditional edits.. This shim converts between them.

        This will be removed once Maxwell emits conditional edits directly.

        Args:
            action_scores (Dict[actions.Edit, float]): weights for each action.

        Returns:
            Dict[actions.Edit, float]: edit action-weight pairs using
                conditional edit types.
        """
        remapped: Dict[actions.Edit, float] = {}
        for action, score in action_scores.items():
            if isinstance(action, actions.GenerativeEdit):
                remapped[action.conditional_counterpart()] = score
            elif isinstance(action, actions.Edit):
                remapped[action] = score
            else:
                raise expert.ActionError(
                    f"Unknown action: {action}, {score}, "
                    f"action_scores: {action_scores}"
                )
        return remapped

    def _update_prediction(
        self,
        action: torch.Tensor,
        source: List[List[int]],
        alignment: torch.Tensor,
        prediction: List[List[int]],
    ) -> torch.Tensor:
        """Batch updates prediction and alignment information from actions.

        Args:
            action (torch.Tensor): action indices, one per item in batch of
                shape B.
            source (List[List[int]]): source token-id sequences, one per item.
            alignment (torch.Tensor): index of current source symbol for each
                item in batch of shape B.
            prediction (List[List[int]]): current token-id predictions for each
               item in batch; this is modified in-place.

        Returns:
            torch.Tensor: updated alignments of shape B.
        """
        # Bulk conversions to CPU lists.
        action_list = action.tolist()
        alignment_list = alignment.tolist()
        alignment_update = [0] * len(source)
        for i, (act_idx, align) in enumerate(zip(action_list, alignment_list)):
            decoded_action = self.actions.decode(act_idx)
            if isinstance(decoded_action, actions.ConditionalCopy):
                prediction[i].append(source[i][align])
                alignment_update[i] = 1
            elif isinstance(decoded_action, actions.ConditionalDel):
                alignment_update[i] = 1
            elif isinstance(decoded_action, actions.ConditionalIns):
                prediction[i].append(decoded_action.new)
            elif isinstance(decoded_action, actions.ConditionalSub):
                prediction[i].append(decoded_action.new)
                alignment_update[i] = 1
            elif isinstance(decoded_action, actions.End):
                prediction[i].append(special.END_IDX)
            else:
                raise expert.ActionError(f"Unknown action: {act_idx}")
        # Builds the update tensor from a Python list.
        update = torch.tensor(
            alignment_update, device=self.device, dtype=torch.int64
        )
        return alignment + update

    @staticmethod
    def _log_sum_softmax_loss(
        logits: torch.Tensor,
        optimal_actions: List[List[int]],
    ) -> torch.Tensor:
        """Computes per-item negative marginal log-likelihood loss.

        For each item, sums the probability mass over all optimal actions and
        computes the negative log of that sum normalized by the total mass.
        This is equivalent to minimizing the negative marginal log-likelihood
        of the set of optimal actions.

        Args:
            logits (torch.Tensor): raw logits of shape B x num_actions.
            optimal_actions (List[List[int]]): per-item lists of optimal action
                indices; each inner list must be non-empty.

        Returns:
            torch.Tensor: per-item NLL values of shape B.

        After:
            Riezler, S., Prescher, D., Kuhn, J., and Johnson, M. 2000.
            Lexicalized stochastic modeling of constraint-based grammars using
            log-linear measures and EM training. In _Proceedings of the 38th
            Annual Meeting of the Association for Computational
            Linguistics_, pages 480–487.
        """
        # Numerator: logsumexp over optimal action logits, per item.
        # TODO: consider padding to a fixed width and using a masked logsumexp.
        opt_logits = [
            logits[i, acts] for i, acts in enumerate(optimal_actions)
        ]
        log_sum_exp_optimal = torch.stack(
            [torch.logsumexp(ol, dim=-1) for ol in opt_logits]
        )
        # Denominator: logsumexp over all actions.
        log_sum_exp_all = torch.logsumexp(logits, dim=-1)
        # Returns NLL; the caller will negate after accumulating.
        return log_sum_exp_all - log_sum_exp_optimal

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        predictions, _ = self(batch)
        length = max(len(prediction) for prediction in predictions)
        # Pads; truncation cannot occur by construction.
        return self._convert_predictions(predictions, length)

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        predictions, _ = self(batch)
        self._update_metrics(
            self._convert_predictions(
                predictions, batch.target.tensor.size(1)
            ),
            batch.target.tensor,
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
            predictions (List[List[int]]): lists of prediction token indices.
            length (int): desired sequence length.

        Returns:
            torch.Tensor: predictions of shape B x length.
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
            prediction (List[int]): prediction token indices.
            length (int): desired length.

        Returns:
            torch.Tensor: prediction of shape length.
        """
        tensor = torch.tensor(prediction, device=self.device)
        if len(prediction) == length:
            return tensor
        elif len(prediction) < length:
            padding = length - len(prediction)
            return nn.functional.pad(
                tensor, (0, padding), "constant", self.actions.end_idx
            )
        else:
            # Truncates; this is never used during the prediction step hence
            # its late ordering.
            return tensor[:length]

    @staticmethod
    def init_embeddings(
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
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
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
            embedding_size=self.embedding_size,
            hidden_size=self.decoder_hidden_size,
            layers=self.decoder_layers,
        )

    @property
    def name(self) -> str:
        return "transducer LSTM"
