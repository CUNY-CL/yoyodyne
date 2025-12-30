"""Hard monotonic neural HMM classes."""

import abc
from typing import Optional, Tuple

import torch
from torch import nn

from .. import data, defaults, special
from . import base, embeddings, modules


class HardAttentionRNNModel(base.BaseModel):
    """Abstract base class for hard attention models.

    Learns probability distribution of target string by modeling transduction
    of source string to target string as Markov process. Assumes each symbol
    produced is conditioned by state transitions over each source symbol.

    The default model assumes independence between state and non-monotonic
    progression over source string. `enforce_monotonic` enforces monotonic
    state transition (model progresses over each source symbol), and
    attention context allows conditioning of state transition over the
    previous n states.

    If features are provided, the encodings are fused by concatenation of the
    source encoding with the features encoding, averaged across the length
    dimension and then scattered along the source length dimension, on the
    encoding dimension.

    After:
        Wu, S. and Cotterell, R. 2019. Exact hard monotonic attention for
        symbol-level transduction. In _Proceedings of the 57th Annual
        Meeting of the Association for Computational Linguistics_, pages
        1530-1537.

    Original implementation:
        https://github.com/shijie-wu/neural-transducer

     Args:
        *args: passed to superclass.
        attention_context (int, optional): size of context window for
            conditioning state transition; if 0, state transitions are
            independent.
        enforce_monotonic (bool, optional): enforces monotonic state
            transition in decoding.
        **kwargs: passed to superclass.
    """

    enforce_monotonic: bool
    attention_context: int

    def __init__(
        self,
        *args,
        attention_context: int = defaults.ATTENTION_CONTEXT,
        enforce_monotonic: bool = defaults.ENFORCE_MONOTONIC,
        **kwargs,
    ):
        self.attention_context = attention_context
        self.enforce_monotonic = enforce_monotonic
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(
            self.decoder.output_size, self.target_vocab_size
        )

    # Prevents a loss function object from being constructed.

    def _get_loss_func(self) -> None:
        return

    def _loss(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes average loss.

        Args:
            encoded (torch.Tensor): encoded source symbols
                of shape B x src_len x (encoder_hidden * num_directions).
            mask (torch.Tensor): mask.
            target (torch.Tensor): target symbols.

        Returns:
            torch.Tensor: average loss.
        """
        batch_size = mask.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        emissions, transitions, state = self.decode_step(
            encoded, mask, symbol, state
        )
        likelihood = self._emissions(
            transitions[:, 0].unsqueeze(1),
            emissions,
            target[:, 0],
        )
        target_len = target.size(1)
        for t in range(1, target_len):
            emissions, transitions, state = self.decode_step(
                encoded, mask, target[:, t - 1].unsqueeze(1), state
            )
            likelihood = self._transitions(likelihood, transitions)
            likelihood = self._emissions(likelihood, emissions, target[:, t])
        return self._scale(likelihood, target_len)

    def decode_step(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        symbol: torch.Tensor,
        state: modules.RNNState,
    ) -> Tuple[torch.Tensor, torch.Tensor, modules.RNNState]:
        """Single decoder step.

        Args:
            encoded (torch.Tensor): encoded source symbols of shape
                B x src_len x (encoder_hidden * num_directions).
            mask (torch.Tensor): mask.
            symbol (torch.Tensor): target symbol for current state.
            state (modules.RNNState): RNN state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: emission
                probabilities, transition probabilities, and the RNN state.
        """
        emissions, transitions, state = self.decoder(
            encoded,
            mask,
            symbol,
            state,
            self.embeddings,
        )
        logits = self.classifier(emissions)
        scores = nn.functional.log_softmax(logits, dim=2)
        # Expands matrix for all time steps.
        if self.enforce_monotonic:
            transitions = self._apply_mono_mask(transitions)
        return scores, transitions, state

    @staticmethod
    def _apply_mono_mask(transitions: torch.Tensor) -> torch.Tensor:
        """Applies monotonic attention mask to transition probabilities.

        Enforces zero log-probability values for all non-monotonic relations
        in the transition tensor (i.e., all values i < j per row j).

        Args:
            transitions (torch.Tensor): transition probabilities between
                all hidden states of shape B x src_len x src_len.

        Returns:
            torch.Tensor: masked transition probabilities of shape
                B x src_len x src_len.
        """
        mask = torch.ones_like(transitions[0]).triu().unsqueeze(0)
        # Using 0 log-probability value for masking; this is borrowed from the
        # original implementation.
        mask = (mask - 1) * defaults.NEG_LOG_EPSILON
        transitions = transitions + mask
        transitions = transitions - transitions.logsumexp(dim=2, keepdim=True)
        return transitions

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

    def forward(self, batch: data.Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            batch (data.Batch).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: if
                training, the loss tensor; otherwise, also predictions of shape
                B x pred_seq_length.

        Raises:
            base.ConfigurationError: Features encoder specified but no feature
                column specified.
            base.ConfigurationError: Feature column specified but no feature
                encoder specified.
        """
        encoded = self.source_encoder(batch.source, self.embeddings)
        if self.has_features_encoder:
            if not batch.has_features:
                raise base.ConfigurationError(
                    "Features encoder specified but "
                    "no feature column specified"
                )
            features_encoded = self.features_encoder(
                batch.features, self.embeddings
            )
            features_encoded = features_encoded.mean(dim=1, keepdim=True)
            features_encoded = features_encoded.expand(-1, encoded.size(1), -1)
            encoded = torch.cat((encoded, features_encoded), dim=2)
        elif batch.has_features:
            raise base.ConfigurationError(
                "Feature column specified but no feature encoder specified"
            )
        if self.training:
            return self._loss(
                encoded,
                batch.source.mask,
                batch.target.padded,
            )
        else:
            return self.greedy_decode(
                encoded,
                batch.source.mask,
                batch.target.padded if batch.has_target else None,
            )

    def greedy_decode(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached END up to a
        specified length depending on the `target` args.

        Args:
            source_encoded (torch.Tensor): encoded source symbols of shape
                B x src_len x (encoder_hidden * num_directions)
            source_mask (torch.Tensor): mask.
            target (torch.Tensor, optional): target symbols; if provided
                decoding continues until this length is reached.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the loss and predictions of
                shape B x pred_seq_len.
        """
        batch_size = source_mask.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        emissions, transitions, state = self.decode_step(
            source_encoded, source_mask, symbol, state
        )
        likelihood = transitions[:, 0].unsqueeze(1)
        symbol = self._greedy_step(likelihood, emissions)
        predictions = [symbol]
        if target is None:
            max_num_steps = self.max_target_length
            # Tracks when each sequence has decoded an END.
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            max_num_steps = target.size(1)
        # We already did one step.
        for t in range(max_num_steps - 1):
            emissions, transitions, state = self.decode_step(
                source_encoded,
                source_mask,
                symbol.unsqueeze(1),
                state,
            )
            likelihood = self._transitions(likelihood, transitions)
            symbol = self._greedy_step(likelihood, emissions)
            predictions.append(symbol)
            if target is None:
                final = torch.logical_or(final, symbol == special.END_IDX)
                if final.all():
                    break
            likelihood = self._emissions(likelihood, emissions, symbol)
        # -> B x seq_len.
        predictions = torch.stack(predictions, dim=1)
        return self._scale(likelihood, t + 1), predictions

    @classmethod
    def _emissions(
        cls,
        likelihood: torch.Tensor,
        emissions: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        """Adds emission to likelihood.

        Args:
            likelihood (torch.Tensor): probabilities of shape
                B x 1 x src_len.
            emissions (torch.Tensor): slice of emission probabilities of shape
                B x src_len x vocab_size.
            symbol (torch.Tensor): slice of symbols of shape B.

        Returns:
            torch.Tensor: probabilities of shape B x 1 x src_len.
        """
        return likelihood + cls._gather_at_idx(emissions, symbol)

    @staticmethod
    def _gather_at_idx(
        emissions: torch.Tensor, symbol: torch.Tensor
    ) -> torch.Tensor:
        """Collects probability of the symbol across all states.

        To calculate the final emission probability, the pseudo-HMM
        graph needs to aggregate the final emission probabilities of
        target symbols across all potential hidden states in the emissions.

        Args:
            emissions (torch.Tensor): slice of emission probabilities of
                shape B x src_len.
            symbol (torch.Tensor): target symbol to poll probabilities for
                shape B.

        Returns:
            torch.Tensor: emission probabilities of the symbol for each hidden
                state, of size B x 1 x src_len.
        """
        batch_size = emissions.size(0)
        src_seq_len = emissions.size(1)
        idx = symbol.view(-1, 1).expand(batch_size, src_seq_len).unsqueeze(-1)
        output = torch.gather(emissions, -1, idx).view(
            batch_size, 1, src_seq_len
        )
        idx = idx.view(batch_size, 1, src_seq_len)
        pad_mask = (idx != special.PAD_IDX).float()
        return output * pad_mask

    @staticmethod
    def _greedy_step(
        likelihood: torch.Tensor,
        emissions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding of current timestep.

        Args:
            likelihood (torch.Tensor): probabilities of shape B x 1 x src_len.
            emissions (torch.Tensor): slice of emission probabilities of shape
                B x src_len x vocab_size.

        Returns:
            torch.Tensor: symbols of shape B.
        """
        likelihood = likelihood + emissions.transpose(1, 2)
        return torch.argmax(likelihood.logsumexp(dim=2), dim=1)

    @staticmethod
    def _scale(likelihood: torch.Tensor, size: int) -> torch.Tensor:
        """Scales likelihood to compute loss.

        Args:
            likelihood (torch.Tensor): probailities of shape
                B x 1 x src_len.
            size (int).

        Returns:
            torch.Tensor: scaled loss.
        """
        return -likelihood.logsumexp(dim=2).mean() / size

    @staticmethod
    def _transitions(
        likelihood: torch.Tensor, transitions: torch.Tensor
    ) -> torch.Tensor:
        """Adds transitions to likelihood.

        Args:
            likelihood (torch.Tensor): probabilities of shape
                B x 1 x src_len.
            transitions (torch.Tensor): slice of transition probabilities
                of shape B x src_len x src_len.

        Returns:
            torch.Tensor: probabilities of shape B x 1 x src_len.
        """
        return torch.logsumexp(
            likelihood + transitions.transpose(1, 2),
            dim=2,
            keepdim=True,
        ).transpose(1, 2)

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        _, predictions = self(batch)
        return predictions

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        _, predictions = self(batch)
        self._update_metrics(predictions, batch.target.padded)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        loss = self(batch)
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
        loss, predictions = self(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        self._update_metrics(predictions, batch.target.padded)

    @property
    def decoder_input_size(self) -> int:
        # We concatenate along the encoding dimension.
        if self.has_features_encoder:
            return (
                self.source_encoder.output_size
                + self.features_encoder.output_size
            )
        else:
            return self.source_encoder.output_size

    @abc.abstractmethod
    def get_decoder(self) -> modules.HardAttentionRNNDecoder: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class HardAttentionGRUModel(HardAttentionRNNModel):
    """Hard attention with GRU backend."""

    def get_decoder(self):
        if self.attention_context > 0:
            return modules.ContextHardAttentionGRUDecoder(
                attention_context=self.attention_context,
                decoder_input_size=self.decoder_input_size,
                dropout=self.decoder_dropout,
                embedding_size=self.embedding_size,
                hidden_size=self.decoder_hidden_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )
        else:
            return modules.HardAttentionGRUDecoder(
                decoder_input_size=self.decoder_input_size,
                dropout=self.decoder_dropout,
                embedding_size=self.embedding_size,
                hidden_size=self.decoder_hidden_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )

    @property
    def name(self) -> str:
        return "hard attention GRU"


class HardAttentionLSTMModel(HardAttentionRNNModel):
    """Hard attention with LSTM backend."""

    def get_decoder(self):
        if self.attention_context > 0:
            return modules.ContextHardAttentionLSTMDecoder(
                attention_context=self.attention_context,
                decoder_input_size=self.decoder_input_size,
                dropout=self.decoder_dropout,
                hidden_size=self.decoder_hidden_size,
                embedding_size=self.embedding_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )
        else:
            return modules.HardAttentionLSTMDecoder(
                decoder_input_size=self.decoder_input_size,
                dropout=self.decoder_dropout,
                embedding_size=self.embedding_size,
                hidden_size=self.decoder_hidden_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )

    @property
    def name(self) -> str:
        return "hard attention LSTM"
