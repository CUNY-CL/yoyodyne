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
        super().__init__(*args, **kwargs)
        self.attention_context = attention_context
        self.enforce_monotonic = enforce_monotonic
        self.decoder = self.get_decoder()
        self.classifier = nn.Linear(
            self.decoder.output_size, self.target_vocab_size
        )
        self._log_model()
        self.save_hyperparameters(
            ignore=[
                "classifier",
                "decoder",
                "embeddings",
                "features_encoder",
                "source_encoder",
            ]
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
        """Computes loss.

        Following Wu & Cotterell this is scaled by max sequence length.

        Args:
            encoded (torch.Tensor): encoded source symbols
                of shape B x src_len x (encoder_hidden * num_directions).
            mask (torch.Tensor): mask.
            target (torch.Tensor): target symbols.

        Returns:
            torch.Tensor: loss.
        """
        batch_size = mask.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        likelihood = self._initial_likelihood(batch_size, encoded.size(1))
        for t in range(target.size(1)):
            emissions, transitions, state = self.decode_step(
                encoded, mask, symbol, state
            )
            likelihood = self._transitions(likelihood, transitions)
            likelihood = self._emissions(likelihood, emissions, target[:, t])
            symbol = target[:, t : t + 1]
        return -likelihood.logsumexp(dim=2).mean() / target.size(1)

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
        emissions = nn.functional.log_softmax(logits, dim=2)
        assert torch.allclose(
            emissions.exp().sum(dim=-1),
            torch.tensor(1.0, device=self.device),
        ), "emissions do not sum to 1"
        # Expands matrix for all time steps.
        if self.enforce_monotonic:
            transitions = self._apply_mono_mask(transitions)
            assert torch.allclose(
                transitions.exp().sum(dim=-1),
                torch.tensor(1.0, device=self.device),
            ), "transitions do not sum to 1"
        return emissions, transitions, state

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
        mask = (
            torch.ones_like(transitions[0], dtype=bool)
            .triu()
            .logical_not()
            .unsqueeze(0)
        )
        transitions = transitions.masked_fill(mask, -1e7)
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
                training, the loss tensor; otherwise, also returns the
                predictions of shape B x pred_seq_length.

        Raises:
            base.ConfigurationError: Features encoder specified but no feature
                column specified.
            base.ConfigurationError: Feature column specified but no feature
                encoder specified.
        """
        encoded = self.source_encoder(
            batch.source, self.embeddings, is_source=True
        )
        if self.has_features_encoder:
            if not batch.has_features:
                raise base.ConfigurationError(
                    "Features encoder specified but "
                    "no feature column specified"
                )
            features_encoded = self.features_encoder(
                batch.features, self.embeddings, is_source=False
            )
            features_encoded = features_encoded.mean(dim=1, keepdim=True)
            features_encoded = features_encoded.expand(-1, encoded.size(1), -1)
            encoded = torch.cat((encoded, features_encoded), dim=2)
        elif batch.has_features:
            raise base.ConfigurationError(
                "Feature column specified but no feature encoder specified"
            )
        if self.training:
            return self._loss(encoded, batch.source.mask, batch.target.tensor)
        predictions = self.greedy_decode(
            encoded,
            batch.source.mask,
            batch.target.tensor if batch.has_target else None,
        )
        if self.validating:
            loss = self._loss(encoded, batch.source.mask, batch.target.tensor)
            return loss, predictions
        else:
            return predictions

    def greedy_decode(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached END up to a
        specified length depending on the `target` args.

        Args:
            encoded (torch.Tensor): encoded source symbols of shape
                B x src_len x (encoder_hidden * num_directions)
            mask (torch.Tensor): mask.
            target (torch.Tensor, optional): target symbols; if provided
                decoding continues until this length is reached.

        Returns:
            torch.Tensor: predictions of shape B x pred_seq_len.
        """
        batch_size = mask.size(0)
        symbol = self.start_symbol(batch_size)
        state = self.decoder.initial_state(batch_size)
        likelihood = self._initial_likelihood(batch_size, encoded.size(1))
        predictions = []
        if target is None:
            max_num_steps = self.max_target_length
            # Tracks when each sequence has decoded an END.
            final = torch.zeros(batch_size, device=self.device, dtype=bool)
        else:
            max_num_steps = target.size(1)
        for t in range(max_num_steps):
            emissions, transitions, state = self.decode_step(
                encoded,
                mask,
                symbol,
                state,
            )
            likelihood = self._transitions(likelihood, transitions)
            symbol = torch.argmax(
                torch.logsumexp(likelihood.transpose(1, 2) + emissions, dim=1),
                dim=1,
                keepdim=True,
            )
            predictions.append(symbol.squeeze(1))
            likelihood = self._emissions(likelihood, emissions, symbol)
            likelihood = likelihood - likelihood.logsumexp(dim=2, keepdim=True)
            assert not torch.isnan(likelihood).any(), "NaN(s) in likelihood"
            assert (
                likelihood.logsumexp(dim=2).isfinite().all()
            ), "no alignment is reachable"
            if target is None:
                final = torch.logical_or(final, symbol == special.END_IDX)
                if final.all():
                    break
        # -> B x seq_len.
        return torch.stack(predictions, dim=1)

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

    def _initial_likelihood(
        self, batch_size: int, src_len: int
    ) -> torch.Tensor:
        likelihood = torch.full(
            (batch_size, 1, src_len), defaults.NEG_INF, device=self.device
        )
        likelihood[:, 0, 0] = 0.0
        return likelihood

    @staticmethod
    def _gather_at_idx(
        emissions: torch.Tensor, symbol: torch.Tensor
    ) -> torch.Tensor:
        """Collects log-probability of a specific symbol across all states.

        Args:
            emissions (torch.Tensor): emissions of shape
                B x src_len x vocab_size.
            symbol (torch.Tensor): target symbol(s) of shape B.

        Returns:
            torch.Tensor: log-probs of that symbol for each source state,
                of shape B x 1 x src_len.
        """
        src_len = emissions.size(1)
        # Reshapes symbol to B, src_len, 1 to gather from the vocab dimension.
        idx = symbol.view(-1, 1, 1).expand(-1, src_len, 1)
        # Gathers the log-probs: (B, src_len, 1)
        output = torch.gather(emissions, 2, idx).transpose(1, 2)
        # Masks out padding positions if necessary.
        pad_mask = symbol.view(-1, 1, 1) == special.PAD_IDX
        output.masked_fill_(pad_mask, 1e-7)
        return output

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
            likelihood.transpose(1, 2) + transitions, dim=1, keepdim=True
        )

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        return predictions

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        predictions = self(batch)
        self._update_metrics(predictions, batch.target.tensor)

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
        self._update_metrics(predictions, batch.target.tensor)

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
