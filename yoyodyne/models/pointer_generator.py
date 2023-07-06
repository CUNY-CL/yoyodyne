"""Pointer-generator model classes."""

import math
from typing import Optional, Tuple

import torch
from torch import nn

from .. import batches
from . import lstm, modules


class Error(Exception):
    pass


class GenerationProbability(nn.Module):
    """Calculates the generation probability for a pointer generator."""

    stdev = 1 / math.sqrt(100)

    W_attention: nn.Linear
    W_hs: nn.Linear
    W_inp: nn.Linear
    bias: nn.Parameter

<<<<<<< HEAD
    def __init__(
        self, embedding_size: int, hidden_size: int, attention_size: int
    ):
        """Initializes the generation probability operator.

        Args:
            embedding_size (int): embedding dimensions.
            hidden_size (int): decoder hidden state dimensions.
            attention_size (int): dimensions of combined encoder attentions.
        """
        super().__init__()
        self.W_attention = nn.Linear(attention_size, 1, bias=False)
        self.W_hs = nn.Linear(hidden_size, 1, bias=False)
        self.W_inp = nn.Linear(embedding_size, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-self.stdev, self.stdev)
=======
    def __init__(self, *args, **kwargs):
        """Initializes the pointer-generator model with an LSTM backend."""
        super().__init__(*args, **kwargs)
        self._check_layer_sizes()
        # We use the inherited defaults for the source embeddings/encoder.
        encoder_size = self.hidden_size * self.num_directions
        self.source_attention = attention.Attention(
            encoder_size, self.hidden_size
        )
        # Overrides classifier to take larger input.
        self.classifier = nn.Linear(
            3 * self.hidden_size, self.target_vocab_size
        )
        self.generation_probability = (
            generation_probability.GenerationProbability(
                self.embedding_size, self.hidden_size, encoder_size
            )
        )  # noqa: E501

    def _check_layer_sizes(self) -> None:
        """Checks that encoder and decoder layers are the same number.

        Raises:
            Error.
        """
        if self.encoder_layers != self.decoder_layers:
            raise Error(
                f"The number of encoder layers ({self.encoder_layers}) and "
                f"decoder layers ({self.decoder_layers}) must match"
            )

    def encode(
        self,
        source: batches.PaddedTensor,
        encoder: torch.nn.LSTM,
    ) -> torch.Tensor:
        """Encodes the input.

        Args:
            source (batches.PaddedTensor).
            encoder (torch.nn.LSTM).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.source_embeddings(source.padded)
        embedded = self.dropout_layer(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source.lengths(), batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim,
        # (D*layers x B x hidden_size, D*layers x B x hidden_size)
        packed_outs, (h, c) = encoder(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Sums over directions, keeping layers.
        # -> num_layers x B x hidden_size.
        h = h.view(
            self.encoder_layers, self.num_directions, h.size(1), h.size(2)
        ).sum(axis=1)
        c = c.view(
            self.encoder_layers, self.num_directions, c.size(1), c.size(2)
        ).sum(axis=1)
        return encoded, (h, c)

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h, last_c = last_hiddens
        source_context, source_attention_weights = self.source_attention(
            last_h.transpose(0, 1), source_enc, source_mask
        )
        _, (h, c) = self.decoder(
            torch.cat((embedded, source_context), 2), (last_h, last_c)
        )
        # -> B x 1 x hidden_size
        hidden = h[-1, :, :].unsqueeze(1)
        output_probs = self.classifier(
            torch.cat([hidden, source_context], dim=2)
        )
        # Ordinary softmax, log will be taken at the end.
        output_probs = nn.functional.softmax(output_probs, dim=2)
        # -> B x 1 x target_vocab_size.
        ptr_probs = torch.zeros(
            symbol.size(0),
            self.target_vocab_size,
            device=self.device,
            dtype=source_attention_weights.dtype,
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x target_vocab_size.
        ptr_probs.scatter_add_(
            2, source_indices.unsqueeze(1), source_attention_weights
        )
        # Probability of generating (from output_probs).
        gen_probs = self.generation_probability(
            source_context, hidden, embedded
        ).unsqueeze(2)
        gen_scores = gen_probs * output_probs
        ptr_scores = (1 - gen_probs) * ptr_probs
        scores = gen_scores + ptr_scores
        # Puts scores in log space.
        scores = torch.log(scores)
        return scores, (h, c)

    def decode(
        self,
        batch_size: int,
        decoder_hiddens: torch.Tensor,
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
            batch_size (int).
            decoder_hiddens (torch.Tensor): .
            source_indices (torch.Tensor): Indices of the input for calculating
                pointer weights.
            source_enc (torch.Tensor): batch of encoded input symbols.
            source_mask (torch.Tensor): mask for the batch of encoded input
                symbols.
            features_enc (torch.Tensor): batch of encoded features symbols.
            teacher_forcing (bool): Whether or not to decode
                with teacher forcing.
            target (torch.Tensor, optional): target symbols;  we
                decode up to `len(target)` symbols. If it is None, then we
                decode up to `self.max_target_length` symbols.

        Returns:
            torch.Tensor.
        """
        # Feeds in the first decoder input, as a start tag.
        # -> B x 1
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
            # pred: B x 1 x target_vocab_size.
            output, decoder_hiddens = self.decode_step(
                decoder_input,
                decoder_hiddens,
                source_indices,
                source_enc,
                source_mask,
            )
            predictions.append(output.squeeze(1))
            # In teacher forcing mode the next input is the gold symbol
            # for this step.
            if teacher_forcing:
                assert (
                    target is not None
                ), "Teacher forcing requested but no target provided"
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise we pass the top pred to the next timestep
            # (i.e., student forcing, greedy decoding).
            else:
                decoder_input = self._get_predicted(output)
                # Tracks which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all batches predicted an EOS symbol.
                # If we have a target (and are thus computing loss),
                # we only break when we have decoded at least the the
                # same number of steps as the target length.
                if finished.all():
                    if target is None or decoder_input.size(-1) >= target.size(
                        -1
                    ):
                        break
        predictions = torch.stack(predictions)
        return predictions
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5

    def forward(
        self,
        h_attention: torch.Tensor,
        decoder_hs: torch.Tensor,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        # TODO(Adamits): improve documentation.
        """Computes Wh * ATTN_t + Ws * HIDDEN_t + Wy * Y_{t-1} + b.

        Args:
            h_attention (torch.Tensor): combined context vector over source and
                features of shape B x 1 x attention_size.
            decoder_hs (torch.Tensor): decoder hidden state of shape
                B x 1 x hidden_size.
            inp (torch.Tensor): decoder input of shape B x 1 x embedding_size.

        Returns:
            (torch.Tensor): generation probability of shape B.
        """
<<<<<<< HEAD
        # -> B x 1 x 1.
        p_gen = self.W_attention(h_attention) + self.W_hs(decoder_hs)
        p_gen += self.W_inp(inp) + self.bias.expand(h_attention.size(0), 1, -1)
        # -> B.
        p_gen = torch.sigmoid(p_gen.squeeze(1))
        return p_gen
=======
        source_encoded, (h_source, c_source) = self.encode(
            batch.source, self.encoder
        )
        if self.beam_width is not None and self.beam_width > 1:
            # predictions = self.beam_decode(
            #     batch_size, x_mask, encoder_out, beam_width=self.beam_width
            # )
            raise NotImplementedError
        else:
            predictions = self.decode(
                len(batch),
                (h_source, c_source),
                batch.source.padded,
                source_encoded,
                batch.source.mask,
                self.teacher_forcing if self.training else False,
                batch.target.padded if batch.target else None,
            )
        # -> B x seq_len x target_vocab_size.
        predictions = predictions.transpose(0, 1)
        return predictions
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5


class PointerGeneratorLSTMEncoderDecoder(lstm.LSTMEncoderDecoder):
    """Pointer-generator model with an LSTM backend and no features.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    # Constructed inside __init__.
<<<<<<< HEAD
    geneneration_probability: GenerationProbability
=======
    features_attention: attention.Attention
    features_embeddings: nn.Embedding
    features_encoder: nn.LSTM
    linear_h: nn.Linear
    linear_c: nn.Linear
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5

    def __init__(self, *args, **kwargs):
        """Initializes the pointer-generator model with an LSTM backend."""
        super().__init__(*args, **kwargs)
        self._check_layer_sizes()
        # We use the inherited defaults for the source embeddings/encoder.
<<<<<<< HEAD
        # Overrides classifier to take larger input.
        if not self.has_feature_encoder:
            self.classifier = nn.Linear(
                self.hidden_size + self.source_encoder.output_size,
                self.output_size,
            )
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.hidden_size,
                self.source_encoder.output_size,
            )
        else:
            self.merge_h = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.merge_c = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.feature_attention = modules.attention.Attention(
                self.feature_encoder.output_size, self.hidden_size
            )
            self.classifier = nn.Linear(
                self.hidden_size
                + self.source_encoder.output_size
                + self.feature_encoder.output_size,
                self.output_size,
            )
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.hidden_size,
                self.source_encoder.output_size
                + self.feature_encoder.output_size,
            )
=======
        self.features_embeddings = self.init_embeddings(
            self.features_vocab_size, self.embedding_size, self.pad_idx
        )
        self.features_encoder = nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.encoder_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        # Initializes the decoder.
        self.linear_h = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear_c = nn.Linear(2 * self.hidden_size, self.hidden_size)
        encoder_size = self.hidden_size * self.num_directions
        self.features_attention = attention.Attention(
            encoder_size, self.hidden_size
        )
        # Overrides decoder to be larger.
        self.decoder = nn.LSTM(
            2 * encoder_size + self.embedding_size,
            self.hidden_size,
            dropout=self.dropout,
            num_layers=self.decoder_layers,
            batch_first=True,
        )
        # Overrides classifier to take larger input.
        self.classifier = nn.Linear(
            5 * self.hidden_size, self.target_vocab_size
        )
        # Overrides GenerationProbability to have larger hidden_size.
        self.generation_probability = (
            generation_probability.GenerationProbability(
                self.embedding_size, self.hidden_size, 2 * encoder_size
            )
        )  # noqa: E501

    def encode(
        self,
        source: batches.PaddedTensor,
        embeddings: nn.Embedding,
        encoder: torch.nn.LSTM,
    ) -> torch.Tensor:
        """Encodes the input with the TransformerEncoder.
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5

    def get_decoder(self) -> modules.lstm.LSTMAttentiveDecoder:
        return modules.lstm.LSTMAttentiveDecoder(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            decoder_input_size=self.source_encoder.output_size
            + self.feature_encoder.output_size
            if self.has_feature_encoder
            else self.source_encoder.output_size,
            num_embeddings=self.output_size,
            dropout=self.dropout,
            bidirectional=False,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
            attention_input_size=self.source_encoder.output_size,
        )

    def _check_layer_sizes(self) -> None:
        """Checks that encoder and decoder layers are the same number.

<<<<<<< HEAD
        Raises:
            Error.
        """
        if self.encoder_layers != self.decoder_layers:
            msg = "encoder_layers needs to be the same as decoder_layers."
            msg += f" {self.encoder_layers} != {self.decoder_layers}."
            raise Error(msg)
=======
        Returns:
            torch.Tensor, (torch.Tensor, torch.Tensor).
        """
        embedded = embeddings(source.padded)
        embedded = self.dropout_layer(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source.lengths(), batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim,
        # (D*layers x B x hidden_size, D*layers x B x hidden_size).
        packed_outs, (h, c) = encoder(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Sums over directions, keeping layers.
        # -> num_layers x B x hidden_size.
        h = h.view(
            self.encoder_layers, self.num_directions, h.size(1), h.size(2)
        ).sum(axis=1)
        c = c.view(
            self.encoder_layers, self.num_directions, c.size(1), c.size(2)
        ).sum(axis=1)
        return encoded, (h, c)
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
<<<<<<< HEAD
        feature_enc: Optional[torch.Tensor] = None,
        feature_mask: Optional[torch.Tensor] = None,
=======
        features_enc: torch.Tensor,
        features_mask: torch.Tensor,
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).
<<<<<<< HEAD
            feature_enc (Optional[torch.Tensor]).
            feature_mask (Optional[torch.Tensor]).
=======
            features_enc (torch.Tensor).
            features_mask (torch.Tensor).
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
<<<<<<< HEAD
        embedded = self.decoder.embed(symbol)
        last_h0, last_c0 = last_hiddens
        context, attention_weights = self.decoder.attention(
            last_h0.transpose(0, 1), source_enc, source_mask
        )
        if self.has_feature_encoder:
            feature_context, _ = self.feature_attention(
                last_h0.transpose(0, 1), feature_enc, feature_mask
            )
            # -> B x 1 x 4*hidden_size.
            context = torch.cat([context, feature_context], dim=2)
        _, (h, c) = self.decoder.module(
            torch.cat((embedded, context), 2), (last_h0, last_c0)
=======
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h, last_c = last_hiddens
        source_context, source_attention_weights = self.source_attention(
            last_h.transpose(0, 1), source_enc, source_mask
        )
        features_context, _ = self.features_attention(
            last_h.transpose(0, 1), features_enc, features_mask
        )
        # -> B x 1 x 4*hidden_size.
        context = torch.cat((source_context, features_context), dim=2)
        # num_layers x B x hidden_size
        _, (h, c) = self.decoder(
            torch.cat((embedded, context), 2), (last_h, last_c)
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5
        )
        # -> B x 1 x hidden_size
        hidden = h[-1, :, :].unsqueeze(1)
        output_probs = self.classifier(torch.cat([hidden, context], dim=2))
        output_probs = nn.functional.softmax(output_probs, dim=2)
        # -> B x 1 x target_vocab_size.
        ptr_probs = torch.zeros(
            symbol.size(0),
            self.target_vocab_size,
            device=self.device,
            dtype=attention_weights.dtype,
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x target_vocab_size.
        ptr_probs.scatter_add_(
            2, source_indices.unsqueeze(1), attention_weights
        )
        # Probability of generating (from output_probs).
        gen_probs = self.generation_probability(
            context, hidden, embedded
        ).unsqueeze(2)
        ptr_scores = (1 - gen_probs) * ptr_probs
        gen_scores = gen_probs * output_probs
        scores = gen_scores + ptr_scores
        # Puts scores in log space.
        scores = torch.log(scores)
        return scores, (h, c)

    def decode(
        self,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
<<<<<<< HEAD
        source_indices: torch.Tensor,
        decoder_hiddens: torch.Tensor,
=======
        features_enc: torch.Tensor,
        features_mask: torch.Tensor,
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5
        teacher_forcing: bool,
        feature_enc: Optional[torch.Tensor] = None,
        feature_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
            source_enc (torch.Tensor): batch of encoded input symbols.
<<<<<<< HEAD
            source_mask (torch.Tensor): mask for the batch of encoded input
                symbols.
            source_indices (torch.Tensor): Indices of the input for calculating
                pointer weights.
            decoder_hiddens (torch.Tensor): .
=======
            source_mask (torch.Tensor): mask for the batch of encoded
                input symbols.
            features_enc (torch.Tensor): batch of encoded feaure symbols.
            features_mask (torch.Tensor): mask for the batch of encoded
                features symbols.
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5
            teacher_forcing (bool): Whether or not to decode
                with teacher forcing.
            feature_enc (torch.Tensor, optional): batch of encoded feaure
                symbols.
            feature_mask (torch.Tensor, optional): mask for the batch of
                encoded feature symbols.
            target (torch.Tensor, optional): target symbols;  we
                decode up to `len(target)` symbols. If it is None, then we
                decode up to `self.max_target_length` symbols.

        Returns:
            torch.Tensor.
        """
        batch_size = source_enc.shape[0]
        # Feeds in the first decoder input, as a start tag.
        # -> B x 1
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
            # pred: B x 1 x target_vocab_size.
            output, decoder_hiddens = self.decode_step(
                decoder_input,
                decoder_hiddens,
                source_indices,
                source_enc,
                source_mask,
<<<<<<< HEAD
                feature_enc=feature_enc,
                feature_mask=feature_mask,
=======
                features_enc,
                features_mask,
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5
            )
            predictions.append(output.squeeze(1))
            # In teacher forcing mode the next input is the gold symbol
            # for this step.
            if teacher_forcing:
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise we pass the top pred to the next timestep
            # (i.e., student forcing, greedy decoding).
            else:
                decoder_input = self._get_predicted(output)
                # Tracks which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all batches predicted an EOS symbol.
                # If we have a target (and are thus computing loss),
                # we only break when we have decoded at least the the
                # same number of steps as the target length.
                if finished.all():
                    if target is None or decoder_input.size(-1) >= target.size(
                        -1
                    ):
                        break
        predictions = torch.stack(predictions)
        return predictions

    def forward(
        self,
        batch: batches.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (batches.PaddedBatch).

        Returns:
            torch.Tensor.
        """
<<<<<<< HEAD
        encoder_output = self.source_encoder(batch.source)
        source_encoded = encoder_output.output
        if encoder_output.has_hiddens:
            h_source, c_source = encoder_output.hiddens
            last_hiddens = self._reshape_hiddens(
                h_source,
                c_source,
                self.source_encoder.layers,
                self.source_encoder.num_directions,
            )
=======
        assert batch.has_features
        source_encoded, (h_source, c_source) = self.encode(
            batch.source, self.source_embeddings, self.encoder
        )
        features_encoded, (h_features, c_features) = self.encode(
            batch.features, self.features_embeddings, self.features_encoder
        )
        h_0 = self.linear_h(torch.cat([h_source, h_features], dim=2))
        c_0 = self.linear_c(torch.cat([c_source, c_features], dim=2))
        if self.beam_width is not None and self.beam_width > 1:
            # predictions = self.beam_decode(
            #     batch_size, x_mask, encoder_out, beam_width=self.beam_width
            # )
            raise NotImplementedError
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5
        else:
            last_hiddens = self.init_hiddens(
                len(batch), self.source_encoder.layers
            )
<<<<<<< HEAD
        if not self.has_feature_encoder:
            if self.beam_width is not None and self.beam_width > 1:
                # predictions = self.beam_decode(
                # batch_size, x_mask, encoder_out, beam_width=self.beam_width
                # )
                raise NotImplementedError
            else:
                predictions = self.decode(
                    source_encoded,
                    batch.source.mask,
                    batch.source.padded,
                    last_hiddens,
                    self.teacher_forcing if self.training else False,
                    target=batch.target.padded if batch.target else None,
                )
        else:
            features_encoder_output = self.feature_encoder(batch.features)
            features_encoded = features_encoder_output.output
            if features_encoder_output.has_hiddens:
                h_features, c_features = features_encoder_output.hiddens
                h_features, c_features = self._reshape_hiddens(
                    h_features,
                    c_features,
                    self.feature_encoder.layers,
                    self.feature_encoder.num_directions,
                )
            else:
                h_features, c_features = self.init_hiddens(
                    len(batch), self.source_encoder.layers
                )
                predictions = self.decode(
                    source_encoded,
                    batch.source.mask,
                    batch.source.padded,
                    last_hiddens,
                    self.teacher_forcing if self.training else False,
                    feature_enc=features_encoded,
                    feature_mask=batch.features.mask,
                    target=batch.target.padded if batch.target else None,
                )
=======
        # -> B x seq_len x target_vocab_size.
        predictions = predictions.transpose(0, 1)
>>>>>>> c148eb8cd29f264b02567689ae7703febf712ec5
        return predictions

    @staticmethod
    def _reshape_hiddens(
        H: torch.Tensor, C: torch.Tensor, layers: int, num_directions: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H = H.view(layers, num_directions, H.size(1), H.size(2)).sum(axis=1)
        C = C.view(layers, num_directions, C.size(1), C.size(2)).sum(axis=1)
        return (H, C)
