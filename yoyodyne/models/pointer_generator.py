"""Pointer-generator model classes."""

from typing import Optional, Tuple

import torch
from torch import nn

from . import attention, generation_probability, lstm
from .. import batches


class Error(Exception):
    pass


class PointerGeneratorLSTMEncoderDecoderNoFeatures(lstm.LSTMEncoderDecoder):
    """Pointer-generator model with an LSTM backend and no features.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    # Constructed inside __init__.
    source_attention: attention.Attention
    geneneration_probability: generation_probability.GenerationProbability

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
        self.classifier = nn.Linear(3 * self.hidden_size, self.output_size)
        self.generation_probability = (
            generation_probability.GenerationProbability(  # noqa: E501
                self.embedding_size, self.hidden_size, encoder_size
            )
        )

    def _check_layer_sizes(self):
        """Checks that encoder and decoder layers are the same number.

        Raises:
            Error.
        """
        if self.encoder_layers != self.decoder_layers:
            msg = "encoder_layers needs to be the same as decoder_layers."
            msg += f" {self.encoder_layers} != {self.decoder_layers}."
            raise Error(msg)

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
        packed_outs, (H, C) = encoder(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Sums over directions, keeping layers.
        # -> num_layers x B x hidden_size.
        H = H.view(
            self.encoder_layers, self.num_directions, H.size(1), H.size(2)
        ).sum(axis=1)
        C = C.view(
            self.encoder_layers, self.num_directions, C.size(1), C.size(2)
        ).sum(axis=1)
        return encoded, (H, C)

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
        # -> B x 1 x output_size.
        ptr_probs = torch.zeros(
            symbol.size(0),
            self.output_size,
            device=self.device,
            dtype=source_attention_weights.dtype,
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x output_size.
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
            feature_enc (torch.Tensor): batch of encoded feaure symbols.
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
            # pred: B x 1 x output_size.
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
        # -> B x seq_len x output_size.
        predictions = predictions.transpose(0, 1)
        return predictions


class PointerGeneratorLSTMEncoderDecoderFeatures(
    PointerGeneratorLSTMEncoderDecoderNoFeatures
):
    """Pointer-generator model with an LSTM backend.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    # Constructed inside __init__.
    feature_attention: attention.Attention
    feature_embeddings: nn.Embedding
    feature_encoder: nn.LSTM
    linear_h: nn.Linear
    linear_c: nn.Linear

    def __init__(self, *args, **kwargs):
        """Initializes the pointer-generator model with an LSTM backend."""
        super().__init__(*args, **kwargs)
        self._check_layer_sizes()
        # We use the inherited defaults for the source embeddings/encoder.
        self.feature_embeddings = self.init_embeddings(
            self.features_vocab_size, self.embedding_size, self.pad_idx
        )
        self.feature_encoder = nn.LSTM(
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
        self.feature_attention = attention.Attention(
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
        self.classifier = nn.Linear(5 * self.hidden_size, self.output_size)
        # Overrides GenerationProbability to have larger hidden_size.
        self.generation_probability = (
            generation_probability.GenerationProbability(  # noqa: E501
                self.embedding_size, self.hidden_size, 2 * encoder_size
            )
        )

    def encode(
        self,
        source: batches.PaddedTensor,
        embeddings: nn.Embedding,
        encoder: torch.nn.LSTM,
    ) -> torch.Tensor:
        """Encodes the input with the TransformerEncoder.

        We pass the encoder as an argument to enable use of this function
        with multiple encoders in derived classes.

        Args:
            source (batches.PaddedTensor).
            embedding (torch.nn.Embedding).
            encoder (torch.nn.LSTM).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = embeddings(source.padded)
        embedded = self.dropout_layer(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source.lengths(), batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim,
        # (D*layers x B x hidden_size, D*layers x B x hidden_size).
        packed_outs, (H, C) = encoder(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Sums over directions, keeping layers.
        # -> num_layers x B x hidden_size.
        H = H.view(
            self.encoder_layers, self.num_directions, H.size(1), H.size(2)
        ).sum(axis=1)
        C = C.view(
            self.encoder_layers, self.num_directions, C.size(1), C.size(2)
        ).sum(axis=1)
        return encoded, (H, C)

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        feature_enc: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).
            feature_enc (torch.Tensor).
            feature_mask (torch.Tensor).

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
        feature_context, _ = self.feature_attention(
            last_h.transpose(0, 1), feature_enc, feature_mask
        )
        # -> B x 1 x 4*hidden_size.
        context = torch.cat([source_context, feature_context], dim=2)
        # num_layers x B x hidden_size
        _, (h, c) = self.decoder(
            torch.cat((embedded, context), 2), (last_h, last_c)
        )
        # -> B x 1 x hidden_size
        hidden = h[-1, :, :].unsqueeze(1)
        output_probs = self.classifier(torch.cat([hidden, context], dim=2))
        # Ordinary softmax, log will be taken at the end.
        output_probs = nn.functional.softmax(output_probs, dim=2)
        # -> B x 1 x output_size.
        ptr_probs = torch.zeros(
            symbol.size(0),
            self.output_size,
            device=self.device,
            dtype=source_attention_weights.dtype,
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x output_size.
        ptr_probs.scatter_add_(
            2, source_indices.unsqueeze(1), source_attention_weights
        )
        # Probability of generating (from output_probs).
        gen_probs = self.generation_probability(
            context, hidden, embedded
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
        feature_enc: torch.Tensor,
        feature_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
            batch_size (int).
            decoder_hiddens (torch.Tensor): .
            source_indices (torch.Tensor): Indices of the input for
                calculating pointer weights.
            source_enc (torch.Tensor): batch of encoded input symbols.
            source_mask (torch.Tensor): mask for the batch of encoded
                input symbols.
            feature_enc (torch.Tensor): batch of encoded feaure symbols.
            feature_mask (torch.Tensor): mask for the batch of encoded
                feature symbols.
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
            # pred: B x 1 x output_size.
            output, decoder_hiddens = self.decode_step(
                decoder_input,
                decoder_hiddens,
                source_indices,
                source_enc,
                source_mask,
                feature_enc,
                feature_mask,
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
        assert batch.has_features
        source_encoded, (h_source, c_source) = self.encode(
            batch.source, self.source_embeddings, self.encoder
        )
        features_encoded, (h_features, c_features) = self.encode(
            batch.features, self.feature_embeddings, self.feature_encoder
        )
        h_0 = self.linear_h(torch.cat([h_source, h_features], dim=2))
        c_0 = self.linear_c(torch.cat([c_source, c_features], dim=2))
        if self.beam_width is not None and self.beam_width > 1:
            # predictions = self.beam_decode(
            #     batch_size, x_mask, encoder_out, beam_width=self.beam_width
            # )
            raise NotImplementedError
        else:
            predictions = self.decode(
                len(batch),
                (h_0, c_0),
                batch.source.padded,
                source_encoded,
                batch.source.mask,
                features_encoded,
                batch.features.mask,
                self.teacher_forcing if self.training else False,
                batch.target.padded if batch.target else None,
            )
        # -> B x seq_len x output_size.
        predictions = predictions.transpose(0, 1)
        return predictions
