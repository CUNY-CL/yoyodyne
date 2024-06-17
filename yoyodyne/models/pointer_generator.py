"""Pointer-generator model classes."""

import numpy
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from .. import data
from . import lstm, modules, transformer


class Error(Exception):
    pass


class GenerationProbability(nn.Module):
    """Calculates the generation probability for a pointer generator."""

    stdev = 1 / numpy.sqrt(100)

    W_attention: nn.Linear
    W_hs: nn.Linear
    W_emb: nn.Linear
    bias: nn.Parameter

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
        self.W_emb = nn.Linear(embedding_size, 1, bias=False)
        self.W_hs = nn.Linear(hidden_size, 1, bias=False)
        self.W_attention = nn.Linear(attention_size, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-self.stdev, self.stdev)

    def forward(
        self,
        attention_context: torch.Tensor,
        decoder_hiddens: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the generation probability

        This is a function of the context vectors, decoder hidden states, and
        target embeddings, where each is first mapped to a scalar value by a
        learnable weight matrix.

        Args:
            attention_context (torch.Tensor): combined context vector over
                source and features of shape
                B x sequence_length x attention_size.
            decoder_hiddens (torch.Tensor): decoder hidden state of shape
                B x sequence_length x hidden_size.
            target_embeddings (torch.Tensor): decoder input of shape
                B x sequence_length x embedding_size.

        Returns:
            torch.Tensor: generation probability of shape B.
        """
        # -> B x sequence_length x 1.
        p_gen = self.W_attention(attention_context) + self.W_hs(
            decoder_hiddens
        )
        p_gen += self.W_emb(target_embeddings) + self.bias.expand(
            attention_context.size(0), 1, -1
        )
        # -> B x 1 x sequence_length.
        return torch.sigmoid(p_gen)


class PointerGenerator(nn.Module):
    """Base pointer generator"""

    # Constructed inside __init__.
    geneneration_probability: GenerationProbability

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        This overrides the loss function behavior in
        models.base.BaseEncoderDecoder because we need to use NLLLoss in
        order to factor the addition of two separate probability
        distributions. An NLLLoss-compatible implementation of label smoothing
        is also provided when label smoothing is requested.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        if not self.label_smoothing:
            return nn.NLLLoss(ignore_index=self.pad_idx)
        else:
            return self._smooth_nllloss

    def _smooth_nllloss(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the NLLLoss with a smoothing factor such that some
        proportion of the output distribution is replaced with a
        uniform distribution.

        After:

            https://github.com/NVIDIA/DeepLearningExamples/blob/
            8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/
            ConvNets/image_classification/smoothing.py#L18

        Args:
            predictions (torch.Tensor): tensor of prediction
                distribution of shape B x target_vocab_size x seq_len.
            target (torch.Tensor): tensor of golds of shape
                B x seq_len.

        Returns:
            torch.Tensor: loss.
        """
        # -> (B * seq_len) x target_vocab_size.
        predictions = predictions.transpose(1, 2).reshape(
            -1, self.target_vocab_size
        )
        # -> (B * seq_len) x 1.
        target = target.view(-1, 1)
        non_pad_mask = target.ne(self.pad_idx)
        # Gets the ordinary loss.
        nll_loss = -predictions.gather(dim=-1, index=target)[
            non_pad_mask
        ].mean()
        # Gets the smoothed loss.
        smooth_loss = -predictions.sum(dim=-1, keepdim=True)[
            non_pad_mask
        ].mean()
        smooth_loss = smooth_loss / self.target_vocab_size
        # Combines both according to label smoothing weight.
        loss = (1.0 - self.label_smoothing) * nll_loss
        loss.add_(self.label_smoothing * smooth_loss)
        return loss


class PointerGeneratorLSTMEncoderDecoder(
    lstm.LSTMEncoderDecoder, PointerGenerator
):
    """Pointer-generator model with an LSTM backend.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the pointer-generator model with an LSTM backend."""
        super().__init__(*args, **kwargs)
        self._check_layer_sizes()
        # We use the inherited defaults for the source embeddings/encoder.
        # Overrides classifier to take larger input.
        if not self.has_features_encoder:
            self.classifier = nn.Linear(
                self.hidden_size + self.source_encoder.output_size,
                self.target_vocab_size,
            )
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.hidden_size,
                self.source_encoder.output_size,
            )
        else:
            self.merge_h = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.merge_c = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.features_attention = modules.attention.Attention(
                self.features_encoder.output_size, self.hidden_size
            )
            self.classifier = nn.Linear(
                self.hidden_size
                + self.source_encoder.output_size
                + self.features_encoder.output_size,
                self.target_vocab_size,
            )
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.hidden_size,
                self.source_encoder.output_size
                + self.features_encoder.output_size,
            )

    def get_decoder(self) -> modules.lstm.LSTMAttentiveDecoder:
        return modules.lstm.LSTMAttentiveDecoder(
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
            attention_input_size=self.source_encoder.output_size,
        )

    def _check_layer_sizes(self) -> None:
        """Checks that encoder and decoder layers are the same number.

        Raises:
            Error.
        """
        if self.encoder_layers != self.decoder_layers:
            raise Error(
                "The number of encoder and decoder layers must match "
                f"({self.encoder_layers} != {self.decoder_layers})"
            )

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        features_enc: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).
            features_enc (Optional[torch.Tensor]).
            features_mask (Optional[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        embedded = self.decoder.embed(symbol)
        last_h0, last_c0 = last_hiddens
        source_context, attention_weights = self.decoder.attention(
            last_h0.transpose(0, 1), source_enc, source_mask
        )
        if self.has_features_encoder:
            features_context, _ = self.features_attention(
                last_h0.transpose(0, 1), features_enc, features_mask
            )
            # -> B x 1 x 4*hidden_size.
            context = torch.cat([source_context, features_context], dim=2)
        else:
            context = source_context
        _, (h, c) = self.decoder.module(
            torch.cat((embedded, context), 2), (last_h0, last_c0)
        )
        # -> B x 1 x hidden_size
        hidden = h[-1, :, :].unsqueeze(1)
        output_dist = self.classifier(torch.cat([hidden, context], dim=2))
        output_dist = nn.functional.softmax(output_dist, dim=2)
        # -> B x 1 x target_vocab_size.
        ptr_dist = torch.zeros(
            symbol.size(0),
            self.target_vocab_size,
            device=self.device,
            dtype=attention_weights.dtype,
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x target_vocab_size.
        ptr_dist.scatter_add_(
            2, source_indices.unsqueeze(1), attention_weights
        )
        # Probability of generating (from output_dist).
        gen_probs = self.generation_probability(context, hidden, embedded)
        scaled_ptr_dist = ptr_dist * (1 - gen_probs)
        scaled_output_dist = output_dist * gen_probs
        return torch.log(scaled_output_dist + scaled_ptr_dist), (h, c)

    def decode(
        self,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        source_indices: torch.Tensor,
        decoder_hiddens: torch.Tensor,
        teacher_forcing: bool,
        features_enc: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
            source_enc (torch.Tensor): batch of encoded input symbols.
            source_mask (torch.Tensor): mask for the batch of encoded input
                symbols.
            source_indices (torch.Tensor): indices of the input for calculating
                pointer weights.
            decoder_hiddens (torch.Tensor).
            teacher_forcing (bool): whether or not to decode with teacher
                forcing.
            features_enc (torch.Tensor, optional): batch of encoded feaure
                symbols.
            features_mask (torch.Tensor, optional): mask for the batch of
                encoded feature symbols.
            target (torch.Tensor, optional): target symbols; we decode up to
                `len(target)` symbols. If it is None, then we decode up to
                `self.max_target_length` symbols.

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
                features_enc=features_enc,
                features_mask=features_mask,
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
                # Breaks when all sequences have predicted an EOS symbol. If we
                # have a target (and are thus computing loss), we only break
                # when we have decoded at least the the same number of steps as
                # the target length.
                if finished.all():
                    if target is None or decoder_input.size(-1) >= target.size(
                        -1
                    ):
                        break
        predictions = torch.stack(predictions).transpose(0, 1)
        return predictions

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.
        """
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
        else:
            last_hiddens = self.init_hiddens(
                len(batch), self.source_encoder.layers
            )
        if not self.has_features_encoder:
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
            features_encoder_output = self.features_encoder(batch.features)
            features_encoded = features_encoder_output.output
            if features_encoder_output.has_hiddens:
                h_features, c_features = features_encoder_output.hiddens
                h_features, c_features = self._reshape_hiddens(
                    h_features,
                    c_features,
                    self.features_encoder.layers,
                    self.features_encoder.num_directions,
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
                features_enc=features_encoded,
                features_mask=batch.features.mask,
                target=batch.target.padded if batch.target else None,
            )
        return predictions

    @staticmethod
    def _reshape_hiddens(
        h: torch.Tensor, c: torch.Tensor, layers: int, num_directions: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = h.view(layers, num_directions, h.size(1), h.size(2)).sum(axis=1)
        c = c.view(layers, num_directions, c.size(1), c.size(2)).sum(axis=1)
        return h, c

    @property
    def name(self) -> str:
        return "pointer-generator LSTM"


class PointerGeneratorTransformerEncoderDecoder(
    PointerGenerator,
    transformer.TransformerEncoderDecoder,
):
    """Pointer-generator model with a transformer backend.

    After:
        Singer, A., and Kann, K. 2020. The NYU-CUBoulder Systems for
        SIGMORPHON 2020 Task 0 and Task 2. In Proceedings of the 17th
        SIGMORPHON Workshop on Computational Research in Phonetics, Phonology,
        and Morphology, pages 90–98.
    """

    # Model arguments.
    features_attention_heads: int

    def __init__(self, *args, features_attention_heads, **kwargs):
        """Initializes a pointer-generator model with transformer backend."""
        self.features_attention_heads = features_attention_heads
        super().__init__(*args, **kwargs)
        if not self.has_features_encoder:
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.embedding_size,
                self.source_encoder.output_size,
            )
        else:
            # Removes inherited features attention.
            self.features_attention = None
            self.generation_probability = GenerationProbability(  # noqa: E501
                self.embedding_size,
                self.embedding_size,
                self.embedding_size,
            )

    def get_decoder(self):
        return modules.transformer.TransformerPointerDecoder(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            num_embeddings=self.target_vocab_size,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            source_attention_heads=self.source_attention_heads,
            separate_features=self.has_features_encoder,
            features_attention_heads=self.features_attention_heads,
            max_source_length=self.max_source_length,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
        )

    def decode_step(
        self,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
        source_indices: torch.Tensor,
        target_tensor: torch.Tensor,
        target_mask: torch.Tensor,
        features_enc: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the decoder in one step.

        This will work on any sequence length, and returns output
        probabilities for all targets, meaning we can use this method for
        greedy decoding, wherein only a single new token is decoded at a time,
        or for teacher-forced training, wherein all tokens can be decoded in
        parallel with a diagonal mask.

        Args:
            encoder_outputs (torch.Tensor): encoded output representations.
            source_mask (torch.Tensor): mask for the encoded source tokens.
            source_indices (torch.Tensor): source token vocabulary ids.
            target_tensor (torch.Tensor): target token vocabulary ids.
            target_mask (torch.Tensor): mask for the target tokens.
            features_enc (Optional[torch.Tensor]): encoded features.
            features_mask (Optional[torch.Tensor]): mask for encoded features.

        Returns:
            torch.Tensor: output probabilities of the shape
                B x target_seq_len x target_vocab_size, where
                target_seq_len is inferred form the target_tensor.
        """
        decoder_output = self.decoder(
            encoder_outputs,
            source_mask,
            target_tensor,
            target_mask,
            features_memory=features_enc,
            features_memory_mask=features_mask,
        )
        target_embeddings = decoder_output.embeddings
        decoder_output = decoder_output.output
        # Outputs from multi-headed attention from each decoder step to
        # the encoded inputs.
        # Values have been averaged over each attention head.
        # -> B x tgt_seq_len x src_seq_len.
        mha_outputs = self.decoder.attention_output.outputs[0]
        # Clears the stored attention result.
        self.decoder.attention_output.clear()
        logits = self.classifier(decoder_output)
        output_dist = nn.functional.softmax(logits, dim=2)
        # -> B x target_seq_len x target_vocab_size.
        ptr_dist = torch.zeros(
            mha_outputs.size(0),
            mha_outputs.size(1),
            self.target_vocab_size,
            device=self.device,
            dtype=mha_outputs.dtype,
        )
        # Repeats the source indices for each target.
        # -> B x tgt_seq_len x src_seq_len.
        repeated_source_indices = source_indices.unsqueeze(1).repeat(
            1, mha_outputs.size(1), 1
        )
        # Scatters the attention weights onto the ptr_dist tensor at their
        # vocab indices in order to get outputs that match the indexing of the
        # generation probability.
        ptr_dist.scatter_add_(2, repeated_source_indices, mha_outputs)
        # A matrix of context vectors from applying attention to the encoder
        # representations w.r.t. each decoder step.
        context = torch.bmm(mha_outputs, encoder_outputs)
        # Probability of generating (from output_dist).
        gen_probs = self.generation_probability(
            context, decoder_output, target_embeddings
        )
        scaled_ptr_dist = ptr_dist * (1 - gen_probs)
        scaled_output_dist = output_dist * gen_probs
        return torch.log(scaled_output_dist + scaled_ptr_dist)

    def _decode_greedy(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        source_indices: torch.Tensor,
        targets: Optional[torch.Tensor],
        features_enc: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        Args:
            encoder_hidden (torch.Tensor): hidden states from the encoder.
            source_mask (torch.Tensor): mask for the encoded source tokens.
            source_indices (torch.Tensor): indices of the source symbols.
            targets (torch.Tensor, optional): the optional target tokens,
                which is only used for early stopping during validation
                if the decoder has predicted [EOS] for every sequence in
                the batch.
            features_enc (Optional[torch.Tensor]): encoded features.
            features_mask (Optional[torch.Tensor]): mask for encoded features.

        Returns:
            torch.Tensor: predictions from the decoder.
        """
        # The output distributions to be returned.
        outputs = []
        batch_size = encoder_hidden.size(0)
        # The predicted symbols at each iteration.
        predictions = [
            torch.tensor(
                [self.start_idx for _ in range(encoder_hidden.size(0))],
                device=self.device,
            )
        ]
        # Tracking when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size, device=self.device)
        for _ in range(self.max_target_length):
            target_tensor = torch.stack(predictions, dim=1)
            # Uses a dummy mask of all ones.
            target_mask = torch.ones_like(target_tensor, dtype=torch.float)
            target_mask = target_mask == 0
            scores = self.decode_step(
                encoder_hidden,
                source_mask,
                source_indices,
                target_tensor,
                target_mask,
                features_enc=features_enc,
                features_mask=features_mask,
            )
            last_output = scores[:, -1, :]
            outputs.append(last_output)
            # -> B x 1 x 1.
            _, pred = torch.max(last_output, dim=1)
            predictions.append(pred)
            # Updates to track which sequences have decoded an EOS.
            finished = torch.logical_or(
                finished, (predictions[-1] == self.end_idx)
            )
            # Breaks when all sequences have predicted an EOS symbol. If we
            # have a target (and are thus computing loss), we only break when
            # we have decoded at least the the same number of steps as the
            # target length.
            if finished.all():
                if targets is None or len(outputs) >= targets.size(-1):
                    break
        # -> B x seq_len x target_vocab_size.
        return torch.stack(outputs).transpose(0, 1)

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.
        """
        source_encoded = self.source_encoder(batch.source).output

        if self.training and self.teacher_forcing:
            assert (
                batch.target.padded is not None
            ), "Teacher forcing requested but no target provided"
            # Initializes the start symbol for decoding.
            starts = (
                torch.tensor(
                    [self.start_idx], device=self.device, dtype=torch.long
                )
                .repeat(batch.target.padded.size(0))
                .unsqueeze(1)
            )
            target_padded = torch.cat((starts, batch.target.padded), dim=1)
            target_mask = torch.cat(
                (starts == self.pad_idx, batch.target.mask), dim=1
            )
            features_encoded = None
            if self.has_features_encoder:
                features_encoder_output = self.features_encoder(batch.features)
                features_encoded = features_encoder_output.output
            if self.beam_width is not None and self.beam_width > 1:
                # predictions = self.beam_decode(
                # batch_size, x_mask, encoder_out, beam_width=self.beam_width
                # )
                raise NotImplementedError
            else:
                output = self.decode_step(
                    source_encoded,
                    batch.source.mask,
                    batch.source.padded,
                    target_padded,
                    target_mask,
                    features_enc=features_encoded,
                )
                output = output[:, :-1, :]  # Ignore EOS.
        else:
            features_encoded = None
            if self.has_features_encoder:
                features_encoder_output = self.features_encoder(batch.features)
                features_encoded = features_encoder_output.output
            # -> B x seq_len x output_size.
            output = self._decode_greedy(
                source_encoded,
                batch.source.mask,
                batch.source.padded,
                batch.target.padded if batch.target else None,
                features_enc=features_encoded,
            )
        return output

    @property
    def name(self) -> str:
        return "pointer-generator transformer"
