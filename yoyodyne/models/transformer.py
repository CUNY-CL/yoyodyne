"""Transformer model classes."""

import math
from typing import Optional

import torch
from torch import nn

from .. import evaluators
from . import base, positional_encoding


class Error(Exception):
    pass


class TransformerEncoderDecoder(base.BaseEncoderDecoder):
    """Transformer encoder-decoder."""

    vocab_size: int
    hidden_size: int
    d_model: int
    attention_heads: int
    max_seq_len: int
    encoder_layers: int
    decoder_layers: int
    pad_idx: int
    optimizer: str
    beta1: float
    beta2: float
    warmup_steps: int
    learning_rate: float
    evaluator: evaluators.Evaluator
    scheduler: str
    start_idx: int
    end_idx: int
    embedding_size: int
    output_size: int
    dropout: float
    dropout_layer: nn.Dropout
    source_embeddings: nn.Embedding
    target_embeddings: nn.Embedding
    positional_encoding: positional_encoding.PositionalEncoding
    log_softmax: nn.LogSoftmax
    encoder: nn.TransformerEncoder
    decoder: nn.TransformerDecoder
    classifier: nn.Linear
    max_decode_len: int
    beam_width: Optional[int]
    label_smoothing: Optional[float]

    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        attention_heads,
        max_seq_len,
        output_size,
        pad_idx,
        start_idx,
        end_idx,
        optimizer,
        beta1,
        beta2,
        warmup_steps,
        learning_rate,
        scheduler,
        evaluator,
        max_decode_len,
        dropout=0.2,
        encoder_layers=4,
        decoder_layers=4,
        label_smoothing=None,
        beam_width=None,
        **kwargs,
    ):
        """Initializes the encoder-decoder with attention.

        Args:
            vocab_size (int).
            embedding_size (int).
            hidden_size (int).
            attention_heads (int).
            max_seq_len (int).
            output_size (int).
            pad_idx (int).
            start_idx (int).
            end_idx (int).
            optim (str).
            beta1 (float).
            beta2 (float).
            warmup_steps (int).
            learning_rate (float).
            evaluator (evaluators.Evaluator).
            scheduler (str).
            max_decode_len (int).
            dropout (float, optional).
            encoder_layers (int, optional).
            decoder_layers (int, optional).
            label_smoothing (float, optional).
            beam_width (int, optional): if specified, beam search is used
                during decoding.
            **kwargs: ignored.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_model = embedding_size
        self.attention_heads = attention_heads
        self.max_seq_len = max_seq_len
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.pad_idx = pad_idx
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
        self.source_embeddings = self.init_embeddings(
            vocab_size, self.d_model, self.pad_idx
        )
        self.target_embeddings = self.init_embeddings(
            output_size, self.d_model, self.pad_idx
        )
        self.positional_encoding = positional_encoding.PositionalEncoding(
            self.d_model, self.pad_idx, self.max_seq_len
        )
        self.log_softmax = nn.LogSoftmax(dim=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.encoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.decoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        self.classifier = nn.Linear(self.d_model, output_size)
        self.max_decode_len = max_decode_len
        self.label_smoothing = label_smoothing
        self.beam_width = beam_width
        self.loss_func = self.get_loss_func("mean")
        # Saves hyperparameters for PL checkpointing.
        self.save_hyperparameters()

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        return self._xavier_embedding_initialization(
            num_embeddings, embedding_size, pad_idx
        )

    def source_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        word_embedding = self.source_embeddings(symbols) * math.sqrt(
            self.d_model
        )
        positional_embedding = self.positional_encoding(symbols)
        out = self.dropout_layer(word_embedding + positional_embedding)
        return out

    def target_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the target symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        word_embedding = self.target_embeddings(symbols) * math.sqrt(
            self.d_model
        )
        positional_embedding = self.positional_encoding(symbols)
        out = self.dropout_layer(word_embedding + positional_embedding)
        return out

    def encode(
        self, source: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (torch.Tensor).
            source_mask (torch.Tensor).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedding = self.source_embed(source)
        return self.encoder(embedding, src_key_padding_mask=source_mask)

    def decode(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the logits for each step of the output sequence.

        Args:
            encoder_hidden (torch.Tensor): source encoder hidden state, of
                shape B x seq_len x hidden_size.
            source_mask (torch.Tensor): encoder hidden state mask.
            target (torch.Tensor): current state of targets, which may be the
                full target, or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): target mask.

        Returns:
            _type_: log softmax over targets.
        """
        target_embedding = self.target_embed(target)
        target_seq_len = target_embedding.size(1)
        # -> seq_len x seq_len.
        causal_mask = self.generate_square_subsequent_mask(target_seq_len).to(
            self.device
        )
        # -> B x seq_len x d_model
        decoder_hidden = self.decoder(
            target_embedding,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
        )
        # -> B x seq_len x vocab_size.
        output = self.classifier(decoder_hidden)
        output = self.log_softmax(output)
        return output

    def _decode_greedy(
        self, encoder_hidden: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        # The output distributions to be returned.
        outputs = []
        batch_size = encoder_hidden.size(0)
        # The predicted symbols at each iteration.
        preds = [
            torch.tensor(
                [self.start_idx for _ in range(encoder_hidden.size(0))],
                device=self.device,
            )
        ]
        # Tracking when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size, device=self.device)
        for _ in range(self.max_decode_len):
            target_tensor = torch.stack(preds, dim=1)
            # Uses a dummy mask of all ones.
            target_mask = torch.ones_like(target_tensor, dtype=torch.float)
            target_mask = target_mask == 0
            output = self.decode(
                encoder_hidden, source_mask, target_tensor, target_mask
            )
            # We only care about the last prediction in the sequence.
            last_output = output[:, -1, :]
            outputs.append(last_output)
            pred = self._get_predicted(last_output.unsqueeze(1))
            preds.append(pred.squeeze(1))
            # Updates to track which sequences have decoded an EOS.
            finished = torch.logical_or(finished, (preds[-1] == self.end_idx))
            # Break when all batches predicted an EOS symbol.
            if finished.all():
                break
        # -> B x vocab_size x seq_len
        return torch.stack(outputs).transpose(0, 1).transpose(1, 2)

    def forward(self, batch: base.Batch) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (Tuple[torch.Tensor, ...]): Tuple of tensors in the batch
                of shape (source, source_mask, target, target_mask) during
                training or shape (source, source_mask) during inference.

        Returns:
            torch.Tensor.
        """
        # Training mode with targets.
        if len(batch) == 4:
            source, source_mask, target, target_mask = batch
            # Initializes the start symbol for decoding.
            starts = (
                torch.tensor(
                    [self.start_idx], device=self.device, dtype=torch.long
                )
                .repeat(target.size(0))
                .unsqueeze(1)
            )
            target = torch.cat((starts, target), dim=1)
            target_mask = torch.cat(
                (starts == self.pad_idx, target_mask), dim=1
            )
            encoder_hidden = self.encode(source, source_mask)
            output = self.decode(
                encoder_hidden, source_mask, target, target_mask
            )
            # -> B x vocab_size x seq_len
            output = output.transpose(1, 2)[:, :, :-1]
        # No targets given at inference.
        elif len(batch) == 2:
            source, source_mask = batch
            encoder_hidden = self.encode(source, source_mask)
            # -> B x vocab_size x seq_len.
            output = self._decode_greedy(encoder_hidden, source_mask)
        else:
            raise Error(f"Batch of {len(batch)} elements is invalid")
        return output

    @staticmethod
    def generate_square_subsequent_mask(length: int) -> torch.Tensor:
        """Generates the target mask so the model cannot see future states.

        Args:
            length (int): length of the sequence.

        Returns:
            torch.Tensor: mask of shape length x length.
        """
        return torch.triu(torch.full((length, length), -math.inf), diagonal=1)


class FeatureInvariantTransformerEncoderDecoder(TransformerEncoderDecoder):
    """Transformer encoder-decoder with feature invariance.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    features_idx: int

    def __init__(self, *args, features_idx, **kwargs):
        super().__init__(*args, **kwargs)
        # Distinguishes features vs. character.
        self.features_idx = features_idx
        self.type_embedding = self.init_embeddings(
            2, self.embedding_size, self.pad_idx
        )

    def source_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols.

        This adds positional encodings and special embeddings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        # Distinguishes features and chars.
        char_mask = (symbols < self.features_idx).long()
        # 1 or 0.
        type_embedding = math.sqrt(self.d_model) * self.type_embedding(
            char_mask
        )
        word_embedding = self.source_embeddings(symbols) * math.sqrt(
            self.d_model
        )
        positional_embedding = self.positional_encoding(
            symbols, mask=char_mask
        )
        out = self.dropout_layer(
            word_embedding + positional_embedding + type_embedding
        )
        return out
