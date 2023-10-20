"""Transformer model classes."""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from ... import data
from . import base


class PositionalEncoding(nn.Module):
    """Positional encoding.

    After:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    # Model arguments.
    pad_idx: int

    def __init__(
        self,
        d_model: int,
        pad_idx,
        max_source_length: int,
    ):
        """
        Args:
            d_model (int).
            pad_idx (int).
            max_source_length (int).
        """
        super().__init__()
        self.pad_idx = pad_idx
        positional_encoding = torch.zeros(max_source_length, d_model)
        position = torch.arange(
            0, max_source_length, dtype=torch.float
        ).unsqueeze(1)
        scale_factor = -math.log(10000.0) / d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * scale_factor
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self, symbols: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            symbols (torch.Tensor): symbol indices to encode B x seq_len.
            mask (torch.Tensor, optional): defaults to None; optional mask for
                positions not to be encoded.
        Returns:
            torch.Tensor: positional embedding.
        """
        out = self.positional_encoding.repeat(symbols.size(0), 1, 1)
        if mask is not None:
            # Indices should all be 0's until the first unmasked position.
            indices = torch.cumsum(mask, dim=1)
        else:
            indices = torch.arange(symbols.size(1)).long()
        # Selects the tensors from `out` at the specified indices.
        out = out[torch.arange(out.shape[0]).unsqueeze(-1), indices]
        # Zeros out pads.
        pad_mask = symbols.ne(self.pad_idx).unsqueeze(2)
        # TODO: consider in-place mul_.
        out = out * pad_mask
        return out


class AttentionOutput:
    """Tracks an attention output during the forward pass.

    This object can be passed into a hook to the Transformer forward pass
    in order to modify its behavior such that certain attention weights are
    returned, and then stored in self.outputs."""

    # Model arguments.
    outputs: List

    def __init__(self):
        """Initializes an AttentionOutput."""
        self.outputs = []

    def __call__(
        self,
        module: nn.Module,
        module_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        module_out: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Stores the second return argument of `module`.

        This is intended to be called on a multiehaded attention, which returns
        both the contextualized representation, and the attention weights.

        Args:
            module (nn.Module): A torch module.
            module_in (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Input
                to the module.
            module_out (Tuple[torch.Tensor, torch.Tensor]): Output from
                the module. The second tensor is the attention weights.
        """
        self.outputs.append(module_out[1])

    def clear(self) -> None:
        """Clears the outputs."""
        self.outputs.clear()


class TransformerModule(base.BaseModule):
    """Base module for Transformer."""

    # Model arguments.
    source_attention_heads: int
    # Constructed inside __init__.
    esq: float
    module: nn.TransformerEncoder
    positional_encoding: PositionalEncoding

    def __init__(
        self,
        *args,
        source_attention_heads,
        max_source_length: int,
        **kwargs,
    ):
        """Initializes the module with attention.

        Args:
            *args: passed to superclass.
            source_attention_heads (int).
            max_source_length (int).
            **kwargs: passed to superclass.
        """
        super().__init__(
            *args,
            source_attention_heads=source_attention_heads,
            max_source_length=max_source_length,
            **kwargs,
        )
        self.source_attention_heads = source_attention_heads
        self.esq = math.sqrt(self.embedding_size)
        self.module = self.get_module()
        self.positional_encoding = PositionalEncoding(
            self.embedding_size, self.pad_idx, max_source_length
        )

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

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        word_embedding = self.esq * self.embeddings(symbols)
        positional_embedding = self.positional_encoding(symbols)
        return self.dropout_layer(word_embedding + positional_embedding)


class TransformerEncoder(TransformerModule):
    def forward(self, source: data.PaddedTensor) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (data.PaddedTensor).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedding = self.embed(source.padded)
        output = self.module(embedding, src_key_padding_mask=source.mask)
        return base.ModuleOutput(output)

    def get_module(self) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            dim_feedforward=self.hidden_size,
            nhead=self.source_attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
        )

    @property
    def output_size(self) -> int:
        return self.embedding_size

    @property
    def name(self) -> str:
        return "transformer"


class FeatureInvariantTransformerEncoder(TransformerEncoder):
    """Encoder for transformer with feature invariance.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    features_vocab_size: int
    # Constructed inside __init__.
    type_embedding: nn.Embedding

    def __init__(self, *args, features_vocab_size, **kwargs):
        super().__init__(*args, **kwargs)
        # Distinguishes features vs. character.
        self.features_vocab_size = features_vocab_size
        self.type_embedding = self.init_embeddings(
            2, self.embedding_size, self.pad_idx
        )

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols.

        This adds positional encodings and special embeddings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        # Distinguishes features and chars; 1 or 0.
        char_mask = (
            symbols < (self.num_embeddings - self.features_vocab_size)
        ).long()
        type_embedding = self.esq * self.type_embedding(char_mask)
        word_embedding = self.esq * self.embeddings(symbols)
        positional_embedding = self.positional_encoding(
            symbols, mask=char_mask
        )
        out = self.dropout_layer(
            word_embedding + positional_embedding + type_embedding
        )
        return out

    @property
    def name(self) -> str:
        return "feature-invariant transformer"


class TransformerDecoderLayerSeparateFeatures(nn.TransformerDecoderLayer):
    """Transformer decoder layer with separate features.

    Each decode step gets a second multihead attention representation
    wrt the encoded features. This and the original multihead attention
    representation w.r.t. the encoded symbols are then compressed in a
    linear layer and finally concatenated.

    The implementation is otherwise identical to nn.TransformerDecoderLayer."""

    def __init__(self, *args, nfeature_heads, **kwargs):
        super().__init__(*args, **kwargs)
        factory_kwargs = {
            "device": kwargs.get("device"),
            "dtype": kwargs.get("dtype"),
        }
        self.feature_multihead_attn = nn.MultiheadAttention(
            kwargs["d_model"],  # TODO: Separate feature embedding size?
            nfeature_heads,
            dropout=kwargs["dropout"],
            batch_first=kwargs["batch_first"],
            **factory_kwargs,
        )
        self.symbols_linear = nn.Linear(
            kwargs["d_model"],
            # FIXME: This will break when used if odd d_model
            int(kwargs["d_model"] / 2),
            bias=kwargs.get("bias"),
            **factory_kwargs,
        )
        self.features_linear = nn.Linear(
            kwargs["d_model"],  # TODO: Separate feature embedding size?
            # FIXME: This will break when used if odd d_model
            int(kwargs["d_model"] / 2),
            bias=kwargs.get("bias"),
            **factory_kwargs,
        )

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        features_memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        features_memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        target_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            target (torch.Tensor): the sequence to the decoder layer.
            memory (torch.Tensor): the sequence from the last layer of the
                encoder.
            features_memory (torch.Tensor): the mask for the features.
            target_mask (Optional[torch.Tensor], optional): the mask for the
                target sequence. Defaults to None.
            memory_mask (Optional[torch.Tensor], optional): the mask for the
                memory sequence. Defaults to None.
            features_memory_mask (torch.Tensor, optional): the mask
                for the features. Defaults to None.
            target_key_padding_mask (Optional[torch.Tensor], optional): the
                mask for the target keys per batch. Defaults to None.
            memory_key_padding_mask (Optional[torch.Tensor], optional): the
                mask for the memory keys per batch
            target_is_causal (bool, optional): If specified, applies a causal
                mask as target mask. Mutually exclusive with providing
                target_mask. Defaults to False.
            memory_is_causal (bool, optional): If specified, applies a causal
                mask as target mask. Mutually exclusive with providing
                memory_mask. Defaults to False.

        Returns:
            torch.Tensor: Ouput tensor.
        """
        x = target
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                target_mask,
                target_key_padding_mask,
                # FIXME: Introduced in torch 2.0.
                # is_causal=target_is_causal
            )
            x = self.norm2(x)
            symbol_attention = self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                # FIXME Introduced in torch 2.0.
                # memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            symbol_attention = self.symbols_linear(symbol_attention)
            feature_attention = self._features_mha_block(
                x,
                features_memory,
                features_memory_mask,
                features_memory_mask,
                # FIXME Introduced in torch 2.0.
                # memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            feature_attention = self.features_linear(feature_attention)
            x = torch.cat([symbol_attention, feature_attention], dim=2)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    target_mask,
                    target_key_padding_mask,
                    # FIXME: Introduced in torch 2.0.
                    # is_causal=target_is_causal
                )
            )
            symbol_attention = self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                # FIXME Introduced in torch 2.0.
                # memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            symbol_attention = self.symbols_linear(symbol_attention)
            feature_attention = self._features_mha_block(
                x,
                features_memory,
                features_memory_mask,
                features_memory_mask,
                # FIXME Introduced in torch 2.0.
                # memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            feature_attention = self.features_linear(feature_attention)
            x = x + torch.cat([symbol_attention, feature_attention], dim=2)
            x = self.norm2(x)
            x = self.norm3(x + self._ff_block(x))
        return x

    def _features_mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        # FIXME: Introduced in torch 2.0.
        # is_causal: bool = False,
    ) -> torch.Tensor:
        """Runs the multihead attention block that attends to features.

        Args:
            x (torch.Tensor): The `query` tensor, i.e. the previous decoded
                embeddings.
            mem (torch.Tensor): The `keys` and `values`, i.e. the encoded
                features.
            attn_mask (torch.Tensor, optional): the mask for the features.
            key_padding_mask (torch.Tensor, optional): the mask for the
                feature keys per batch.

        Returns:
            torch.Tensor: Concatenated attention head tensors.
        """
        x = self.feature_multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            # FIXME: Introduced in torch 2.0.
            # is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)


class TransformerDecoderSeparateFeatures(nn.TransformerDecoder):
    """A Transformer decoder with separate features.

    Adding separate features into the transformer stack is implemented with
    TransformerDecoderLayerseparateFeatures layers.
    """

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        features_memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        features_memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            target (torch.Tensor): the sequence to the decoder.
            memory (torch.Tensor): the sequence from the last layer of the
                encoder.
            features_memory (torch.Tensor): the sequence from the last layer
                of the features encoder.
            target_mask (Optional[torch.Tensor], optional): the mask for the
                target sequence. Defaults to None.
            memory_mask (Optional[torch.Tensor], optional): the mask for the
                memory sequence. Defaults to None.
            features_memory_mask (Optional[torch.Tensor], optional): the mask
                for the features. Defaults to None.
            target_key_padding_mask (Optional[torch.Tensor], optional): the
                mask for the target keys per batch. Defaults to None.
            memory_key_padding_mask (Optional[torch.Tensor], optional): the
                mask for the memory keys per batch. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        output = target
        for mod in self.layers:
            output = mod(
                output,
                memory,
                features_memory,
                target_mask=target_mask,
                memory_mask=memory_mask,
                features_memory_mask=features_memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(TransformerModule):
    """Decoder for Transformer."""

    # Output arg.
    decoder_input_size: int
    # Constructed inside __init__.
    module: nn.TransformerDecoder

    def __init__(self, *args, decoder_input_size, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Performs single pass of decoder module.

        Args:
            encoder_hidden (torch.Tensor): source encoder hidden state, of
                shape B x seq_len x hidden_size.
            source_mask (torch.Tensor): encoder hidden state mask.
            target (torch.Tensor): current state of targets, which may be the
                full target, or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): target mask.

        Returns:
            torch.Tensor: torch tensor of decoder outputs.
        """
        target_embedding = self.embed(target)
        target_sequence_length = target_embedding.size(1)
        # -> seq_len x seq_len.
        causal_mask = self.generate_square_subsequent_mask(
            target_sequence_length
        ).to(self.device)
        # -> B x seq_len x d_model.
        output = self.module(
            target_embedding,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
        )
        return base.ModuleOutput(output, embeddings=target_embedding)

    def get_module(self) -> nn.TransformerDecoder:
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_input_size,
            dim_feedforward=self.hidden_size,
            nhead=self.source_attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        return nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
        )

    @staticmethod
    def generate_square_subsequent_mask(length: int) -> torch.Tensor:
        """Generates the target mask so the model cannot see future states.

        Args:
            length (int): length of the sequence.

        Returns:
            torch.Tensor: mask of shape length x length.
        """
        return torch.triu(torch.full((length, length), -math.inf), diagonal=1)

    @property
    def output_size(self) -> int:
        return self.num_embeddings

    @property
    def name(self) -> str:
        return "transformer"


class TransformerPointerDecoder(TransformerDecoder):
    """TransformerDecoder with separate features and `attention_output`.

    `attention_output` tracks the output of multiheaded attention from each
    decoder step wrt the encoded input. This is achieved with a hook into the
    forward pass. We additionally expect separately decoded features, which
    are passed through `features_attention_heads` multiheaded attentions from
    each decoder step wrt the encoded features.

    After:
        https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
    """

    def __init__(
        self, *args, separate_features, features_attention_heads, **kwargs
    ):
        """Initializes the TransformerPointerDecoder object."""
        self.separate_features = separate_features
        self.features_attention_heads = features_attention_heads
        super().__init__(*args, **kwargs)
        # Call this to get the actual cross attentions.
        self.attention_output = AttentionOutput()
        # multihead_attn refers to the attention from decoder to encoder.
        self.patch_attention(self.module.layers[-1].multihead_attn)
        self.hook_handle = self.module.layers[
            -1
        ].multihead_attn.register_forward_hook(self.attention_output)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        features_memory: Optional[torch.Tensor] = None,
        features_memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs single pass of decoder module.

        Args:
            encoder_hidden (torch.Tensor): source encoder hidden state, of
                shape B x seq_len x hidden_size.
            source_mask (torch.Tensor): encoder hidden state mask.
            target (torch.Tensor): current state of targets, which may be the
                full target, or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): target mask.
            features_memory (Optional[torch.Tensor]): Encoded features.
            features_memory_mask (Optional[torch.Tensor]): Mask for encoded
                features.

        Returns:
            torch.Tensor: torch tensor of decoder outputs.
        """
        target_embedding = self.embed(target)
        target_sequence_length = target_embedding.size(1)
        # -> seq_len x seq_len.
        causal_mask = self.generate_square_subsequent_mask(
            target_sequence_length
        ).to(self.device)
        # -> B x seq_len x d_model.
        if self.separate_features:
            output = self.module(
                target_embedding,
                encoder_hidden,
                features_memory=features_memory,
                target_mask=causal_mask,
                memory_key_padding_mask=source_mask,
                features_memory_mask=features_memory_mask,
                target_key_padding_mask=target_mask,
            )
        else:
            # TODO: Resolve mismatch between our 'target' naming convention and
            # torch's use of `tgt`.
            output = self.module(
                target_embedding,
                encoder_hidden,
                tgt_mask=causal_mask,
                memory_key_padding_mask=source_mask,
                tgt_key_padding_mask=target_mask,
            )
        return base.ModuleOutput(output, embeddings=target_embedding)

    def get_module(self) -> nn.TransformerDecoder:
        if self.separate_features:
            decoder_layer = TransformerDecoderLayerSeparateFeatures(
                d_model=self.decoder_input_size,
                dim_feedforward=self.hidden_size,
                nhead=self.source_attention_heads,
                nfeature_heads=self.features_attention_heads,
                dropout=self.dropout,
                activation="relu",
                norm_first=True,
                batch_first=True,
            )
            return TransformerDecoderSeparateFeatures(
                decoder_layer=decoder_layer,
                num_layers=self.layers,
                norm=nn.LayerNorm(self.embedding_size),
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.decoder_input_size,
                dim_feedforward=self.hidden_size,
                nhead=self.source_attention_heads,
                dropout=self.dropout,
                activation="relu",
                norm_first=True,
                batch_first=True,
            )
            return nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=self.layers,
                norm=nn.LayerNorm(self.embedding_size),
            )

    def patch_attention(self, attention_module: torch.nn.Module) -> None:
        """Wraps a module's forward pass such that `need_weights` is True.

        Args:
            attention_module (torch.nn.Module): The module from which we want
                to track multiheaded attention weights.
        """
        forward_orig = attention_module.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            return forward_orig(*args, **kwargs)

        attention_module.forward = wrap
