"""Transformer module classes."""

import abc
import collections
from typing import Optional, Tuple

import numpy
import torch
from torch import nn

from ... import data
from .. import embeddings
from . import base, position


class Error(Exception):
    pass


class AttentionOutput(collections.UserList):
    """Tracks an attention output during the forward pass.

    This object can be passed into a hook to the Transformer forward pass
    in order to modify its behavior such that certain attention weights are
    returned, and then stored in self.outputs.
    """

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
            module (nn.Module): ignored.
            module_in (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                ignored.
            module_out (Tuple[torch.Tensor, torch.Tensor]): Output from
                the module. The second tensor is the attention weights.
        """
        _, attention = module_out
        self.append(attention)


class TransformerModule(base.BaseModule):
    """Abstract base module for transformers.

    Args:
        attention_heads (int).
        max_length (int).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    attention_heads: int
    # Constructed inside __init__.
    esq: float
    module: nn.TransformerEncoder
    positional_encoding: position.PositionalEncoding

    def __init__(
        self,
        *args,
        attention_heads,
        max_length,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention_heads = attention_heads
        self.esq = numpy.sqrt(self.embedding_size)
        self.module = self.get_module()
        self.positional_encoding = position.PositionalEncoding(
            self.embedding_size, max_length
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
        word_embedded = self.esq * self.embeddings(symbols)
        positional_embedded = self.positional_encoding(symbols)
        return self.dropout_layer(word_embedded + positional_embedded)

    @abc.abstractmethod
    def get_module(self) -> base.BaseModule: ...


class TransformerEncoder(TransformerModule):
    """Transformer encoder.

    Our implementation uses "pre-norm", i.e., it applies layer normalization
    before attention. Because of this, we are not currently able to make use
    of PyTorch's nested tensor feature.

    After:
        Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., ..., and
        Liu, T.-Y. 2020. In Proceedings of the 37th International Conference
        on Machine Learning, pages 10524-10533.
    """

    def forward(self, source: data.PaddedTensor) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (data.PaddedTensor).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.embed(source.padded)
        return self.module(embedded, src_key_padding_mask=source.mask)

    def get_module(self) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
            # This silences a warning about the use of nested tensors,
            # currently an experimental feature.
            # TODO(#225): Re-enable if nested tensors are generalized to work
            # with `norm_first`.
            enable_nested_tensor=False,
        )

    @property
    def name(self) -> str:
        return "transformer"

    @property
    def output_size(self) -> int:
        return self.embedding_size


class FeatureInvariantTransformerEncoder(TransformerEncoder):
    """Transformer encoder with feature invariance.

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
        # Uses Xavier initialization.
        self.type_embedding = embeddings.xavier_embedding(
            2,
            self.embedding_size,
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
        # Distinguishes features and chars; 1 or 0; embedding layer requires
        # this to be integral.
        char_mask = (
            symbols < (self.num_embeddings - self.features_vocab_size)
        ).long()
        word_embedded = self.esq * self.embeddings(symbols)
        type_embedded = self.esq * self.type_embedding(char_mask)
        positional_embedded = self.positional_encoding(symbols, mask=char_mask)
        return self.dropout_layer(
            word_embedded + type_embedded + positional_embedded
        )

    @property
    def name(self) -> str:
        return "feature-invariant transformer"


class WrappedTransformerDecoder(nn.TransformerDecoder):
    """Wraps TransformerDecoder API for better variable naming."""

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target_embedded: torch.Tensor,
        target_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            memory=source_encoded,
            memory_key_padding_mask=source_mask,
            tgt=target_embedded,
            tgt_key_padding_mask=target_mask,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )


class TransformerDecoder(TransformerModule):
    """Transformer decoder.

    Args:
        decoder_input_size (int).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    decoder_input_size: int
    # Constructed inside __init__.
    module: nn.TransformerDecoder

    def __init__(self, *args, decoder_input_size, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs single pass of decoder module.

        Args:
            source_encoded (torch.Tensor): encoded source sequence.
            source_mask (torch.Tensor): mask for source.
            target (torch.Tensor): current state of targets, which may be the
                full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): mask for target.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: decoder outputs and the
                embedded targets.
        """
        target_embedded = self.embed(target)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_embedded.size(1),
            device=self.device,
            dtype=bool,
        )
        # -> B x seq_len x d_model.
        decoded = self.module(
            source_encoded,
            source_mask,
            target_embedded,
            target_mask,
            causal_mask,
        )
        return decoded, target_embedded

    def get_module(self) -> WrappedTransformerDecoder:
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_input_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        return WrappedTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
        )

    @property
    def name(self) -> str:
        return "transformer"

    @property
    def output_size(self) -> int:
        return self.num_embeddings


class SeparateFeaturesTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Transformer decoder layer with separate features.

    Each decode step gets a second multihead attention representation
    wrt the encoded features. This and the original multihead attention
    representation w.r.t. the encoded symbols are then compressed in a
    linear layer and finally concatenated.

    The implementation is otherwise identical to nn.TransformerDecoderLayer.

    Args:
        attention_heads (int).
        *args: passed to superclass.
        *kwargs: passed to superclass.
    """

    def __init__(self, *args, attention_heads, **kwargs):
        super().__init__(*args, **kwargs)
        factory_kwargs = {
            "device": kwargs.get("device"),
            "dtype": kwargs.get("dtype"),
        }
        d_model = kwargs["d_model"]
        self.features_multihead_attn = nn.MultiheadAttention(
            d_model,  # TODO: Separate feature embedding size?
            attention_heads,
            dropout=kwargs["dropout"],
            batch_first=kwargs["batch_first"],
            **factory_kwargs,
        )
        # If d_model is not even, an error will result. This is unlikely to
        # trigger since it must also be divisible by the number of attention
        # heads.
        if d_model % 2 != 0:
            raise Error(
                f"Feature-invariant transformer d_model ({d_model}) must be "
                "divisible by 2"
            )
        self.symbols_linear = nn.Linear(
            d_model,
            d_model // 2,
            bias=kwargs.get("bias"),
            **factory_kwargs,
        )
        self.features_linear = nn.Linear(
            d_model,  # TODO: Separate feature embedding size?
            d_model // 2,
            bias=kwargs.get("bias"),
            **factory_kwargs,
        )

    # TODO: Clean up the naming and ordering here.

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

        This is based closely on the internals in torch.nn.transformer and
        follows the somewhat-inscrutable variable naming used there.

        Args:
            target (torch.Tensor): the sequence to the decoder layer.
            memory (torch.Tensor): the sequence from the last layer of the
                encoder.
            features_memory (torch.Tensor): the mask for the features.
            target_mask (torch.Tensor, optional): the mask for the
                target sequence.
            memory_mask (torch.Tensor, optional): the mask for the
                memory sequence.
            features_memory_mask (torch.Tensor, optional): the mask
                for the features.
            target_key_padding_mask (torch.Tensor, optional): the
                mask for the target keys per batch.
            memory_key_padding_mask (torch.Tensor, optional): the
                mask for the memory keys per batch.
            target_is_causal (bool, optional): if specified, applies a causal
                mask as target mask. Mutually exclusive with providing
                target_mask.
            memory_is_causal (bool, optional): if specified, applies a causal
                mask as target mask. Mutually exclusive with providing
                memory_mask.

        Returns:
            torch.Tensor: Ouput tensor.
        """
        x = target
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                target_mask,
                target_key_padding_mask,
                is_causal=target_is_causal,
            )
            x = self.norm2(x)
            symbol_attention = self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            symbol_attention = self.symbols_linear(symbol_attention)
            features_attention = self._features_mha_block(
                x,
                features_memory,
                features_memory_mask,
                features_memory_mask,
                memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            features_attention = self.features_linear(features_attention)
            x = torch.cat((symbol_attention, features_attention), dim=2)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    target_mask,
                    target_key_padding_mask,
                    is_causal=target_is_causal,
                )
            )
            symbol_attention = self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            symbol_attention = self.symbols_linear(symbol_attention)
            features_attention = self._features_mha_block(
                x,
                features_memory,
                features_memory_mask,
                features_memory_mask,
                memory_is_causal,
            )
            # TODO: Do we want a nonlinear activation?
            features_attention = self.features_linear(features_attention)
            x = x + torch.cat([symbol_attention, features_attention], dim=2)
            x = self.norm2(x)
            x = self.norm3(x + self._ff_block(x))
        return x

    # TODO: Clean up the naming here.

    def _features_mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Runs the multihead attention block that attends to features.

        Args:
            x (torch.Tensor): the `query` tensor, i.e. the previous decoded
                embeddings.
            mem (torch.Tensor): the `keys` and `values`, i.e. the encoded
                features.
            attn_mask (torch.Tensor, optional): the mask for the features.
            key_padding_mask (torch.Tensor, optional): the mask for the
                feature keys per batch.

        Returns:
            torch.Tensor: concatenated attention head tensors.
        """
        x = self.features_multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)


class SeparateFeaturesTransformerDecoder(nn.TransformerDecoder):
    """Transformer decoder with separate features.

    Adding separate features into the transformer stack is implemented with
    SeparateFeaturesTransformerDecoderLayer.
    """

    # TODO: Clean up the naming and ordering here.

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
        """Passes the inputs (and mask) through the decoder layer.

        Args:
            target (torch.Tensor): the sequence to the decoder.
            memory (torch.Tensor): the sequence from the last layer of the
                encoder.
            features_memory (torch.Tensor): the sequence from the last layer
                of the features encoder.
            target_mask (Optional[torch.Tensor], optional): the mask for the
                target sequence.
            memory_mask (Optional[torch.Tensor], optional): the mask for the
                memory sequence.
            features_memory_mask (Optional[torch.Tensor], optional): the mask
                for the features.
            target_key_padding_mask (Optional[torch.Tensor], optional): the
                mask for the target keys per batch.
            memory_key_padding_mask (Optional[torch.Tensor], optional): the
                mask for the memory keys per batch.

        Returns:
            torch.Tensor: Output tensor.
        """
        output = target
        for layer in self.layers:
            output = layer(
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


class TransformerPointerDecoder(TransformerDecoder):
    """A transformer decoder with separate features and `attention_output`.

    `attention_output` tracks the output of multiheaded attention from each
    decoder step w.r.t. the encoded input. This is achieved with a hook into
    the forward pass. We additionally expect separately decoded features, which
    are passed through multiheaded attentions from each decoder step w.r.t.
    the encoded features.

    After:
        https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91

    Args:
        attention_heads (int).
        *args: passed to superclass.
        *kwargs: passed to superclass.
    """

    attention_heads: int

    def __init__(
        self,
        *args,
        attention_heads,
        **kwargs,
    ):
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
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs single pass of decoder module.

        Args:
            source_encoded (torch.Tensor): encoded source sequence.
            source_mask (torch.Tensor): mask for source.
            target (torch.Tensor): current state of targets, which may be the
                full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): mask for target.
            features_encoded (Optional[torch.Tensor]): encoded features.
            features_mask (Optional[torch.Tensor]): mask for features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: decoder outputs and the
                embedded targets.
        """
        target_embedded = self.embed(target)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_embedded.size(1),
            device=self.device,
            dtype=bool,
        )
        # -> B x seq_len x d_model.
        # TODO: Clean up the naming and ordering here.
        decoded = self.module(
            target_embedded,
            source_encoded,
            features_memory=features_encoded,
            target_mask=causal_mask,
            memory_key_padding_mask=source_mask,
            features_memory_mask=features_mask,
            target_key_padding_mask=target_mask,
        )
        return decoded, target_embedded

    def get_module(self) -> nn.TransformerDecoder:
        decoder_layer = SeparateFeaturesTransformerDecoderLayer(
            d_model=self.decoder_input_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            nfeature_heads=self.attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        return SeparateFeaturesTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
        )

    def patch_attention(self, attention_module: torch.nn.Module) -> None:
        """Wraps a module's forward pass such that `need_weights` is True.

        Args:
            attention_module (torch.nn.Module): the module from which we want
                to track multiheaded attention weights.
        """
        forward_orig = attention_module.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            return forward_orig(*args, **kwargs)

        attention_module.forward = wrap
