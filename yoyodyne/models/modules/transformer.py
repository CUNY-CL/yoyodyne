"""Transformer module classes."""

import abc
import collections
import math

import torch
from torch import nn

from ... import data, defaults, special
from . import base, position, transformer_layers


class Error(Exception):
    pass


# Helpers.


class AttentionOutput(collections.UserList):
    """Tracks an attention output during the forward pass.

    This object can be passed into a hook to the Transformer forward pass
    in order to modify its behavior such that certain attention weights are
    returned, and then stored in self.outputs.
    """

    def __call__(
        self,
        module: nn.Module,
        module_in: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        module_out: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Stores the second return argument of `module`.

        This is intended to be called on a multiheaded attention, which returns
        both the contextualized representation, and the attention weights.

        Args:
            module (nn.Module): ignored.
            module_in (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                ignored.
            module_out (tuple[torch.Tensor, torch.Tensor]): output from
                the module; the second tensor is the attention weights.
        """
        _, attention = module_out
        self.append(attention)


# Generic modules.


class TransformerModule(base.BaseModule):
    """Abstract base module for transformers.

    Args:
        *args: passed to superclass.
        attention_heads (int, optional): number of attention heads.
        hidden_size (int, optional).
        layers (int, optional): number of layers.
        max_length (int, optional): maximum length for positional encoding.
            If not provided, one must call `set_max_length` before use.
        positional_encoding (position.BasePositionalEncoding, optional):
            a positional encoding object; if not specified, a sinusoidal
            encoding of the appropriate size will be allocated.
        **kwargs: passed to superclass.
    """

    attention_heads: int
    esq: float
    hidden_size: int
    layers: int
    module: nn.TransformerEncoder | nn.TransformerDecoder
    positional_encoding: position.BasePositionalEncoding | None

    def __init__(
        self,
        *args,
        attention_heads: int = defaults.ATTENTION_HEADS,
        hidden_size: int = defaults.HIDDEN_SIZE,
        layers: int = defaults.LAYERS,
        max_length: int | None = None,
        positional_encoding: position.BasePositionalEncoding | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention_heads = attention_heads
        self.esq = math.sqrt(self.embedding_size)
        self.hidden_size = hidden_size
        self.layers = layers
        self.positional_encoding = positional_encoding
        self.module = self.get_module()
        if max_length is not None:
            self.set_max_length(max_length)

    def embed(
        self, symbols: torch.Tensor, embeddings: nn.Embedding
    ) -> torch.Tensor:
        """Embeds the symbols and adds positional encoding."""
        embedded = self.esq * embeddings(symbols)
        return self.dropout_layer(self.positional_encoding(symbols, embedded))

    @abc.abstractmethod
    def get_module(self) -> base.BaseModule: ...

    @property
    def max_length(self) -> int:
        return self.positional_encoding.max_length

    @property
    def name(self) -> str:
        return (
            "transformer "
            f"({self.positional_encoding.name} positional encoding)"
        )

    def set_max_length(self, max_length: int) -> None:
        if self.positional_encoding is None:
            self.positional_encoding = position.SinusoidalPositionalEncoding(
                self.embedding_size,
                max_length,
            )
        elif self.positional_encoding.max_length < max_length:
            raise Error(
                f"{self.positional_encoding.name} max_length "
                f"({self.positional_encoding.max_length}) < "
                f"max_length ({max_length}"
            )


class RotaryTransformerModule:
    """Mixin for RoPE length computation.

    This should be used at the top of the MRO for all rotary transformer
    modules.
    """

    def set_max_length(self, max_length: int) -> None:
        if self.positional_encoding is None:
            self.positional_encoding = position.RotaryPositionalEncoding(
                embedding_size=self._rope_head_dim,
                max_length=max_length,
            )
            self._set_rope(self.positional_encoding)
        elif self.positional_encoding.max_length < max_length:
            raise Error(
                f"rotary max_length ({self.positional_encoding.max_length}) < "
                f"max_length ({max_length})"
            )

    def _set_rope(self, rope: position.RotaryPositionalEncoding) -> None:
        """Resizes the RoPE cache on all layers to the given max_length.

        This replaces the rope attribute on every layer that holds one,
        avoiding a full rebuild of the module stack.
        """
        for layer in self.module.layers:
            layer.self_attn.rope = rope
            if hasattr(layer, "multihead_attn"):
                layer.multihead_attn.rope = rope
            if hasattr(layer, "features_multihead_attn"):
                layer.features_multihead_attn.rope = rope

    @property
    def _rope_max_length(self) -> int:
        """The max_length to use when constructing the RoPE cache.

        Falls back to defaults.MAX_LENGTH if set_max_length has not yet been
        called (i.e., positional_encoding is still None). In that case the
        the cache will be updated via _set_rope once the true length is known.
        """
        if self.positional_encoding is not None:
            return self.positional_encoding.max_length
        return defaults.MAX_LENGTH

    @property
    def _rope_head_dim(self) -> int:
        """The per-head dimension used to size the RoPE frequency table."""
        return self.embedding_size // self.attention_heads


# Encoders.


class TransformerEncoder(TransformerModule, base.BaseEncoder):
    """Transformer encoder.

    Our implementation uses "pre-norm", i.e., it applies layer normalization
    before attention. Because of this, we are not currently able to make use
    of PyTorch's nested tensor feature.

    The caller is responsible for calling set_max_length.

    After:
        Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., ..., and
        Liu, T.-Y. 2020. On layer normalization in the transformer
        architecture. In _Proceedings of the 37th International Conference on
        Machine Learning_, pages 10524-10533.
    """

    def forward(
        self,
        symbols: data.PaddedTensor,
        embeddings: nn.Embedding,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encodes the symbols with the TransformerEncoder.

        Args:
            symbols (data.PaddedTensor).
            embeddings (nn.Embedding).
            *args: ignored.
            **kwargs: ignored.

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.embed(symbols.tensor, embeddings)
        return self.module(embedded, src_key_padding_mask=symbols.mask)

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
            enable_nested_tensor=False,
        )

    @property
    def output_size(self) -> int:
        return self.embedding_size


class FeatureInvariantTransformerEncoder(TransformerEncoder):
    """Transformer encoder with feature invariance.

    This is only sensibly used in a configuration where there is a shared
    source and features encoder, as in the following YAML snippet:

        source_encoder:
          class_path: yoyodyne.models.modules.TransformerEncoder
            init_args:
              ...
        features_encoder: true

    There is already space in the embeddings for the type embeddings; the
    caller just has to indicate whether the symbols are source or target
    symbols.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In _Proceedings of the 16th Conference
        of the European Chapter of the Association for Computational
        Linguistics: Main Volume_, pages 1901-1907.
    """

    def embed(
        self,
        symbols: torch.Tensor,
        mask: torch.Tensor,
        embeddings: nn.Embedding,
        is_source: bool,
    ) -> torch.Tensor:
        """Embeds the symbols and adds type and positional encodings."""
        embedded = self.esq * embeddings(symbols)
        type_embedded = self.esq * embeddings(
            torch.where(
                mask,
                special.PAD_IDX,
                special.SOURCE_IDX if is_source else special.FEATURES_IDX,
            )
        )
        return self.dropout_layer(
            self.positional_encoding(embedded + type_embedded, symbols)
        )

    def forward(
        self,
        symbols: data.PaddedTensor,
        embeddings: nn.Embedding,
        is_source: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encodes the symbols with the TransformerEncoder.

        Args:
            symbols (data.PaddedTensor).
            embeddings (nn.Embedding).
            is_source (bool): is this being used to encode source or features?
            *args: ignored.
            **kwargs: ignored.

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.embed(
            symbols.tensor, symbols.mask, embeddings, is_source
        )
        return self.module(embedded, src_key_padding_mask=symbols.mask)

    @property
    def name(self) -> str:
        return f"feature-invariant {super().name}"


# Rotary encoders.


class RotaryTransformerEncoder(RotaryTransformerModule, TransformerEncoder):
    """Transformer encoder with rotary positional encodings.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def get_module(self) -> nn.TransformerEncoder:
        rope = position.RotaryPositionalEncoding(
            embedding_size=self._rope_head_dim,
            max_length=self._rope_max_length,
        )
        encoder_layer = transformer_layers.RotaryTransformerEncoderLayer(
            rope,
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
            enable_nested_tensor=False,
        )

    @property
    def name(self) -> str:
        return "rotary transformer"


class RotaryFeatureInvariantTransformerEncoder(
    RotaryTransformerModule, FeatureInvariantTransformerEncoder
):
    """FeatureInvariantTransformerEncoder with rotary positional encodings.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def get_module(self) -> nn.TransformerEncoder:
        rope = position.RotaryPositionalEncoding(
            embedding_size=self._rope_head_dim,
            max_length=self._rope_max_length,
        )
        encoder_layer = transformer_layers.RotaryTransformerEncoderLayer(
            rope,
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
            enable_nested_tensor=False,
        )

    @property
    def name(self) -> str:
        return "rotary feature-invariant transformer"


# Decoder helpers.


class WrappedTransformerDecoder(nn.TransformerDecoder):
    """Wraps TransformerDecoder API for better variable naming."""

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            memory=source_encoded,
            memory_key_padding_mask=source_mask,
            tgt=target,
            tgt_key_padding_mask=target_mask,
            tgt_is_causal=True,
            tgt_mask=causal_mask,
        )


# Decoders.


class TransformerDecoder(TransformerModule):
    """Transformer decoder.

    Args:
        *args: passed to superclass.
        decoder_input_size (int).
        **kwargs: passed to superclass.
    """

    decoder_input_size: int
    # Constructed inside __init__.
    module: WrappedTransformerDecoder

    def __init__(
        self,
        *args,
        decoder_input_size: int = defaults.EMBEDDING_SIZE,
        **kwargs,
    ):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None,
        embeddings: nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs single pass of decoder module.

        Args:
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            target (torch.Tensor): current state of targets, which may be the
                full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor).
            embeddings (nn.Embedding)....

        Returns:
            tuple[torch.Tensor, torch.Tensor]: decoder outputs and the
                embedded targets.
        """
        target_embedded = self.embed(target, embeddings)
        causal_mask = self._causal_mask(target_embedded.size(1))
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

    def _causal_mask(self, target_len: int) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(
            target_len,
            device=self.device,
            dtype=bool,
        )

    @property
    def output_size(self) -> int:
        return self.embedding_size


class CausalTransformerDecoder(TransformerEncoder):
    """Decoder for the causal transformer.

    This borrows some implementation from the vanilla transformer encoder,
    even though it is used here as a decoder.
    """

    def forward(
        self,
        symbols: data.PaddedTensor,
        embeddings: nn.Embedding,
        mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encodes the symbols.

        This overrides the superclass definition to take a mask argument.

        Args:
            symbols (data.PaddedTensor).
            embeddings (nn.Embedding).
            mask (torch.Tensor).
            *args: ignored.
            **kwargs: ignored.

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.embed(symbols.tensor, embeddings)
        # Casts this to float.
        padding_mask = torch.where(
            symbols.mask,
            torch.full_like(symbols.mask, defaults.NEG_INF, dtype=torch.float),
            torch.zeros_like(symbols.mask, dtype=torch.float),
        )
        return self.module(
            embedded,
            mask=mask,
            src_key_padding_mask=padding_mask,
        )

    @property
    def name(self) -> str:
        return f"causal {super().name}"


class PointerGeneratorTransformerDecoder(TransformerDecoder):
    """A transformer decoder which tracks the output of multihead attention.

    This is achieved with a hook into the forward pass.

    After:
        https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91

    Args:
        *args: passed to superclass.
        has_features_encoder (bool, optional).
        *kwargs: passed to superclass.
    """

    def __init__(
        self,
        *args,
        has_features_encoder: bool = False,
        **kwargs,
    ):
        self.has_features_encoder = has_features_encoder
        super().__init__(*args, **kwargs)
        # Stores the actual cross attentions.
        self.attention_output = AttentionOutput()
        # Refers to the attention from decoder to encoder.
        self._patch_attention(self.module.layers[-1].multihead_attn)
        self.hook_handle = self.module.layers[
            -1
        ].multihead_attn.register_forward_hook(self.attention_output)

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None,
        embeddings: nn.Embedding,
        *,
        features_encoded: torch.Tensor | None = None,
        features_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs single pass of decoder module.

        Args:
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            target (torch.Tensor): current targets, which may be the full
                target or previous decoded, of shape B x seq_len x hidden_size.
            target_mask (torch.Tensor).
            embeddings (nn.Embedding).
            features_encoded (torch.Tensor, optional).
            features_mask (torch.Tensor, optional)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: decoder outputs and the
                embedded targets.
        """
        target_embedded = self.embed(target, embeddings)
        causal_mask = self._causal_mask(target_embedded.size(1))
        # -> B x seq_len x d_model.
        if self.has_features_encoder:
            decoded = self.module(
                source_encoded,
                source_mask,
                target_embedded,
                target_mask,
                features_encoded,
                features_mask,
                causal_mask,
            )
        else:
            decoded = self.module(
                source_encoded,
                source_mask,
                target_embedded,
                target_mask,
                causal_mask,
            )
        return decoded, target_embedded

    def get_module(self) -> nn.TransformerDecoder:
        if self.has_features_encoder:
            decoder_layer = (
                transformer_layers.SeparateFeaturesTransformerDecoderLayer(
                    d_model=self.decoder_input_size,
                    dim_feedforward=self.hidden_size,
                    nhead=self.attention_heads,
                    dropout=self.dropout,
                    activation="relu",
                    norm_first=True,
                    batch_first=True,
                )
            )
            return SeparateFeaturesTransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=self.layers,
                norm=nn.LayerNorm(self.embedding_size),
            )
        else:
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

    def _patch_attention(self, attention_module: torch.nn.Module) -> None:
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

    @property
    def name(self) -> str:
        return f"pointer-generator {super().name}"


class SeparateFeaturesTransformerDecoder(nn.TransformerDecoder):
    """Transformer decoder with separate features.

    Adding separate features into the transformer stack is implemented with
    SeparateFeaturesTransformerDecoderLayer.
    """

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None,
        features_encoded: torch.Tensor,
        features_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Passes the inputs (and mask) through the decoder layer.

        Args:
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            target (torch.Tensor): current embedded targets, which
                may be the full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor).
            features_encoded (torch.Tensor).
            features_mask (torch.Tensor).
            causal_mask (torch.Tensor).

        Returns:
            torch.Tensor: Output tensor.
        """
        output = target
        for layer in self.layers:
            output = layer(
                source_encoded,
                source_mask,
                output,
                target_mask,
                features_encoded,
                features_mask,
                causal_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


# Rotary decoders.


class RotaryTransformerDecoder(RotaryTransformerModule, TransformerDecoder):
    """Transformer decoder with rotary positional encodings.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def get_module(self) -> WrappedTransformerDecoder:
        rope = position.RotaryPositionalEncoding(
            embedding_size=self._rope_head_dim,
            max_length=self._rope_max_length,
        )
        decoder_layer = (
            transformer_layers.RotaryTransformerDecoderLayer(  # noqa: E501
                rope,
                d_model=self.decoder_input_size,
                dim_feedforward=self.hidden_size,
                nhead=self.attention_heads,
                dropout=self.dropout,
                activation="relu",
                norm_first=True,
                batch_first=True,
            )
        )
        return WrappedTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
        )

    @property
    def name(self) -> str:
        return "rotary transformer"


class RotaryCausalTransformerDecoder(
    RotaryTransformerModule, CausalTransformerDecoder
):
    """Causal transformer decoder with rotary positional encodings.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def get_module(self) -> nn.TransformerEncoder:
        rope = position.RotaryPositionalEncoding(
            embedding_size=self._rope_head_dim,
            max_length=self._rope_max_length,
        )
        encoder_layer = transformer_layers.RotaryTransformerEncoderLayer(
            rope,
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
            enable_nested_tensor=False,
        )

    @property
    def name(self) -> str:
        return "rotary causal transformer"


class RotaryPointerGeneratorTransformerDecoder(
    RotaryTransformerModule, PointerGeneratorTransformerDecoder
):
    """Pointer-generator transformer decoder with rotary positional encodings.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def get_module(self) -> nn.TransformerDecoder:
        rope = position.RotaryPositionalEncoding(
            embedding_size=self._rope_head_dim,
            max_length=self._rope_max_length,
        )
        if self.has_features_encoder:
            decoder_layer = transformer_layers.RotarySeparateFeaturesTransformerDecoderLayer(  # noqa: E501
                rope,
                d_model=self.decoder_input_size,
                dim_feedforward=self.hidden_size,
                nhead=self.attention_heads,
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
        else:
            decoder_layer = transformer_layers.RotaryTransformerDecoderLayer(
                rope,
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
        return "rotary pointer-generator transformer"
