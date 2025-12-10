"""Transformer module classes."""

import abc
import collections
import math
from typing import Optional, Tuple

import torch
from torch import nn

from ... import data, defaults, special
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
        *args: passed to superclass.
        attention_heads (int, optional): number of attention heads.
        dropout (float, optional): dropout probability.
        max_length (int, optional): max length for input.
        **kwargs: passed to superclass.
    """

    attention_heads: int
    esq: float
    dropout_layer: nn.Dropout
    module: nn.TransformerEncoder
    positional_encoding: position.PositionalEncoding

    def __init__(
        self,
        *args,
        attention_heads: int = defaults.ATTENTION_HEADS,
        dropout: float = defaults.DROPOUT,
        max_length: int = defaults.MAX_LENGTH,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention_heads = attention_heads
        self.dropout_layer = nn.Dropout(dropout)
        self.esq = math.sqrt(self.embedding_size)
        self.module = self.get_module()
        self.positional_encoding = position.PositionalEncoding(
            self.embedding_size, max_length
        )

    def embed(
        self, symbols: torch.Tensor, embeddings: nn.Embedding
    ) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings."""
        embedded = self.esq * embeddings(symbols)
        return self.dropout_layer(embedded + self.positional_encoding(symbols))

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

    def forward(
        self, source: data.PaddedTensor, embeddings: nn.Embedding
    ) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (data.PaddedTensor).
            embeddings (nn.Embedding).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.embed(source.padded, embeddings)
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

    The internal embedding is of size 1 because this is either source
    or features.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    # Constructed inside __init__.
    type_embedding: nn.Embedding

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_embedding = embeddings.xavier_embedding(
            2,
            self.embedding_size,
        )

    def embed(
        self, symbols: torch.Tensor, embeddings: nn.Embedding
    ) -> torch.Tensor:
        """Embeds the source symbols.

        This adds positional encodings and special embeddings.

        Args:
            source (data.PaddedTensor).
            embeddings (nn.Embedding).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        # "0" is whatever type we're using here; "1" is reserved for PAD.
        char_mask = (symbols == special.PAD_IDX).long()
        embedded = self.esq * embeddings(symbols)
        type_embedded = self.esq * self.type_embedding(char_mask)
        positional_embedded = self.positional_encoding(symbols, mask=char_mask)
        return self.dropout_layer(
            embedded + type_embedded + positional_embedded
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
        target: torch.Tensor,
        target_mask: torch.Tensor,
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


class TransformerDecoder(TransformerModule):
    """Transformer decoder.

    Args:
        decoder_input_size (int).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    decoder_input_size: int
    # Constructed inside __init__.
    module: WrappedTransformerDecoder

    def __init__(self, *args, decoder_input_size, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        embeddings: nn.Embedding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs single pass of decoder module.

        Args:
            source_encoded (torch.Tensor): encoded source sequence.
            source_mask (torch.Tensor): mask for source.
            target (torch.Tensor): current state of targets, which may be the
                full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): mask for target.
            embeddings (nn.Embedding): embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: decoder outputs and the
                embedded targets.
        """
        embedded = self.embed(target, embeddings)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            embedded.size(1),
            device=self.device,
            dtype=bool,
        )
        # -> B x seq_len x d_model.
        decoded = self.module(
            source_encoded,
            source_mask,
            embedded,
            target_mask,
            causal_mask,
        )
        return decoded, embedded

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

    Each decoding step gets a second multihead attention representation
    w.r.t. the encoded features. This and the original multihead attention
    representation w.r.t. the encoded symbols are then compressed in a
    linear layer and finally concatenated.

    The implementation is otherwise identical to nn.TransformerDecoderLayer.

    Args:
        *args: passed to superclass.
        *kwargs: passed to superclass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        factory_kwargs = {
            "device": kwargs.get("device"),
            "dtype": kwargs.get("dtype"),
        }
        d_model = kwargs["d_model"]
        self.features_multihead_attn = nn.MultiheadAttention(
            d_model,  # TODO: Separate feature embedding size?
            kwargs["nhead"],
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
        bias = kwargs.get("bias")
        self.source_linear = nn.Linear(
            d_model,
            d_model // 2,
            bias=bias,
            **factory_kwargs,
        )
        self.features_linear = nn.Linear(
            d_model,  # TODO: Separate feature embedding size?
            d_model // 2,
            bias=bias,
            **factory_kwargs,
        )

    def forward(
        self,
        source_encoded: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        features_encoded: torch.Tensor,
        features_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        This is based closely on the internals in torch.nn.transformer and
        follows the somewhat-inscrutable variable naming used there.

        Args:
            source_encoded (torch.Tensor): encoded source sequence.
            source_mask (torch.Tensor): mask for source.
            target (torch.Tensor): current embedded target, which
                may be the full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): mask for target.
            features_encoded (torch.Tensor): encoded features.
            features_mask (torch.Tensor): mask for features.
            causal_mask (torch.Tensor).

        Returns:
            torch.Tensor.
        """
        output = self.norm2(
            target
            + self._sa_block(
                self.norm1(target),
                causal_mask,
                target_mask,
                is_causal=True,
            )
        )
        source_attention = self.source_linear(
            self._mha_block(
                output,
                source_encoded,
                attn_mask=None,
                key_padding_mask=source_mask,
            )
        )
        features_attention = self.features_linear(
            self._features_mha_block(
                output,
                features_encoded,
                attn_mask=None,
                key_padding_mask=features_mask,
            ),
        )
        output = torch.cat((source_attention, features_attention), dim=2)
        output = output + self._ff_block(self.norm3(output))
        return output

    def _features_mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Multihead attention block that attends to features.

        This has the same interface as nn.TransformerDecoderLayer._mha_block.
        """
        output = self.features_multihead_attn(
            x,  # The output.
            mem,  # Encoded features.
            mem,  # Ditto.
            attn_mask=attn_mask,  # Causal mask.
            key_padding_mask=key_padding_mask,  # Features mask.
            is_causal=is_causal,  # False; no causal mask will be provided.
            need_weights=False,
        )[0]
        output = self.dropout2(output)
        return output


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
        target_mask: torch.Tensor,
        features_encoded: torch.Tensor,
        features_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Passes the inputs (and mask) through the decoder layer.

        Args:
            source_encoded (torch.Tensor): encoded source sequence.
            source_mask (torch.Tensor): mask for source.
            target (torch.Tensor): current embedded targets, which
                may be the full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): causal mask for target.
            features_encoded (torch.Tensor): encoded features.
            features_mask (torch.Tensor): mask for source.
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


class TransformerPointerDecoder(TransformerDecoder):
    """A transformer decoder which tracks the output of multihead attention.

    This is achieved with a hook into the forward pass.

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
        has_features_encoder: bool,
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
        target_mask: torch.Tensor,
        embeddings: nn.Embedding,
        features_encoded: Optional[torch.Tensor] = None,
        features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs single pass of decoder module.

        Args:
            source_encoded (torch.Tensor): encoded source sequence.
            source_mask (torch.Tensor): mask for source.
            target (torch.Tensor): current targets, which may be the full
                target or previous decoded, of shape B x seq_len x hidden_size.
            embeddings (nn.Embedding): embedding.
            features_encoded (Optional[torch.Tensor]): encoded features.
            features_mask (Optional[torch.Tensor]): mask for features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: decoder outputs and the
                embedded targets.
        """
        embedded = self.embed(target, embeddings)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            embedded.size(1),
            device=self.device,
            dtype=bool,
        )
        # -> B x seq_len x d_model.
        if self.has_features_encoder:
            decoded = self.module(
                source_encoded,
                source_mask,
                embedded,
                target_mask,
                features_encoded,
                features_mask,
                causal_mask,
            )
        else:
            decoded = self.module(
                source_encoded,
                source_mask,
                embedded,
                target_mask,
                causal_mask,
            )
        return decoded, embedded

    def get_module(self) -> nn.TransformerDecoder:
        if self.has_features_encoder:
            decoder_layer = SeparateFeaturesTransformerDecoderLayer(
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
