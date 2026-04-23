"""Transformer layer classes."""

import torch
from torch import nn

from ... import defaults
from . import multihead_attention, position


class Error(Exception):
    pass


# Encoder layers.


class RotaryTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer using rotary positional encoding.

    Args:
        rope (position.RotaryPositionalEncoding).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def __init__(
        self,
        rope: position.RotaryPositionalEncoding,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        dropout = kwargs.get("dropout", defaults.DROPOUT)
        bias = kwargs.get("bias", True)
        # Replaces the self-attention module.
        self.self_attn = multihead_attention.RotaryMultiheadAttention(
            embed_dim=d_model,
            attention_heads=nhead,
            rope=rope,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )


# Decoder layers.


class RotaryTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """TransformerDecoderLayer using rotary positional encoding.

    Replaces both self-attention and cross-attention with rotary multihead
    attention.

    Args:
        rope (position.RotaryPositionalEncoding).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def __init__(
        self,
        rope: position.RotaryPositionalEncoding,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        dropout = kwargs.get("dropout", defaults.DROPOUT)
        bias = kwargs.get("bias", True)
        self.self_attn = multihead_attention.RotaryMultiheadAttention(
            embed_dim=d_model,
            attention_heads=nhead,
            rope=rope,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )
        self.multihead_attn = multihead_attention.RotaryMultiheadAttention(
            embed_dim=d_model,
            attention_heads=nhead,
            rope=rope,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )


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
        target_mask: torch.Tensor | None,
        features_encoded: torch.Tensor,
        features_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        This is based closely on the internals in torch.nn.transformer and
        follows the somewhat-inscrutable variable naming used there.

        Args:
            source_encoded (torch.Tensor).
            source_mask (torch.Tensor).
            target (torch.Tensor): current embedded target, which
                may be the full target or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor).
            features_encoded (torch.Tensor).
            features_mask (torch.Tensor).
            causal_mask (torch.Tensor).

        Returns:
            torch.Tensor.
        """
        # Self-attention (pre-norm).
        x = self.norm1(target)
        sa_output, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=target_mask,
        )
        output = target + self.dropout1(sa_output)
        # Cross-attention (pre-norm).
        x = self.norm2(output)
        # Cross-attends to source.
        source_attn_out, _ = self.multihead_attn(
            x, source_encoded, source_encoded, key_padding_mask=source_mask
        )
        source_attention = self.source_linear(self.dropout2(source_attn_out))
        # Cross-attends to features.
        feat_attn_out, _ = self.features_multihead_attn(
            x,
            features_encoded,
            features_encoded,
            key_padding_mask=features_mask,
        )
        features_attention = self.features_linear(self.dropout2(feat_attn_out))
        # Concatenates and adds residual connection.
        attn_output = torch.cat((source_attention, features_attention), dim=2)
        output = output + attn_output
        # Feed-forward.
        # Normalizes before the feed-forward network.
        output = output + self._ff_block(self.norm3(output))
        return output


class RotarySeparateFeaturesTransformerDecoderLayer(
    SeparateFeaturesTransformerDecoderLayer
):
    """SeparateFeaturesTransformerDecoderLayer using RoPE.

    Args:
        rope (position.RotaryPositionalEncoding).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    def __init__(
        self,
        rope: position.RotaryPositionalEncoding,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        dropout = kwargs.get("dropout", defaults.DROPOUT)
        bias = kwargs.get("bias", True)
        self.self_attn = multihead_attention.RotaryMultiheadAttention(
            embed_dim=d_model,
            attention_heads=nhead,
            rope=rope,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )
        self.multihead_attn = multihead_attention.RotaryMultiheadAttention(
            embed_dim=d_model,
            attention_heads=nhead,
            rope=rope,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )
        self.features_multihead_attn = (
            multihead_attention.RotaryMultiheadAttention(
                embed_dim=d_model,
                attention_heads=nhead,
                rope=rope,
                dropout=dropout,
                batch_first=True,
                bias=bias,
            )
        )
