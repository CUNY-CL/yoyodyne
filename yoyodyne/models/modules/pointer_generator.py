"""Pointer-generator module classes."""

import torch
from torch import nn

from . import position, transformer, transformer_layers


class PointerGeneratorTransformerDecoder(transformer.TransformerDecoder):
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
        self.attention_output = transformer.AttentionOutput()
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
            features_mask (torch.Tensor, optional).

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
            return transformer.SeparateFeaturesTransformerDecoder(
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
            return transformer.WrappedTransformerDecoder(
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


# Rotary decoders.


class RotaryPointerGeneratorTransformerDecoder(
    transformer.RotaryTransformerModule, PointerGeneratorTransformerDecoder
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
            return transformer.SeparateFeaturesTransformerDecoder(
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
            return transformer.WrappedTransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=self.layers,
                norm=nn.LayerNorm(self.embedding_size),
            )

    @property
    def name(self) -> str:
        return f"rotary {super().name}"
