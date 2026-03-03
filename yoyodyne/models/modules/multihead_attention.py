"""Multihead attention module classes."""

import math
from typing import Optional, Tuple

import torch
from torch import nn

from . import position
from ... import defaults


class Error(Exception):
    pass


class RotaryMultiheadAttention(nn.Module):
    """Multi-head attention with rotary positional encoding (RoPE).

    This wraps nn.MultiheadAttention and intercepts the Q and K projections
    to apply rotary embeddings before the dot-product attention. Because
    nn.MultiheadAttention does not expose its internal projections
    cleanly, we project Q, K, and V ourselves and call the functional API.

    Args:
        embed_dim (int).
        rope (position.RotaryPositionalEncoding).
        attention_heads (int, optional)
        bias (bool, optional).
        dropout (float, optional).
    """

    def __init__(
        self,
        embed_dim: int,
        rope: position.RotaryPositionalEncoding,
        attention_heads: int = defaults.ATTENTION_HEADS,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = defaults.DROPOUT,
    ):
        super().__init__()
        if not batch_first:
            raise Error("RotaryMultiheadAttention requires batch_first")
        self.attention_heads = attention_heads
        self.batch_first = batch_first
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // attention_heads
        if self.head_dim * attention_heads != embed_dim:
            raise Error(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"attention_heads ({attention_heads})"
            )
        self.rope = rope
        self.in_proj_bias = None
        self.in_proj_weight = None
        # Projections mirroring nn.MultiheadAttention internals.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Initializes parameters.
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes RoPE-augmented multi-head attention.

        Args:
            query (torch.Tensor): B x query_length x embed_dim.
            key (torch.Tensor): B x source_length x embed_dim.
            value (torch.Tensor): B x source_length x embed_dim.
            key_padding_mask (torch.Tensor, optional): B x source_length.
            need_weights (bool, optional): whether to return attention weights.
            attn_mask (torch.Tensor, optional): additive or bool mask.
            is_causal (bool, optional): if True and attn_mask is not provided,
                applies a causal mask; ignored otherwise.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: output and optionally
                attention weights.
        """
        batch_size = query.size(0)
        query_length = query.size(1)
        source_length = key.size(1)
        # Generates a causal mask if requested and none was provided.
        if is_causal and attn_mask is None:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                query_length,
                device=query.device,
                dtype=torch.bool,
            )
        # -> B x seq_len x attention_heads x head_dim.
        q = self.q_proj(query).view(
            batch_size, query_length, self.attention_heads, self.head_dim
        )
        k = self.k_proj(key).view(
            batch_size, source_length, self.attention_heads, self.head_dim
        )
        v = self.v_proj(value).view(
            batch_size, source_length, self.attention_heads, self.head_dim
        )
        # Rotates Q and K via RoPE.
        q = self.rope.rotate(q, query_length)
        k = self.rope.rotate(k, source_length)
        #
        # -> B * attention_heads x seq_len x head_dim.
        q = q.permute(0, 2, 1, 3).reshape(
            batch_size * self.attention_heads, query_length, self.head_dim
        )
        k = k.permute(0, 2, 1, 3).reshape(
            batch_size * self.attention_heads, source_length, self.head_dim
        )
        v = v.permute(0, 2, 1, 3).reshape(
            batch_size * self.attention_heads, source_length, self.head_dim
        )
        if key_padding_mask is not None:
            additive_kpm = (
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .expand(
                    batch_size,
                    self.attention_heads,
                    query_length,
                    source_length,
                )
                .reshape(
                    batch_size * self.attention_heads,
                    query_length,
                    source_length,
                )
            )
            additive_kpm = torch.zeros_like(
                additive_kpm, dtype=q.dtype
            ).masked_fill(additive_kpm.to(torch.bool), defaults.NEG_INF)
        else:
            additive_kpm = None
        if attn_mask is not None:
            attn_mask_f = (
                torch.zeros(
                    attn_mask.shape, dtype=q.dtype, device=q.device
                ).masked_fill(attn_mask, defaults.NEG_INF)
                if attn_mask.dtype == torch.bool
                else attn_mask
            )
            combined_mask = (
                attn_mask_f
                if additive_kpm is None
                else attn_mask_f + additive_kpm
            )
        else:
            combined_mask = additive_kpm
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(
            self.head_dim
        )
        if combined_mask is not None:
            attn_weights = attn_weights + combined_mask
        attn_weights = torch.softmax(attn_weights, dim=2)
        if self.training and self.dropout > 0.0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout)
        # -> B * attention_heads x query_length x head_dim.
        attn_output = torch.bmm(attn_weights, v)
        attn_output = (
            attn_output.reshape(
                batch_size, self.attention_heads, query_length, self.head_dim
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_length, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        if need_weights:
            # Returns per-head weights averaged over heads.
            # -> B x query_length x source_length.
            weights = attn_weights.reshape(
                batch_size, self.attention_heads, query_length, source_length
            ).mean(dim=1)
            return attn_output, weights
        return attn_output, None
