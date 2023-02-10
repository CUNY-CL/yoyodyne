"""Positional encoding."""

import math
from typing import Optional

import torch
from torch import nn

from .. import defaults


class PositionalEncoding(nn.Module):
    """Positional encoding.

    After:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    pad_idx: int

    def __init__(
        self,
        d_model: int,
        pad_idx,
        max_source_length: int = defaults.MAX_SOURCE_LENGTH,
    ):
        """
        Args:
            d_model (int).
            pad_idx (int).
            max_source_length (int).
        """
        super(PositionalEncoding, self).__init__()
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
        out = out * pad_mask
        return out
