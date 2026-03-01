"""Positional encoding modules for transformers."""

import abc
import math

import lightning
import torch
from torch import nn

from ... import defaults, special


class Error(Exception):
    pass


class BasePositionalEncoding(abc.ABC, lightning.LightningModule):
    """Abstract base class for positional encodings."""

    @abc.abstractmethod
    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor: ...

    @property
    @abc.abstractmethod
    def max_length(self) -> int: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class AbsolutePositionalEncoding(BasePositionalEncoding):
    """Absolute positional encoding.

    Each position is associated with an embedding.

    Args:
        embedding_size (int, optional).
        max_length (int, optional).
    """

    def __init__(
        self,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        max_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        self.positional_encoding = nn.Embedding(max_length, embedding_size)

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            embedded (torch.Tensor): embedded sequence.
            symbol (torch.Tensor): symbol indices to encode.

        Returns:
            torch.Tensor: positional embedding.
        """
        indices = torch.arange(symbol.size(1), device=symbol.device)
        if indices.size(0) > self.max_length:
            raise Error(
                f"Sequence length {indices.size(0)} exceeds "
                f"max_length {self.max_length}"
            )
        out = self.positional_encoding(indices)
        out = out.expand(symbol.size(0), -1, -1)
        # Zeros out pads.
        out = out * symbol.ne(special.PAD_IDX).unsqueeze(2)
        return embedded + out

    @property
    def max_length(self) -> int:
        return self.positional_encoding.num_embeddings

    @property
    def name(self) -> str:
        return "absolute"


class NullPositionalEncoding(BasePositionalEncoding):
    """No-op positional encoding."""

    def __init__(self, *args, max_length: int = defaults.MAX_LENGTH, **kwargs):
        super().__init__()
        self._max_length = max_length

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        return embedded

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def name(self) -> str:
        return "null"


class RotaryPositionalEncoding(BasePositionalEncoding):
    """Rotary positional encoding (RoPE).

    Unlike additive schemes, RoPE is applied inside attention (to Q and K),
    not to the token embeddings themselves. This class therefore acts as a
    no-op in the embedding step and serves only as a container for the
    cos/sin cache and the rotation logic consumed by RoPE attention layers.

    After:
        Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. 2024.
        RoFormer: Enhanced transformer with rotary position embedding.
        _Neurocomputing_ 568: 127063.

    Args:
        embedding_size (int, optional): must be even.
        max_length (int, optional).
    """

    def __init__(
        self,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        max_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        if embedding_size % 2 != 0:
            raise Error(
                f"RoPE requires even embedding_size; got {embedding_size}"
            )
        self._max_length = max_length
        # Cos/sin tables tables of shape max_length x embedding_size // 2.
        half = embedding_size // 2
        theta = 1.0 / (
            10000.0 ** (torch.arange(0, half, dtype=torch.float) / half)
        )
        positions = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        freqs = positions * theta.unsqueeze(0)
        self.register_buffer("cos_table", freqs.cos())
        self.register_buffer("sin_table", freqs.sin())

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        """No-op: RoPE is applied inside attention, not to embeddings."""
        return embedded

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def name(self) -> str:
        return "rotary"

    def rotate(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies the RoPE rotation to a Q or K tensor.

        Args:
            x (torch.Tensor): shape (B, seq_len, num_heads, head_dim).
                Note: head_dim must equal embedding_size used at construction.
            seq_len (int): actual sequence length.

        Returns:
            torch.Tensor: same shape as x, with RoPE applied.
        """
        cos = self.cos_table[:seq_len, :]
        sin = self.sin_table[:seq_len, :]
        # Broadcasts over batch and heads: (1, seq_len, 1, half).
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=4
        )
        return rotated.flatten(3, 4)


class SinusoidalPositionalEncoding(BasePositionalEncoding):
    """Sinusoidal positional encoding.

    After:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html.

    Args:
        embedding_size (int, optional).
        max_length (int, optional).
    """

    def __init__(
        self,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        max_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        positional_encoding = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        scale_factor = -math.log(10000.0) / embedding_size
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float() * scale_factor
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            embedded (torch.Tensor): embedded sequence.
            symbol (torch.Tensor): symbol indices to encode.

        Returns:
            torch.Tensor: positional embedding.
        """
        indices = torch.arange(symbol.size(1), device=symbol.device)
        if indices.size(0) > self.max_length:
            raise Error(
                f"Sequence length {indices.size(0)} exceeds "
                f"max_length {self.max_length}"
            )
        out = self.positional_encoding[:, indices, :]
        out = out.expand(symbol.size(0), -1, -1)
        # Zeros out pads.
        out = out * symbol.ne(special.PAD_IDX).unsqueeze(2)
        return embedded + out

    @property
    def max_length(self) -> int:
        return self.positional_encoding.size(1)

    @property
    def name(self) -> str:
        return "sinusoidal"
