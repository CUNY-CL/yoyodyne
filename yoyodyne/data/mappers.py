"""Encodes and decodes tensors."""

from __future__ import annotations

import dataclasses

from typing import Iterable, List

import torch

from . import indexes
from .. import special


@dataclasses.dataclass
class Mapper:
    """Handles mapping between strings and tensors."""

    index: indexes.Index  # Usually copied from the DataModule.

    @classmethod
    def read(cls, model_dir: str) -> Mapper:
        """Loads mapper from an index.

        Args:
            model_dir (str).

        Returns:
            Mapper.
        """
        return cls(indexes.Index.read(model_dir))

    # Encoding.

    def _encode(self, symbols: Iterable[str]):
        """Encodes a tensor.

        Args:
            ymbols (Iterable[str]).

        Returns:
            torch.Tensor: the encoded tensor.
        """
        return torch.tensor([self.index(symbol) for symbol in symbols])

    def encode_source(self, symbols: Iterable[str]) -> torch.Tensor:
        """Encodes a source string, padding with start and end tags.

        Args:
            symbols (Iterable[str]).

        Returns:
            torch.Tensor.
        """
        wrapped = [special.START]
        wrapped.extend(symbols)
        wrapped.append(special.END)
        return self._encode(wrapped)

    def encode_features(self, symbols: Iterable[str]) -> torch.Tensor:
        """Encodes a features string.

        Args:
            symbols (Iterable[str]).

        Returns:
            torch.Tensor.
        """
        return self._encode(symbols)

    def encode_target(self, symbols: Iterable[str]) -> torch.Tensor:
        """Encodes a features string, padding with end tags.

        Args:
            symbols (Iterable[str]).

        Returns:
            torch.Tensor.
        """
        wrapped = list(symbols)
        wrapped.append(special.END)
        return self._encode(wrapped)

    # Decoding.

    def _decode(
        self,
        indices: torch.Tensor,
    ) -> List[str]:
        """Decodes a tensor.

        Decoding halts at END; other special symbols are omitted.

        Args:
            indices (torch.Tensor): 1d tensor of indices.

        Yields:
            List[str]: Decoded symbols.
        """
        symbols = []
        for idx in indices:
            if idx == special.END_IDX:
                return symbols
            elif not special.isspecial(idx):
                symbols.append(self.index.get_symbol(idx))
        return symbols

    # These are here for compatibility; they all have the same implementation.

    def decode_source(
        self,
        indices: torch.Tensor,
    ) -> List[str]:
        return self._decode(indices)

    def decode_features(
        self,
        indices: torch.Tensor,
    ) -> List[str]:
        return self._decode(indices)

    def decode_target(
        self,
        indices: torch.Tensor,
    ) -> List[str]:
        return self._decode(indices)
