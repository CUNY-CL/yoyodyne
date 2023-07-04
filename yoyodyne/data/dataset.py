"""Datasets."""

from typing import Iterator, List, Optional

import torch
from torch import nn
from torch.utils import data

from . import index, tsv
from .. import special


class Item(nn.Module):
    """Source tensor, with optional features and target tensors.

    This represents a single item or observation."""

    source: torch.Tensor
    features: Optional[torch.Tensor]
    target: Optional[torch.Tensor]

    def __init__(self, source, features=None, target=None):
        """Initializes the item.

        Args:
            source (torch.Tensor).
            features (torch.Tensor, optional).
            target (torch.Tensor, optional).
        """
        super().__init__()
        self.register_buffer("source", source)
        self.register_buffer("features", features)
        self.register_buffer("target", target)


class Dataset(data.Dataset):
    """Dataset object without feature column."""

    samples: List[str]
    index: index.Index  # Normally a copy from the DataModule.
    cell_parser: tsv.CellParser  # Ditto.

    def __init__(
        self,
        samples,
        index,
        cell_parser,
    ):
        """Initializes the dataset.

        Args:
            samples (List[str]).
            index (indexes.Index).
            cell_parser (tsv.CellParser).
        """
        super().__init__()
        self.samples = samples
        self.index = index
        self.cell_parser = cell_parser

    # Encoding.

    def _encode(
        self,
        symbols: List[str],
        symbol_map: index.SymbolMap,
    ) -> torch.Tensor:
        """Encodes a sequence as a tensor of indices with cell boundary IDs.

        Args:
            cell (str): string to be encoded.
            sep (str): separator to use.
            symbol_map (indexes.SymbolMap): symbol map to encode with.
            add_start_tag (bool, optional): whether the sequence should be
                prepended with a start tag.
            add_end_tag (bool, optional): whether the sequence should be
                prepended with a end tag.

        Returns:
            torch.Tensor: the encoded tensor.
        """
        return torch.tensor(
            [
                symbol_map.index(symbol, self.index.unk_idx)
                for symbol in symbols
            ],
            dtype=torch.long,
        )

    def encode_source(self, cell: str) -> torch.Tensor:
        """Encodes a source string, padding with start and end tags.

        Args:
            cell (str).

        Returns:
            torch.Tensor.
        """
        wrapped = [special.START]
        wrapped.extend(self.cell_parser.source_symbols(cell))
        wrapped.append(special.END)
        return self._encode(wrapped, self.index.source_map)

    def encode_features(self, cell: str) -> torch.Tensor:
        """Encodes a features string.

        Args:
            cell (str).

        Returns:
            torch.Tensor.
        """
        return self._encode(
            self.cell_parser.features_symbols(cell), self.index.features_map
        )

    def encode_target(self, cell: str) -> torch.Tensor:
        """Encodes a features string, padding with end tags.

        Args:
            cell (str).

        Returns:
            torch.Tensor.
        """
        wrapped = self.cell_parser.target_symbols(cell)
        wrapped.append(special.END)
        return self._encode(wrapped, self.index.target_map)

    # Decoding.

    def _decode(
        self,
        indices: torch.Tensor,
        symbol_map: index.SymbolMap,
    ) -> Iterator[List[str]]:
        """Decodes the tensor of indices into lists of symbols.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbol_map (indexes.SymbolMap).

        Yields:
            List[str]: Decoded symbols.
        """
        for idx in indices.cpu().numpy():
            yield [
                symbol_map.symbol(c)
                for c in idx
                if idx not in self.index.special
            ]

    def decode_source(
        self,
        indices: torch.Tensor,
    ) -> Iterator[str]:
        """Decodes a source tensor.

        Args:
            indices (torch.Tensor): 2d tensor of indices.

        Yields:
            str: Decoded source strings.
        """
        for symbols in self._decode(indices, self.index.source_map):
            yield self.cell_parser.source_string(symbols)

    def decode_features(
        self,
        indices: torch.Tensor,
    ) -> Iterator[str]:
        """Decodes a features tensor.

        Args:
            indices (torch.Tensor): 2d tensor of indices.

        Yields:
            str: Decoded features strings.
        """
        for symbols in self._decode(indices, self.index.target_map):
            yield self.cell_parser.feature_string(symbols)

    def decode_target(
        self,
        indices: torch.Tensor,
    ) -> Iterator[str]:
        """Decodes a target tensor.

        Args:
            indices (torch.Tensor): 2d tensor of indices.

        Yields:
            str: Decoded target strings.
        """
        for symbols in self._decode(indices, self.index.target_map):
            yield self.cell_parser.target_string(symbols)

    # Required interface.

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Item:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Item.
        """
        if self.index.has_features:
            if self.index.has_target:
                source, features, target = self.samples[idx]
                return Item(
                    source=self.encode_source(source),
                    features=self.encode_features(features),
                    target=self.encode_target(target),
                )
            else:
                source, features = self.samples[idx]
                return Item(
                    source=self.encode_source(source),
                    features=self.encode_features(features),
                )
        elif self.index.has_target:
            source, target = self.samples[idx]
            return Item(
                source=self.encode_source(source),
                target=self.encode_target(target),
            )
        else:
            source = self.samples[idx]
            return Item(source=self.encode_source(source))
