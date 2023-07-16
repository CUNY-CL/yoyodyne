"""Datasets and related utilities.

Anything which has a tensor member should inherit from nn.Module, run the
superclass constructor, and register the tensor as a buffer. This enables the
Trainer to move them to the appropriate device."""

from typing import List, Optional, Union

import torch
from torch import nn
from torch.utils import data

from .. import special

from . import indexes, tsv


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

    @property
    def has_features(self):
        return self.features is not None

    @property
    def has_target(self):
        return self.target is not None


class BaseDataset(data.Dataset):
    """Base datatset class."""

    samples: List[str]
    index: indexes.Index  # Usually copied.
    string_parser: tsv.StringParser  # Ditto.

    def __init__(self):
        super().__init__()

    @property
    def has_features(self) -> bool:
        return self.index.has_features

    @property
    def has_target(self) -> bool:
        return self.index.has_target


class DatasetNoFeatures(BaseDataset):
    """Dataset object without feature column."""

    def __init__(
        self,
        filename,
        tsv_parser,
        string_parser,
        index: indexes.Index,
    ):
        """Initializes the dataset.

        Args:
            filename (str): input filename.
            string_parser (tsv.StringParser).
            other (indexes.Index, optional): if provided,
                use this index to avoid recomputing it.
        """
        super().__init__()
        self.samples = list(tsv_parser.samples(filename))
        self.string_parser = string_parser
        self.index = index

    def encode(
        self,
        symbol_map: indexes.SymbolMap,
        word: List[str],
        add_start_tag: bool = True,
        add_end_tag: bool = True,
    ) -> torch.Tensor:
        """Encodes a sequence as a tensor of indices with word boundary IDs.

        Args:
            symbol_map (indexes.SymbolMap).
            word (List[str]): word to be encoded.
            add_start_tag (bool, optional): whether the sequence should be
                prepended with a start tag.
            add_end_tag (bool, optional): whether the sequence should be
                prepended with a end tag.

        Returns:
            torch.Tensor: the encoded tensor.
        """
        sequence = []
        if add_start_tag:
            sequence.append(special.START)
        sequence.extend(word)
        if add_end_tag:
            sequence.append(special.END)
        return torch.tensor(
            [
                symbol_map.index(symbol, self.index.unk_idx)
                for symbol in sequence
            ],
            dtype=torch.long,
        )

    def _decode(
        self,
        symbol_map: indexes.SymbolMap,
        indices: torch.Tensor,
        symbols: bool,
        special: bool,
    ) -> List[List[str]]:
        """Decodes the tensor of indices into symbols.

        Args:
            symbol_map (indexes.SymbolMap).
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool): whether to include the regular symbols when
                decoding the string.
            special (bool): whether to include the special symbols when
                decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """

        def include(c: int) -> bool:
            """Whether to include the symbol when decoding.

            Args:
                c (int): a single symbol index.

            Returns:
                bool: whether to include the symbol.
            """
            include = False
            is_special_char = c in self.index.special_idx
            if special:
                include |= is_special_char
            if symbols:
                # Symbols will be anything that is not SPECIAL.
                include |= not is_special_char
            return include

        decoded = []
        for index in indices.cpu().numpy():
            decoded.append([symbol_map.symbol(c) for c in index if include(c)])
        return decoded

    def decode_source(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of source indices, returns lists of symbols.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                vocabulary when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        return self._decode(
            self.index.source_map,
            indices,
            symbols=symbols,
            special=special,
        )

    def decode_target(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of target indices, returns lists of symbols.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            special (bool, optional): whether to include the regular symbol
                vocabulary when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        return self._decode(
            self.index.target_map,
            indices,
            symbols=symbols,
            special=special,
        )

    def __getitem__(self, idx: int) -> Item:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Item.
        """
        if self.has_target:
            source, target = self.samples[idx]
        else:
            source = self.samples[idx]
        source_encoded = self.encode(self.index.source_map, source)
        if self.has_target:
            target_encoded = self.encode(
                self.index.target_map, target, add_start_tag=False
            )
            return Item(source_encoded, target=target_encoded)
        else:
            return Item(source_encoded)

    def __len__(self) -> int:
        return len(self.samples)


class DatasetFeatures(DatasetNoFeatures):
    """Dataset object with feature column."""

    samples: List[str]
    index: indexes.Index  # Usually copied.
    string_parser: tsv.StringParser  # Ditto.

    def __getitem__(self, idx: int) -> Item:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Item.
        """
        if self.has_target:
            source, features, target = self.samples[idx]
        else:
            source, features = self.samples[idx]
        source_encoded = self.encode(self.index.source_map, source)
        features_encoded = self.encode(
            self.index.features_map,
            features,
            add_start_tag=False,
            add_end_tag=False,
        )
        if self.has_target:
            return Item(
                source_encoded,
                target=self.encode(
                    self.index.target_map, target, add_start_tag=False
                ),
                features=features_encoded,
            )
        else:
            return Item(source_encoded, features=features_encoded)

    def decode_features(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of feature indices, returns lists of symbols.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        return self._decode(
            self.index.features_map,
            indices,
            symbols=symbols,
            special=special,
        )


def get_dataset(
    filename: str,
    tsv_parser: tsv.TsvParser,
    string_parser: tsv.StringParser,
    index: Union[indexes.Index, str, None] = None,
) -> data.Dataset:
    """Dataset factory.

    Args:
        filename (str): input filename.
        tsv_parser (tsv.TsvParser).
        string_parser (tsv.StringParser).
        index (indexes.Index).

    Returns:
        data.Dataset: the dataset.
    """
    cls = DatasetFeatures if tsv_parser.has_features else DatasetNoFeatures
    return cls(filename, tsv_parser, string_parser, index)
