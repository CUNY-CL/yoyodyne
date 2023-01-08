"""Dataset classes."""

import abc
import pickle
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.utils import data

from . import dataconfig, special


class Error(Exception):
    """Module-specific exception."""

    pass


class SymbolMap:
    """Tracks mapping from index to symbol and symbol to index."""

    index2symbol: List[str]
    symbol2index: Dict[str, int]

    def __init__(self, symbols: Iterable[str]):
        self._index2symbol = list(symbols)
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __len__(self) -> int:
        return len(self._index2symbol)

    def index(self, symbol: str, unk_idx: Optional[int] = None) -> int:
        """Looks up index by symbol."""
        return self._symbol2index.get(symbol, unk_idx)

    def symbol(self, index: int) -> str:
        """Looks up symbol by index."""
        return self._index2symbol[index]

    def pprint(self) -> str:
        """Pretty-prints the vocabulary."""
        return ", ".join(f"{c!r}" for c in self._index2symbol)


class BaseDataset(data.Dataset):
    """Base datatset class, with some core methods."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def _make_index(self) -> Dict:
        ...

    @abc.abstractmethod
    def _get_index(self) -> Dict:
        ...

    def _attach_index(self, index: Dict) -> None:
        """Attaches index variables to self."""
        for attr, value in index.items():
            setattr(self, attr, value)

    @staticmethod
    def _write_pkl(obj: Any, path: str) -> None:
        """Writes pickled object to path.

        Args:
            obj (Any): the object to be written.
            path (str): output path.
        """
        with open(path, "wb") as sink:
            pickle.dump(obj, sink)

    @staticmethod
    def _read_pkl(path: str) -> Any:
        """Reads a pickled object from the path.

        Args:
            path (str): input path.

        Returns:
            Any: the object read.
        """
        with open(path, "rb") as source:
            return pickle.load(source)

    def write_index(self, path: str) -> None:
        """Saves character mappings.

        Args:
            path (str): index path.
        """
        index = self._get_index()
        self._write_pkl(index, path)

    def read_index(self, path: str) -> None:
        """Loads character mappings.

        Args:
            index (str): index path.
        """
        index = self._read_pkl(path)
        self._attach_index(index)


class DatasetNoFeatures(BaseDataset):
    """Dataset object without feature column."""

    filename: str
    config: dataconfig.DataConfig
    samples: List
    source_map: SymbolMap
    target_map: SymbolMap

    def __init__(
        self,
        filename,
        config,
        other: Optional[BaseDataset] = None,
    ):
        """Initializes the dataset.

        Args:
            filename (str): input filename.
            config (dataconfig.DataConfig): dataset configuration.
            other (BaseDataset, optional): if provided, use the index from
                this dataset rather than recomputing one.
        """
        super().__init__()
        self.config = config
        self.samples = list(self.config.samples(filename))
        self._attach_index(
            other._get_index() if other is not None else self._make_index()
        )

    def _make_index(self) -> Dict:
        """Generates index."""
        # Ensures the idx of special symbols are identical in both indexs.
        special_vocabulary = special.SPECIAL.copy()
        target_vocabulary: Set[str] = set()
        source_vocabulary: Set[str] = set()
        if self.config.has_target:
            for source, target in self.samples:
                source_vocabulary.update(source)
                target_vocabulary.update(target)
            if self.config.tied_vocabulary:
                source_vocabulary.update(target_vocabulary)
                target_vocabulary.update(source_vocabulary)
        else:
            for source in self.samples:
                source_vocabulary.update(source)
        return {
            "source_map": SymbolMap(
                special_vocabulary + sorted(source_vocabulary)
            ),
            "target_map": SymbolMap(
                special_vocabulary + sorted(target_vocabulary)
            ),
        }

    def _get_index(self) -> Dict:
        return {
            "source_map": self.source_map,
            "target_map": self.target_map,
        }

    def encode(
        self,
        symbol_map: SymbolMap,
        word: List[str],
        add_start_tag: bool = True,
        add_end_tag: bool = True,
    ) -> torch.Tensor:
        """Encodes a sequence as a tensor of indices with word boundary IDs.

        Args:
            symbol_map (SymbolMap).
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
            [symbol_map.index(symbol, self.unk_idx) for symbol in sequence],
            dtype=torch.long,
        )

    def _decode(
        self,
        symbol_map: SymbolMap,
        indices: torch.Tensor,
        symbols: bool,
        special: bool,
    ) -> List[List[str]]:
        """Decodes the tensor of indices into characters.

        Args:
            symbol_map (SymbolMap).
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
                bool: if True, include the symbol.
            """
            include = False
            is_special_char = c in self.special_idx
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
        """Given a tensor of source indices, returns lists of characters.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                alphabet when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        return self._decode(
            self.source_map,
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
        """Given a tensor of target indices, returns lists of characters.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                alphabet when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        return self._decode(
            self.target_map,
            indices,
            symbols=symbols,
            special=special,
        )

    @property
    def source_vocab_size(self) -> int:
        return len(self.source_map)

    @property
    def target_vocab_size(self) -> int:
        return len(self.target_map)

    @property
    def pad_idx(self) -> int:
        return self.source_map.index(special.PAD)

    @property
    def start_idx(self) -> int:
        return self.source_map.index(special.START)

    @property
    def end_idx(self) -> int:
        return self.source_map.index(special.END)

    @property
    def unk_idx(self) -> int:
        return self.source_map.index(special.UNK)

    @property
    def special_idx(self) -> Set[int]:
        """The set of indexes for all `special` symbols."""
        return {self.unk_idx, self.pad_idx, self.start_idx, self.end_idx}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: source/target sample to be
                consumed by the model.
        """
        source, target = self.samples[idx]
        source_encoded = self.encode(self.source_map, source)
        target_encoded = (
            self.encode(self.target_map, target, add_start_tag=False)
            if self.config.has_target
            else None
        )
        return source_encoded, target_encoded


class DatasetFeatures(DatasetNoFeatures):
    """Dataset object with feature column."""

    features_idx: int

    def _make_index(self) -> None:
        """Generates index.

        Same as in superclass, but also handles features.
        """
        # Ensures the idx of special symbols are identical in both indexs.
        special_vocabulary = special.SPECIAL.copy()
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        target_vocabulary: Set[str] = set()
        if self.config.has_target:
            for source, features, target in self.samples:
                source_vocabulary.update(source)
                features_vocabulary.update(features)
                target_vocabulary.update(target)
            if self.config.tied_vocabulary:
                source_vocabulary.update(target_vocabulary)
                target_vocabulary.update(source_vocabulary)
        else:
            for source, features in self.samples:
                source_vocabulary.update(source)
                features_vocabulary.update(features)
        source_vocabulary = special_vocabulary + sorted(source_vocabulary)
        return {
            # Source and features index share embedding dict.
            "source_map": SymbolMap(
                source_vocabulary + sorted(features_vocabulary)
            ),
            "target_map": SymbolMap(
                special_vocabulary + sorted(target_vocabulary)
            ),
            # features_idx assists in indexing features.
            "features_idx": len(special_vocabulary) + len(source_vocabulary),
        }

    def _get_index(self) -> Dict:
        """Overridding to include features index."""
        index = super()._get_index()
        index["features_idx"] = self.features_idx
        return index

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                source/features/target sample to be consumed by the model.
        """
        source, features, target = self.samples[idx]
        source_encoded = self.encode(self.source_map, source)
        features_encoded = self.encode(
            self.source_map,
            features,
            add_start_tag=False,
            add_end_tag=False,
        )
        target_encoded = (
            self.encode(self.target_map, target, add_start_tag=False)
            if self.config.has_target
            else None
        )
        return source_encoded, features_encoded, target_encoded

    def decode_source(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of source indices, returns lists of characters.

        Overriding to prevent use of features encoding.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        # Masking features index.
        indices = torch.where(
            indices < self.features_idx,
            indices,
            self.pad_idx,
        )
        return self._decode(
            self.source_map,
            indices,
            symbols=symbols,
            special=special,
        )

    def decode_features(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of feature indices, returns lists of characters.

        This is simply an alias for using decode_source for features.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        # Masking source index.
        indices = torch.where(
            indices >= self.features_idx, indices, self.pad_idx
        )
        return super().decode_source(indices, symbols=symbols, special=special)

    @property
    def features_vocab_size(self) -> int:
        return len(self.source_map) - self.features_idx


def get_dataset(
    filename: str,
    config: dataconfig.DataConfig,
    other: Optional[BaseDataset] = None,
) -> data.Dataset:
    """Dataset factory.

    Args:
        filename (str): input filename.
        config (dataconfig.DataConfig): dataset configuration.
        other (BaseDataset, optional): if provided, use the index from this
            dataset rather than recomputing one.

    Returns:
        data.Dataset: the dataset.
    """
    cls = DatasetFeatures if config.has_features else DatasetNoFeatures
    return cls(filename, config, other)
