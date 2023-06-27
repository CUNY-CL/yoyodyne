"""Indexes for symbols."""
import os
import pickle
from typing import Dict, List, Optional, Set

from . import special


class SymbolMap:
    """Tracks mapping from index to symbol and symbol to index."""

    index2symbol: List[str]
    symbol2index: Dict[str, int]

    def __init__(self, vocabulary: List[str]):
        # Keep special.SPECIAL first to maintain overlap with features.
        self._index2symbol = special.SPECIAL + vocabulary
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __len__(self) -> int:
        return len(self._index2symbol)

    def index(self, symbol: str, unk_idx: Optional[int] = None) -> int:
        """Looks up index by symbol.

        Args:
            symbol (str).
            unk_idx (int, optional): the <UNK> index, returned if the symbol
                is not found.
        Returns:
            int.
        """
        return self._symbol2index.get(symbol, unk_idx)

    def symbol(self, index: int) -> str:
        """Looks up symbol by index.

        Args:
            index (int).

        Returns:
            str.
        """
        return self._index2symbol[index]

    def pprint(self) -> str:
        """Pretty-prints the vocabulary."""
        return ", ".join(f"{c!r}" for c in self._index2symbol)


class BaseIndex:
    """Container for symbol maps."""

    # Serialization support.

    @classmethod
    def read(cls, path: str):
        """Loads symbol mappings.

        Args:
            path (str): input path.
        """
        index = cls.__new__(cls)
        with open(path, "rb") as source:
            dictionary = pickle.load(source)
        for key, value in dictionary.items():
            setattr(index, key, value)
        return index

    def write(self, path: str) -> None:
        """Saves index.

        Args:
            path (str): output path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as sink:
            pickle.dump(vars(self), sink)


class IndexNoFeatures(BaseIndex):
    """Container for symbol maps.

    For consistency, one is recommended to lexicographically sort the
    vocabularies ahead of time."""

    source_map: SymbolMap
    target_map: SymbolMap

    def __init__(
        self,
        source_vocabulary: List[str],
        target_vocabulary: List[str],
    ):
        """Initializes the index.

        Args:
            source_vocabulary (List[str]).
            target_vocabulary (List[str]).
        """
        super().__init__()
        self.source_map = SymbolMap(source_vocabulary)
        self.target_map = SymbolMap(target_vocabulary)

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
        return {self.unk_idx, self.pad_idx, self.start_idx, self.end_idx}

    @property
    def has_features(self) -> bool:
        return False


class IndexFeatures(IndexNoFeatures):
    """Also stores a feature index."""

    features_map: SymbolMap

    def __init__(
        self,
        source_vocabulary: List[str],
        features_vocabulary: List[str],
        target_vocabulary: List[str],
    ):
        """Initializes the index.

        Args:
            source_vocabulary (List[str]).
            features (List[str]).
            target_vocabulary (List[str]).
        """
        # We concatenate the source and feature vocabularies.
        super().__init__(
            source_vocabulary=source_vocabulary,
            target_vocabulary=target_vocabulary,
        )
        self.features_map = SymbolMap(features_vocabulary)

    @property
    def features_vocab_size(self) -> int:
        return len(self.features_map)

    @property
    def has_features(self) -> bool:
        return True
