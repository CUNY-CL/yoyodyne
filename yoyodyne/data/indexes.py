"""Symbol index."""

import os
import math
import pickle
from typing import Dict, List, Optional, Set
from collections import Counter
from .. import special


class SymbolMap:
    """Tracks mapping from index to symbol and symbol to index."""

    index2symbol: List[str]
    symbol2index: Dict[str, int]

    def __init__(self, vocabulary: Counter[str, int], p=1.0):
        # Filters to cover p% of vocabulary.
        n = len(vocabulary)
        print(n)
        vocab = [c for (c, _) in vocabulary.most_common(int(math.ceil(p * n)))]
        vocab.sort()
        print(len(vocab))
        # Keeps special.SPECIAL first to maintain overlap with features.
        self._index2symbol = special.SPECIAL + vocab
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


class Index:
    """Container for symbol maps.

    For consistency, one is recommended to lexicographically sort the
    vocabularies ahead of time."""

    source_map: SymbolMap
    target_map: SymbolMap
    features_map: Optional[SymbolMap]

    def __init__(
        self,
        *,
        source_vocabulary: Counter[str, int],
        source_coverage: Optional[float] = 1.0,
        features_vocabulary: Optional[Counter[str, int]] = None,
        features_coverage: Optional[float] = 1.0,
        target_vocabulary: Optional[Counter[str, int]] = None,
        target_coverage: Optional[float] = 1.0,
    ):
        """Initializes the index.

        Args:
            source_vocabulary (Counter[str]).
            source_coverage (float, optional): Percent of tokens coverd
            by source_vocabulary.
                Default: 1.0 (All tokens are present)
            features_vocabulary (Counter[str], optional): Percent of tokens
            coverd by features_vocabulary.
                Default: 1.0 (All tokens are present)
            target_vocabulary (Counter[str], optional): Percent of tokens
            coverd by target_vocabulary.
                Default: 1.0 (All tokens are present)
        """
        super().__init__()
        self.source_map = SymbolMap(source_vocabulary, p=source_coverage)
        self.features_map = (
            SymbolMap(features_vocabulary, p=features_coverage)
            if features_vocabulary
            else None
        )
        self.target_map = (
            SymbolMap(target_vocabulary, p=target_coverage)
            if target_vocabulary
            else None
        )

    # Serialization support.

    @classmethod
    def read(cls, model_dir: str, experiment: str) -> "Index":
        """Loads index.

        Args:
            model_dir (str).
            experiment (str).

        Returns:
            Index.
        """
        index = cls.__new__(cls)
        path = index.index_path(model_dir, experiment)
        with open(path, "rb") as source:
            dictionary = pickle.load(source)
        for key, value in dictionary.items():
            setattr(index, key, value)
        return index

    @staticmethod
    def index_path(model_dir: str, experiment: str) -> str:
        """Computes path for the index file.

        Args:
            model_dir (str).
            experiment (str).

        Returns:
            str.
        """
        return f"{model_dir}/{experiment}/index.pkl"

    def write(self, model_dir: str, experiment: str) -> None:
        """Writes index.

        Args:
            model_dir (str).
            experiment (str).
        """
        path = self.index_path(model_dir, experiment)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as sink:
            pickle.dump(vars(self), sink)

    # Properties.

    @property
    def source_vocab_size(self) -> int:
        return len(self.source_map)

    @property
    def has_features(self) -> bool:
        return self.features_map is not None

    @property
    def features_vocab_size(self) -> int:
        return len(self.features_map) if self.has_features else 0

    @property
    def has_target(self) -> bool:
        return self.target_map is not None

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
