"""Symbol index."""

from __future__ import annotations

import itertools
import pickle

from typing import Dict, Iterable, List, Optional

from .. import defaults, special, util


class Error(Exception):
    pass


class Index:
    """Maintains the index over the vocabularies.

    Args:
        source_vocabulary (Iterable[str]).
        features_vocabulary (Iterable[str], optional).
        target_vocabulary (Iterable[str], optional).
        tie_embeddings: (bool).
    """

    source_vocabulary: List[str]
    target_vocabulary: List[str]
    features_vocabulary = Optional[List[str]]
    _index2symbol: List[str]
    _symbol2index: Dict[str, int]

    def __init__(
        self,
        *,
        source_vocabulary: Iterable[str],
        features_vocabulary: Optional[Iterable[str]] = None,
        target_vocabulary: Optional[Iterable[str]] = None,
        tie_embeddings: bool = defaults.TIE_EMBEDDINGS,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        # We store vocabularies separately for logging purposes.
        self.source_vocabulary = sorted(source_vocabulary)
        self.features_vocabulary = (
            sorted(features_vocabulary) if features_vocabulary else None
        )
        self.target_vocabulary = (
            sorted(target_vocabulary) if target_vocabulary else None
        )
        if self.tie_embeddings:
            # Vocabulary is the union of source and target.
            vocabulary = sorted(
                frozenset(
                    itertools.chain(source_vocabulary, target_vocabulary)
                )
            )
        else:
            # Vocabulary consists of target symbols followed by source symbols.
            vocabulary = self.target_vocabulary + self.source_vocabulary
        # FeatureInvariantTransformer assumes that features_vocabulary is at
        # the end of the vocabulary.
        if self.features_vocabulary is not None:
            vocabulary.extend(self.features_vocabulary)
        # Keeps special.SPECIAL first to maintain overlap with features.
        self._index2symbol = special.SPECIAL + vocabulary
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __len__(self) -> int:
        return len(self._index2symbol)

    # Lookup.

    def __call__(self, lookup: str) -> int:
        """Looks up index by symbol.

        Args:
            symbol (str).

        Returns:
            int.
        """
        return self._symbol2index.get(lookup, special.UNK_IDX)

    def get_symbol(self, index: int) -> str:
        """Looks up symbol by index.

        Args:
            index (int).

        Returns:
            str.
        """
        return self._index2symbol[index]

    # Serialization.

    @classmethod
    def read(cls, model_dir: str) -> Index:
        """Loads index.

        Args:
            model_dir (str).

        Returns:
            Index.
        """
        with open(cls.path(model_dir), "rb") as source:
            return pickle.load(source)

    def write(self, model_dir: str) -> None:
        """Writes index.

        Args:
            model_dir (str).
        """
        path = self.path(model_dir)
        util.mkpath(path)
        with open(path, "wb") as sink:
            pickle.dump(self, sink)

    @staticmethod
    def path(model_dir: str) -> str:
        """Computes path for the index file.

        Args:
            model_dir (str).

        Returns:
            str.
        """
        return f"{model_dir}/index.pkl"

    # Properties.

    @property
    def symbols(self) -> List[str]:
        return list(self._symbol2index.keys())

    @property
    def vocab_size(self) -> int:
        return len(self._symbol2index)

    @property
    def source_vocab_size(self) -> int:
        if self.tie_embeddings:
            return self.vocab_size
        else:
            return len(self.SPECIAL) + len(self.source_vocabulary)

    @property
    def features_vocab_size(self) -> int:
        return len(self.features_vocabulary) if self.features_vocabulary else 0

    @property
    def target_vocab_size(self) -> int:
        if self.tie_embeddings:
            return self.vocab_size
        elif self.target_vocabulary:
            return len(special.SPECIAL) + len(self.target_vocabulary)
        else:
            return 0
