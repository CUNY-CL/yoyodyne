"""Symbol index."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Optional

from .. import defaults, special


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
        self.target_vocabulary = sorted(target_vocabulary)
        if self.tie_embeddings:
            # Vocabulary is the union of source and target.
            vocabulary = sorted(
                frozenset(source_vocabulary + target_vocabulary)
            )
        else:
            # Vocabulary consists of target symbols followed by source symbols.
            vocabulary = sorted(target_vocabulary) + sorted(source_vocabulary)
        # FeatureInvariantTransformer assumes that features_vocabulary is at
        # the end of the vocabulary.
        if features_vocabulary is not None:
            self.features_vocabulary = sorted(features_vocabulary)
            vocabulary.extend(self.features_vocabulary)
        else:
            self.features_vocabulary = None
        # Keeps special.SPECIAL first to maintain overlap with features.
        self._index2symbol = special.SPECIAL + vocabulary
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __len__(self) -> int:
        return len(self._index2symbol)

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

    # Serialization support.

    @classmethod
    def read(cls, model_dir: str, experiment: str) -> Index:
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

    @property
    def symbols(self) -> List[str]:
        return list(self._symbol2index.keys())

    @property
    def has_features(self) -> bool:
        return self.features_vocab_size > 0

    @property
    def has_target(self) -> bool:
        return self.target_vocab_size > 0

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
    def target_vocab_size(self) -> int:
        if self.tie_embeddings:
            return self.vocab_size
        elif self.target_vocabulary:
            return len(special.SPECIAL) + len(self.target_vocabulary)
        else:
            return 0

    @property
    def features_vocab_size(self) -> int:
        return len(self.features_vocabulary) if self.features_vocabulary else 0

    # These are also recorded in the `special` module.

    @property
    def pad_idx(self) -> int:
        return self._symbol2index[special.PAD]

    @property
    def start_idx(self) -> int:
        return self._symbol2index[special.START]

    @property
    def end_idx(self) -> int:
        return self._symbol2index[special.END]

    @property
    def unk_idx(self) -> int:
        return self._symbol2index[special.UNK]
