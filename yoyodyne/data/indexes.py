"""Symbol index."""

import os
import pickle
from typing import Dict, List, Optional, Set

from .. import defaults, special


class Error(Exception):
    pass


class Index:
    """Maintains the index over the vocabularies.

    For consistency, one is recommended to lexicographically sort the
    vocabularies ahead of time."""

    index2symbol: List[str]
    symbol2index: Dict[str, int]

    def __init__(
        self,
        *,
        source_vocabulary: List[str],
        features_vocabulary: Optional[List[str]] = None,
        target_vocabulary: Optional[List[str]] = None,
        tie_embeddings: bool = defaults.TIE_EMBEDDINGS,
    ):
        """Initializes the index.

        Args:
            source_vocabulary (List[str]).
            features_vocabulary (List[str], optional).
            target_vocabulary (List[str], optional).
            tie_embeddings: (bool).
        """
        super().__init__()
        self.tie_embeddings = tie_embeddings
        # We store all separate vocabularies for logging purposes.
        # If embeddings are tied, so are the vocab items.
        # Then, the source and target vocabularies are the union.
        if self.tie_embeddings:
            vocabulary = list(
                sorted(set(source_vocabulary) | set(target_vocabulary))
            )
            self.source_vocabulary = special.SPECIAL + vocabulary
            self.target_vocabulary = special.SPECIAL + vocabulary
        else:
            # If not tie_embeddings, then the target vocabulary must come
            # first so that output predictions correctly index our
            # vocabulary and embeddiings matrix
            vocabulary = sorted(target_vocabulary) + sorted(source_vocabulary)
            self.source_vocabulary = special.SPECIAL + source_vocabulary
            self.target_vocabulary = special.SPECIAL + target_vocabulary
        self.features_vocabulary = features_vocabulary
        # NOTE: features_vocabulary must be at the end of the List for
        # FeatureInvariantTransformer to work.
        vocabulary += sorted(features_vocabulary)
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
        return self._symbol2index.get(lookup, self.unk_idx)

    def get_symbol(self, index: int) -> str:
        """Looks up symbol by index.

        Args:
            index (int).

        Returns:
            str.
        """
        return self._index2symbol[index]

    def pprint(self) -> str:
        """Pretty-prints the full vocabulary."""
        return ", ".join(f"{c!r}" for c in self._index2symbol)

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
        return len(self.source_vocabulary)

    @property
    def target_vocab_size(self) -> int:
        return len(self.target_vocabulary) if self.target_vocabulary else 0

    @property
    def features_vocab_size(self) -> int:
        return len(self.features_vocabulary) if self.features_vocabulary else 0

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

    @property
    def special_idx(self) -> Set[int]:
        return {self.unk_idx, self.pad_idx, self.start_idx, self.end_idx}
