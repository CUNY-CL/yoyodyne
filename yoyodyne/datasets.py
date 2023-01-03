"""Dataset classes."""

import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils import data

from . import dataconfig, special


class Error(Exception):
    """Module-specific exception."""

    pass


class DatasetNoFeatures(data.Dataset):
    """Dataset object without feature column."""

    filename: str
    config: dataconfig.DataConfig
    source_symbol2i: Dict[str, int]
    source_i2symbol: List[str]
    target_symbol2i: Dict[str, int]
    target_i2symbol: List[str]

    def __init__(
        self,
        filename,
        config,
    ):
        """Initializes the dataset.

        Args:
            filename (str): input filename.
            config (dataconfig.DataConfig): dataset configuration.
        """
        super().__init__()
        self.config = config
        self.samples = list(self.config.samples(filename))
        self._make_indices()

    def _make_indices(self) -> None:
        """Generates Dicts for encoding/decoding symbols as unique indices."""
        # Ensures the idx of special symbols are identical in both vocabs.
        special_vocabulary = special.SPECIAL.copy()
        target_vocabulary: Set[str] = set()
        source_vocabulary: Set[str] = set()
        if self.config.has_targets:
            for source, target in self.samples:
                source_vocabulary.update(source)
                target_vocabulary.update(target)
            if self.config.tied_vocabulary:
                source_vocabulary.update(target_vocabulary)
                target_vocabulary.update(source_vocabulary)
        else:
            for source in self.samples:
                source_vocabulary.update(source)
        self.source_symbol2i = {
            c: i
            for i, c in enumerate(
                special_vocabulary + sorted(source_vocabulary)
            )
        }
        self.source_i2symbol = list(self.source_symbol2i.keys())
        self.target_symbol2i = {
            c: i
            for i, c in enumerate(
                special_vocabulary + sorted(target_vocabulary)
            )
        }
        self.target_i2symbol = list(self.target_symbol2i.keys())

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

    def write_index(self, outdir: str, filename: str) -> None:
        """Saves character mappings.

        Args:
            outdir (str): output directory.
            filename (str): output filename.
        """
        vocab = {
            "source_symbol2i": self.source_symbol2i,
            "source_i2symbol": self.source_i2symbol,
            "target_symbol2i": self.target_symbol2i,
            "target_i2symbol": self.target_i2symbol,
        }
        self._write_pkl(vocab, os.path.join(outdir, f"{filename}_vocab.pkl"))

    def load_index(self, indir: str, filename: str) -> None:
        """Loads character mappings.

        Args:
            indir (str): input directory.
            filename (str): input filename.
        """
        vocab = self._read_pkl(os.path.join(indir, f"{filename}_vocab.pkl"))
        self.source_symbol2i = vocab["source_symbol2i"]
        self.source_i2symbol = vocab["source_i2symbol"]
        self.target_symbol2i = vocab["target_symbol2i"]
        self.target_i2symbol = vocab["target_i2symbol"]

    def encode(
        self,
        symbol2i: Dict,
        word: List[str],
        add_start_tag: bool = True,
        add_end_tag: bool = True,
    ) -> torch.Tensor:
        """Encodes a sequence as a tensor of indices with word boundary IDs.

        Args:
            symbol2i (Dict).
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
        return torch.LongTensor(
            [symbol2i.get(symbol, self.unk_idx) for symbol in sequence]
        )

    def _decode(
        self,
        indices: torch.Tensor,
        decoder: List[str],
        symbols: bool,
        special: bool,
    ) -> List[List[str]]:
        """Decodes the tensor of indices into characters.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            decoder (List[str]): decoding lookup table.
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
            decoded.append([decoder[c] for c in index if include(c)])
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
            indices,
            decoder=self.source_i2symbol,
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
            indices,
            decoder=self.target_i2symbol,
            symbols=symbols,
            special=special,
        )

    @property
    def source_vocab_size(self) -> int:
        return len(self.source_symbol2i)

    @property
    def target_vocab_size(self) -> int:
        return len(self.target_symbol2i)

    @property
    def pad_idx(self) -> int:
        return self.source_symbol2i[special.PAD]

    @property
    def start_idx(self) -> int:
        return self.source_symbol2i[special.START]

    @property
    def end_idx(self) -> int:
        return self.source_symbol2i[special.END]

    @property
    def unk_idx(self) -> int:
        return self.source_symbol2i[special.UNK]

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
        source_encoded = self.encode(self.source_symbol2i, source)
        target_encoded = (
            self.encode(self.target_symbol2i, target, add_start_tag=False)
            if self.config.has_targets
            else None
        )
        return source_encoded, target_encoded


class DatasetFeatures(DatasetNoFeatures):
    """Dataset object with feature column."""

    features_idx: int

    def _make_indices(self) -> None:
        """Generates unique indices dictionaries.

        Same as in superclass, but also handles features.
        """
        # Ensures the idx of special symbols are identical in both vocabs.
        special_vocabulary = special.SPECIAL.copy()
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        target_vocabulary: Set[str] = set()
        if self.config.has_targets:
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
        # Source and features vocab share embedding dict.
        # features_idx assists in indexing features.
        self.features_idx = len(source_vocabulary)
        self.source_symbol2i = {
            c: i
            for i, c in enumerate(
                source_vocabulary + sorted(features_vocabulary)
            )
        }
        self.source_i2symbol = list(self.source_symbol2i.keys())
        self.target_symbol2i = {
            c: i
            for i, c in enumerate(
                special_vocabulary + sorted(target_vocabulary)
            )
        }
        self.target_i2symbol = list(self.target_symbol2i.keys())

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
        source_encoded = self.encode(self.source_symbol2i, source)
        features_encoded = self.encode_features(features)
        target_encoded = (
            self.encode(self.target_symbol2i, target, add_start_tag=False)
            if self.config.has_targets
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
        # Masking features vocab.
        indices = torch.where(
            indices < self.features_idx,
            indices,
            self.pad_idx,
        )
        return self._decode(
            indices,
            decoder=self.source_i2symbol,
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
        # Masking source vocab.
        indices = torch.where(
            indices >= self.features_idx, indices, self.pad_idx
        )
        return super().decode_source(indices, symbols=symbols, special=special)

    @property
    def features_vocab_size(self) -> int:
        return len(self.source_symbol2i) - self.features_idx

    def write_index(self, outdir: str, filename: str) -> None:
        # Overwrites method to save features encoding.
        vocab = {
            "source_symbol2i": self.source_symbol2i,
            "source_i2symbol": self.source_i2symbol,
            "target_symbol2i": self.target_symbol2i,
            "target_i2symbol": self.target_i2symbol,
            "features_idx": self.features_idx,
        }
        self._write_pkl(vocab, os.path.join(outdir, f"{filename}_vocab.pkl"))

    def load_index(self, indir: str, filename: str) -> None:
        # Overwrites method to load features encoding.
        vocab = self._read_pkl(os.path.join(indir, f"{filename}_vocab.pkl"))
        self.source_symbol2i = vocab["source_symbol2i"]
        self.source_i2symbol = vocab["source_i2symbol"]
        self.target_symbol2i = vocab["target_symbol2i"]
        self.target_i2symbol = vocab["target_i2symbol"]
        self.features_idx = vocab["features_idx"]


def get_dataset_cls(include_features: bool) -> data.Dataset:
    """Dataset factory.

    Args:
        include_features (bool).

    Returns:
        data.Dataset: the dataset.
    """
    return DatasetFeatures if include_features else DatasetNoFeatures
