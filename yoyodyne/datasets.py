"""Dataset classes."""

import csv
import os
import pickle
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.utils import data

from . import special


class Error(Exception):
    """Module-specific exception."""

    pass


class DatasetNoFeatures(data.Dataset):
    """Dataset object.

    The user specifies:

    * an input filename path
    * the 1-based indices for the columns (defaults: source is 1, target is 2)
    * separator characters used to split the input columns strings, with an
      empty string used to indicate that the string should be split into
      Unicode characters

    These together define an enormous set of possibilities; the defaults
    correspond to the SIGMORPHON 2017 data format.
    """

    filename: str
    source_col: int
    target_col: int
    source_sep: str
    target_sep: str
    source_symbol2i: Dict[str, int]
    source_i2symbol: List[str]
    target_symbol2i: Dict[str, int]
    target_i2symbol: List[str]

    def __init__(
        self,
        filename,
        *,
        tied_vocabulary,
        source_col,
        target_col,
        source_sep,
        target_sep,
        **kwargs,
    ):
        """Initializes the dataset.

        Args:
            filename (str): input filename.
            tied_vocabulary (bool): whether the source and target should
                share a vocabulary.
            source_col (int): 1-indexed column in TSV containing
                source strings.
            target_col (int): 1-indexed column in TSV containing
                target strings.
            source_sep (str): separator character between symbols in source
                string. "" treats each character in source as a symbol.
            target_sep (str): separator character between symbols in target
                string. "" treats each character in target as a symbol.
            **kwargs: ignored.
        """
        if source_col < 1:
            raise Error(f"Invalid source column: {source_col}")
        self.source_col = source_col
        if target_col < 0:
            raise Error(f"Invalid target column: {target_col}")
        self.target_col = target_col
        self.source_sep = source_sep
        self.target_sep = target_sep
        self.samples = list(self._iter_samples(filename))
        self._make_indices(tied_vocabulary)

    @staticmethod
    def _get_cell(row: List[str], col: int, sep: str) -> List[str]:
        """Returns the split cell of a row.

        Args:
           row (List[str]): the split row.
           col (int): the column index
           sep (str): the string to split the column on; if the empty string,
              the column is split into characters instead.

        Returns:
           List[str]: symbols from that cell.
        """
        cell = row[col - 1]  # -1 because we're using one-based indexing.
        return list(cell) if not sep else cell.split(sep)

    def _iter_samples(
        self, filename: str
    ) -> Iterator[Tuple[List[str], Optional[List[str]]]]:
        """Yields specific input samples from a file.

        Args:
            filename (str): input file.

        Yields:
            Tuple[List[str], Optional[List[str]]]: source and target
            string; target is None if self.target_col is 0.
        """
        with open(filename, "r") as source:
            tsv_reader = csv.reader(source, delimiter="\t")
            for row in tsv_reader:
                source = self._get_cell(row, self.source_col, self.source_sep)
                target = (
                    self._get_cell(row, self.target_col, self.target_sep)
                    if self.target_col
                    else None
                )
                yield source, target

    def _make_indices(self, tied_vocabulary: bool) -> None:
        """Generates Dicts for encoding/decoding symbols as unique indices.

        Args:
            tied_vocabulary (bool): whether the source and target should
                share a vocabulary.
        """
        # Ensures the idx of special symbols are identical in both vocabs.
        special_vocabulary = special.SPECIAL.copy()
        target_vocabulary: Set[str] = set()
        source_vocabulary: Set[str] = set()
        for source, target in self.samples:
            source_vocabulary.update(source)
            # Only updates if target.
            if self.target_col:
                target_vocabulary.update(target)
        if tied_vocabulary:
            source_vocabulary.update(target_vocabulary)
            if self.target_col:
                target_vocabulary.update(source_vocabulary)
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
            if self.target_col
            else None
        )
        return source_encoded, target_encoded


class DatasetFeatures(DatasetNoFeatures):
    """Dataset object with separate features.

    This accepts an additional secondary input of feature labels. Features are
    specified in a similar way to source and target.

    The user specifies:

    * an input filename path
    * the 1-based indices for the columns (defaults: source is 1,
      target is 2, features is 3)
    * separator characters used to split the input columns strings, with an
      empty string used to indicate that the string should be split into
      Unicode characters

    These together define an enormous set of possibilities; the defaults
    correspond to the SIGMORPHON 2017 data format.
    """

    features_col: int
    features_sep: str
    features_idx: int

    def __init__(
        self,
        *args,
        features_col,
        features_sep,
        **kwargs,
    ):
        """Initializes the dataset.

        Args:
            features_col (int): 1-indexed column in TSV containing features
                labels.
            features_sep (str): separator character between symbols in target
                string. "" treats each character in target as symbol.
            **kwargs: passed to superclass constructor.
        """
        if features_col < 0:
            raise Error(f"Invalid features column: {features_col}")
        self.features_col = features_col
        self.features_sep = features_sep
        self.features_idx = 0
        super().__init__(*args, **kwargs)

    def _iter_samples(
        self,
        filename: str,
    ) -> Iterator[Tuple[List[str], List[str], Optional[List[str]],]]:
        """Yields specific input samples from a file.

        Sames as in superclass, but also handles features.

        Args:
            filename (str): input file.

        Yields:
            Tuple[List[str], List[str], Optional[List[str]]]: source,
                feature, and target tuple; target is None if self.target_col
                is 0.
        """
        with open(filename, "r") as source:
            tsv_reader = csv.reader(source, delimiter="\t")
            for row in tsv_reader:
                source = self._get_cell(row, self.source_col, self.source_sep)
                features = self._get_cell(
                    row, self.features_col, self.features_sep
                )
                target = (
                    self._get_cell(row, self.target_col, self.target_sep)
                    if self.target_col
                    else None
                )
                yield source, features, target

    def _make_indices(self, tied_vocabulary: bool) -> None:
        """Generates unique indices dictionaries.

        Same as in superclass, but also handles features.

        Args:
            tied_vocabulary (bool): whether the source and target should
                share a vocabulary.
        """
        # Ensures the idx of special symbols are identical in both vocabs.
        special_vocabulary = special.SPECIAL.copy()
        target_vocabulary: Set[str] = set()
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        for source, features, target in self.samples:
            source_vocabulary.update(source)
            features_vocabulary.update(features)
            # Only updates if target.
            if self.target_col:
                target_vocabulary.update(target)
        if tied_vocabulary:
            source_vocabulary.update(target_vocabulary)
            if self.target_col:
                target_vocabulary.update(source_vocabulary)
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
            if self.target_col
            else None
        )
        return source_encoded, features_encoded, target_encoded

    def encode_features(
        self,
        features: List[str],
    ) -> torch.Tensor:
        """Encodes a sequence as a tensor of indices with word boundary IDs.

        This essentially copies behavior of encode but limits return values
        to only features seen from initialization, so unknown feature
        values are not permitted.

        Args:
            features (List[str]): features to be encoded.

        Returns:
            torch.Tensor: the encoded tensor.
        """
        sequence = []
        for feature in features:
            if feature in self.source_symbol2i:
                sequence.append(self.source_symbol2i[feature])
            else:
                raise Error(
                    f"Feature {feature!r} seen during inference was not "
                    "seen in training data; use consistent feature labels "
                    "across datasets"
                )
        return torch.LongTensor(sequence)

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
            (indices >= self.features_idx) | (indices == self.pad_idx),
            indices,
            self.pad_idx,
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


def get_dataset(
    filename: str,
    *,
    tied_vocabulary: bool = True,
    source_col: int = 1,
    target_col: int = 2,
    features_col: int = 3,
    source_sep: str = "",
    target_sep: str = "",
    features_sep: str = ";",
) -> data.Dataset:
    """Dataset factory.

    Args:
        filename (str).
        tied_vocabulary (bool).
        source_col (int).
        target_col (int).
        features_col (int).
        source_sep (str).
        target_sep (str).
        features_sep (str).

    Returns:
        data.Dataset: the dataset.
    """
    dataset_cls = DatasetFeatures if features_col != 0 else DatasetNoFeatures
    return dataset_cls(
        filename,
        tied_vocabulary=tied_vocabulary,
        source_col=source_col,
        target_col=target_col,
        features_col=features_col,
        source_sep=source_sep,
        target_sep=target_sep,
        features_sep=features_sep,
    )
