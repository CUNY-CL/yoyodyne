"""Dataset config class."""

import csv
import dataclasses
from typing import Iterator, List, Optional, Tuple

from . import util


class Error(Exception):
    """Module-specific exception."""

    pass


@dataclasses.dataclass
class DataConfig:
    """Configuration specifications for a dataset.

    Args:
        source_col (int, optional): 1-indexed column in TSV containing
            source strings.
        target_col (int, optional): 1-indexed column in TSV containing
            target strings.
        features_col (int, optional): 1-indexed column in TSV containing
            features strings.
        source_sep (str, optional): separator character between symbols in
            source string. "" treats each character in source as a symbol.
        target_sep (str, optional): separator character between symbols in
            target string. "" treats each character in target as a symbol.
        features_sep (str, optional): separator character between symbols in
            features string. "" treats each character in features as a symbol.
        tied_vocabulary (bool, optional): whether the source and target
            should share a vocabulary.
    """

    source_col: int = 1
    target_col: int = 2
    features_col: int = 0
    source_sep: str = ""
    target_sep: str = ""
    features_sep: str = ";"
    tied_vocabulary: bool = True

    def __post_init__(self) -> None:
        # This is automatically called after initialization.
        if self.source_col < 1:
            raise Error(f"Invalid source column: {self.source_col}")
        if self.target_col < 0:
            raise Error(f"Invalid target column: {self.target_col}")
        if self.target_col == 0:
            util.log_info("Ignoring targets in input")
        if self.features_col < 0:
            raise Error(f"Invalid features column: {self.features_col}")
        if self.features_col != 0:
            util.log_info("Including features")

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

    # Source is always present.

    @property
    def has_target(self) -> bool:
        return self.target_col != 0

    @property
    def has_features(self) -> bool:
        return self.features_col != 0

    # Iterators over the data; just provide a filename path.

    def source_samples(self, filename: str) -> Iterator[List[str]]:
        """Yields source."""
        with open(filename, "r") as source:
            tsv_reader = csv.reader(source, delimiter="\t")
            for row in tsv_reader:
                yield self._get_cell(row, self.source_col, self.source_sep)

    def source_target_samples(
        self, filename: str
    ) -> Iterator[Tuple[List[str], List[str]]]:
        """Yields source and target."""
        with open(filename, "r") as source:
            tsv_reader = csv.reader(source, delimiter="\t")
            for row in tsv_reader:
                source = self._get_cell(row, self.source_col, self.source_sep)
                target = self._get_cell(row, self.target_col, self.target_sep)
                yield source, target

    def source_features_target_samples(
        self, filename: str
    ) -> Iterator[Tuple[List[str], List[str], List[str]]]:
        """Yields source, features, and target."""
        with open(filename, "r") as source:
            tsv_reader = csv.reader(source, delimiter="\t")
            for row in tsv_reader:
                source = self._get_cell(row, self.source_col, self.source_sep)
                features = self._get_cell(
                    row, self.features_col, self.features_sep
                )
                target = self._get_cell(row, self.target_col, self.target_sep)
                yield source, features, target

    def samples(self, filename: str) -> Iterator[Tuple[List[str], ...]]:
        """Picks the right one for this config."""
        if self.has_features:
            return (
                self.source_features_target_samples(filename)
                if self.has_target
                else self.source_features_samples(filename)
            )
        else:
            return (
                self.source_target_samples(filename)
                if self.has_target
                else self.source_samples(filename)
            )

    # Writer help.

    def row(
        self,
        source: List[str],
        target: List[str],
        features: Optional[List[str]] = None,
    ) -> List[str]:
        """Returns a TSV-style row using the config.

        Args:
            source (List[str]).
            target (List[str]).
            features (List[str], optional).

        Returns:
            List[str].
        """
        row = [""] * max(self.source_col, self.target_col, self.features_col)
        # -1 because we're using base-1 indexing.
        row[self.source_col - 1] = self.source_sep.join(source)
        row[self.target_col - 1] = self.target_sep.join(target)
        if self.has_features:
            assert features is not None, "Expected features"
            row[self.features_col - 1] = self.features_sep.join(features)
        return row
