"""TSV parsing.

The TsvParser yield string tuples from TSV files using 1-based indexing.

The CellParser converts between raw strings ("cells") and lists of string
symbols.
"""

import csv
import dataclasses
from typing import Iterator, List, Tuple

from .. import defaults, util


class Error(Exception):
    """Module-specific exception."""

    pass


@dataclasses.dataclass
class TsvParser:
    """Streams cells from a TSV file.

    Args:
        source_col (int, optional): 1-indexed column in TSV containing
            source strings.
        features_col (int, optional): 1-indexed column in TSV containing
            features strings.
        target_col (int, optional): 1-indexed column in TSV containing
            target strings.
    """

    source_col: int = defaults.SOURCE_COL
    features_col: int = defaults.FEATURES_COL
    target_col: int = defaults.TARGET_COL

    # This is automatically called after initialization.
    def __post_init__(self) -> None:
        if self.features_col < 0:
            raise Error(f"Invalid features column: {self.features_col}")
        if self.features_col != 0:
            util.log_info("Including features")
        if self.source_col < 1:
            raise Error(f"Invalid source column: {self.source_col}")
        if self.target_col < 0:
            raise Error(f"Invalid target column: {self.target_col}")
        if self.target_col == 0:
            util.log_info("Ignoring targets in input")

    @staticmethod
    def _tsv_reader(path: str) -> Iterator[str]:
        with open(path, "r") as tsv:
            yield from csv.reader(tsv, delimiter="\t")

    @staticmethod
    def _get_cell(row: List[str], col: int) -> str:
        """Returns a cell from a row by index.

        Args:
           row (List[str]): the split row.
           col (int): the column index.

        Returns:
           str: symbol from that cell.
        """
        return row[col - 1]  # -1 because we're using one-based indexing.

    @property
    def has_source(self) -> bool:
        return True

    @property
    def has_features(self) -> bool:
        return self.features_col != 0

    @property
    def has_target(self) -> bool:
        return self.target_col != 0

    def source_samples(self, path: str) -> Iterator[str]:
        """Yields source."""
        for row in self._tsv_reader(path):
            yield self._get_cell(row, self.source_col)

    def source_target_samples(self, path: str) -> Iterator[Tuple[str, str]]:
        """Yields source and target."""
        for row in self._tsv_reader(path):
            source = self._get_cell(row, self.source_col)
            target = self._get_cell(row, self.target_col)
            yield source, target

    def source_features_target_samples(
        self, path: str
    ) -> Iterator[Tuple[str, str, str]]:
        """Yields source, features, and target."""
        for row in self._tsv_reader(path):
            source = self._get_cell(row, self.source_col)
            features = self._get_cell(row, self.features_col)
            target = self._get_cell(row, self.target_col)
            yield source, features, target

    def source_features_samples(self, path: str) -> Iterator[Tuple[str, str]]:
        """Yields source, and features."""
        for row in self._tsv_reader(path):
            source = self._get_cell(row, self.source_col)
            features = self._get_cell(row, self.features_col)
            yield source, features

    def samples(self, path: str) -> Iterator[Tuple[str, ...]]:
        """Picks the right one."""
        if self.has_features:
            if self.has_target:
                self.source_features_target_samples(path)
            else:
                return self.source_features_samples(path)
        elif self.has_target:
            return self.source_target_samples(path)
        else:
            return self.source_samples(path)


@dataclasses.dataclass
class CellParser:
    """Converts between cells and lists of string symbols.

    Args:
        source_sep (str, optional): string used to split source string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
        features_sep (str, optional): string used to split features string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
        target_sep (str, optional): string used to split target string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
    """

    source_sep: str = defaults.SOURCE_SEP
    features_sep: str = defaults.FEATURES_SEP
    target_sep: str = defaults.TARGET_SEP

    # Parsing methods.

    @staticmethod
    def _get_symbols(cell: str, sep: str) -> List[str]:
        return list(cell) if not sep else sep.split(cell)

    def source_symbols(self, cell: str) -> List[str]:
        return self._get_symbols(cell, self.features_sep)

    def features_symbols(self, cell: str) -> List[str]:
        # We deliberately obfuscate these to avoid overlap with source.
        return [
            f"[{symbol}]"
            for symbol in self._get_symbols(cell, self.features_sep)
        ]

    def target_symbols(self, cell: str) -> List[str]:
        return self._get_symbols(cell, self.target_sep)

    # Deserialization methods.

    def source_string(self, symbols: List[str]) -> str:
        return self.source_sep.join(symbols)

    def features_string(self, symbols: List[str]) -> str:
        return self.features_sep.join(
            # This indexing strips off the obfuscation.
            [symbol[1:-1] for symbol in symbols],
        )

    def target_string(self, symbols: List[str]) -> str:
        return self.target_sep.join(symbols)
