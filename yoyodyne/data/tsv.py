"""TSV parsing.

The TsvParser yields data from TSV files using 1-based indexing and custom
separators.
"""

import csv
import dataclasses

from typing import Iterator, List, Tuple, Union

from .. import defaults


class Error(Exception):
    pass


SampleType = Union[
    List[str],
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str], List[str]],
]


@dataclasses.dataclass
class TsvParser:
    """Streams data from a TSV file.

    Args:
        source_col (int, optional): 1-indexed column in TSV containing
            source strings.
        features_col (int, optional): 1-indexed column in TSV containing
            features strings.
        target_col (int, optional): 1-indexed column in TSV containing
            target strings.
        source_sep (str, optional): string used to split source string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
        features_sep (str, optional): string used to split features string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
        target_sep (str, optional): string used to split target string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
        tie_embeddings (bool, optional): Whether or not source and
            target embeddings are tied. If not, then source symbols
            are wrapped in {...}.
    """

    source_col: int = defaults.SOURCE_COL
    features_col: int = defaults.FEATURES_COL
    target_col: int = defaults.TARGET_COL
    source_sep: str = defaults.SOURCE_SEP
    features_sep: str = defaults.FEATURES_SEP
    target_sep: str = defaults.TARGET_SEP
    tie_embeddings: bool = defaults.TIE_EMBEDDINGS

    def __post_init__(self) -> None:
        # This is automatically called after initialization.
        if self.source_col < 1:
            raise Error(f"Out of range source column: {self.source_col}")
        if self.features_col < 0:
            raise Error(f"Out of range features column: {self.features_col}")
        if self.target_col < 0:
            raise Error(f"Out of range target column: {self.target_col}")

    @staticmethod
    def _tsv_reader(path: str) -> Iterator[str]:
        with open(path, "r", encoding=defaults.ENCODING) as tsv:
            yield from csv.reader(tsv, delimiter="\t")

    @staticmethod
    def _get_string(row: List[str], col: int) -> str:
        """Returns a string from a row by index.

        Args:
           row (List[str]): the split row.
           col (int): the column index.
        Returns:
           str: symbol from that string.
        """
        return row[col - 1]  # -1 because we're using one-based indexing.

    @property
    def has_features(self) -> bool:
        return self.features_col != 0

    @property
    def has_target(self) -> bool:
        return self.target_col != 0

    def samples(self, path: str) -> Iterator[SampleType]:
        """Yields source, and features and/or target if available."""
        for row in self._tsv_reader(path):
            source = self.source_symbols(
                self._get_string(row, self.source_col)
            )
            if self.has_features:
                features = self.features_symbols(
                    self._get_string(row, self.features_col)
                )
                if self.has_target:
                    target = self.target_symbols(
                        self._get_string(row, self.target_col)
                    )
                    yield source, features, target
                else:
                    yield source, features
            elif self.has_target:
                target = self.target_symbols(
                    self._get_string(row, self.target_col)
                )
                yield source, target
            else:
                yield source

    # String parsing methods.

    @staticmethod
    def _get_symbols(string: str, sep: str) -> List[str]:
        return list(string) if not sep else string.split(sep)

    def source_symbols(self, string: str) -> List[str]:
        symbols = self._get_symbols(string, self.source_sep)
        # If not tied, then we distinguish the source vocab with {...}.
        if not self.tie_embeddings:
            return [f"{{{symbol}}}" for symbol in symbols]
        return symbols

    def features_symbols(self, string: str) -> List[str]:
        # We deliberately obfuscate these to avoid overlap with source.
        return [
            f"[{symbol}]"
            for symbol in self._get_symbols(string, self.features_sep)
        ]

    def target_symbols(self, string: str) -> List[str]:
        return self._get_symbols(string, self.target_sep)

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
