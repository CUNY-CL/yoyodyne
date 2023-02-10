"""Dataset config class."""

import argparse
import csv
import dataclasses
import inspect
from typing import Iterator, List, Tuple

from . import defaults, util


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
        source_sep (str, optional): separator character between symbol in
            source string. "" treats each character in source as a symbol.
        target_sep (str, optional): separator character between symbol in
            target string. "" treats each character in target as a symbol.
        features_sep (str, optional): separator character between symbol in
            features string. "" treats each character in features as a symbol.
        tied_vocabulary (bool, optional): whether the source and target
            should share a vocabulary.
    """

    source_col: int = defaults.SOURCE_COL
    target_col: int = defaults.TARGET_COL
    features_col: int = defaults.FEATURES_COL
    source_sep: str = defaults.SOURCE_SEP
    target_sep: str = defaults.TARGET_SEP
    features_sep: str = defaults.FEATURES_SEP
    tied_vocabulary: bool = defaults.TIED_VOCABULARY

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

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """Creates an instance from CLI arguments."""
        params = vars(args)
        valid_kwargs = inspect.signature(cls.__init__).parameters
        dataconfig_kwargs = {
            name: params[name] for name in valid_kwargs if name in params
        }
        dataconfig_kwargs.update(**kwargs)
        return cls(**dataconfig_kwargs)

    @staticmethod
    def _get_cell(row: List[str], col: int, sep: str) -> List[str]:
        """Returns the split cell of a row.

        Args:
           row (List[str]): the split row.
           col (int): the column index
           sep (str): the string to split the column on; if the empty string,
              the column is split into characters instead.

        Returns:
           List[str]: symbol from that cell.
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
                # Avoids overlap with source.
                features = [
                    f"[{feature}]"
                    for feature in self._get_cell(
                        row, self.features_col, self.features_sep
                    )
                ]
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

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds data configuration options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--source_col",
            type=int,
            default=defaults.SOURCE_COL,
            help="1-based index for source column. Default: %(default)s.",
        )
        parser.add_argument(
            "--target_col",
            type=int,
            default=defaults.TARGET_COL,
            help="1-based index for target column. Default: %(default)s.",
        )
        parser.add_argument(
            "--features_col",
            type=int,
            default=defaults.FEATURES_COL,
            help="1-based index for features column; "
            "0 indicates the model will not use features. "
            "Default: %(default)s.",
        )
        parser.add_argument(
            "--source_sep",
            type=str,
            default=defaults.SOURCE_SEP,
            help="String used to split source string into symbols; "
            "an empty string indicates that each Unicode codepoint "
            "is its own symbol. Default: %(default)r.",
        )
        parser.add_argument(
            "--target_sep",
            type=str,
            default=defaults.TARGET_SEP,
            help="String used to split target string into symbols; "
            "an empty string indicates that each Unicode codepoint "
            "is its own symbol. Default: %(default)r.",
        )
        parser.add_argument(
            "--features_sep",
            type=str,
            default=defaults.FEATURES_SEP,
            help="String used to split features string into symbols; "
            "an empty string indicates that each Unicode codepoint "
            "is its own symbol. Default: %(default)r.",
        )
        parser.add_argument(
            "--tied_vocabulary",
            action="store_true",
            default=defaults.TIED_VOCABULARY,
            help="Share source and target embeddings. Default: %(default)s.",
        )
        parser.add_argument(
            "--no_tied_vocabulary",
            action="store_false",
            dest="tied_vocabulary",
            default=True,
        )
