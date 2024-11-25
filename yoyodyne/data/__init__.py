"""Data classes."""

import argparse

from .. import defaults
from .batches import PaddedBatch, PaddedTensor  # noqa: F401
from .datamodules import DataModule  # noqa: F401
from .datasets import Dataset  # noqa: F401
from .indexes import Index  # noqa: F401
from .mappers import Mapper  # noqa: F401
from .tsv import TsvParser  # noqa: F401


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds data options to the argument parser.

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
        "--batch_size",
        type=int,
        default=defaults.BATCH_SIZE,
        help="Batch size. Default: %(default)s.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=defaults.MAX_SOURCE_LENGTH,
        help="Maximum source string length. Default: %(default)s.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=defaults.MAX_TARGET_LENGTH,
        help="Maximum target string length. Default: %(default)s.",
    )
