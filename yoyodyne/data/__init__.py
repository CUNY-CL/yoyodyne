import argparse

from .. import defaults

from .batches import PaddedBatch, PaddedTensor  # noqa: F401
from .collators import Collator  # noqa: F401
from .datasets import Item  # noqa: F401
from .datasets import BaseDataset  # noqa: F401
from .datasets import DatasetNoFeatures  # noqa: F401
from .datasets import DatasetFeatures  # noqa: F401
from .datasets import get_dataset  # noqa: F401


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds collator options to the argument parser.

    Args:
        parser (argparse.ArgumentParser).
    """
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
