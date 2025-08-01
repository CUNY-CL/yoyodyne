"""Collators and related utilities."""

import argparse
import dataclasses

from typing import List

from .. import defaults, util
from . import batches, datasets


class Error(Exception):
    pass


@dataclasses.dataclass
class Collator:
    """Pads data."""

    has_features: bool
    has_target: bool
    max_source_length: int = defaults.MAX_SOURCE_LENGTH
    max_features_length: int = defaults.MAX_FEATURES_LENGTH
    max_target_length: int = defaults.MAX_TARGET_LENGTH

    def _source_length_error(self, padded_length: int) -> None:
        """Callback function for excessive source length.

        Args:
            padded_length (int): The length of the the padded tensor.

        Raises:
            Error.
        """
        if padded_length > self.max_source_length:
            raise Error(
                f"The length of a source sample ({padded_length}) is greater "
                f"than the `--max_source_length` specified "
                f"({self.max_source_length})"
            )

    def _features_length_error(self, padded_length: int) -> None:
        """Callback function for excessive features length.

        Args:
            padded_length (int): The length of the the padded tensor.

        Raises:
            Error.
        """
        if padded_length > self.max_features_length:
            raise Error(
                f"The length of a features sample ({padded_length}) is "
                f"greater than the `--max_features_length` specified "
                f"({self.max_features_length})"
            )

    def _target_length_warning(self, padded_length: int) -> None:
        """Callback function for excessive target length.

        Since `max_target_length` just truncates during inference, this is
        simply a suggestion.

        Args:
            padded_length (int): The length of the the padded tensor.
        """
        if padded_length > self.max_target_length:
            util.log_info(
                f"The length of a batch ({padded_length}) is greater than the "
                f"`--max_target_length` specified ({self.max_target_length}); "
                f"decoding at inference time will likely be truncated. "
                f"Consider increasing `--max_target_length`."
            )

    def pad_source(
        self, itemlist: List[datasets.Item]
    ) -> batches.PaddedTensor:
        """Pads source.

        Args:
            itemlist (List[datasets.Item]).

        Returns:
            batches.PaddedTensor.
        """
        return batches.PaddedTensor(
            [item.source for item in itemlist],
            self._source_length_error,
        )

    def pad_features(
        self,
        itemlist: List[datasets.Item],
    ) -> batches.PaddedTensor:
        """Pads features.

        Args:
            itemlist (List[datasets.Item]).

        Returns:
            batches.PaddedTensor.
        """
        return batches.PaddedTensor([item.features for item in itemlist])

    def pad_target(
        self, itemlist: List[datasets.Item]
    ) -> batches.PaddedTensor:
        """Pads target.

        Args:
            itemlist (List[datasets.Item]).

        Returns:
            batches.PaddedTensor.
        """
        return batches.PaddedTensor(
            [item.target for item in itemlist],
            self._target_length_warning,
        )

    def __call__(self, itemlist: List[datasets.Item]) -> batches.PaddedBatch:
        """Pads all elements of an itemlist.

        Args:
            itemlist (List[datasets.Item]).

        Returns:
            batches.PaddedBatch.
        """
        return batches.PaddedBatch(
            self.pad_source(itemlist),
            self.pad_features(itemlist) if self.has_features else None,
            self.pad_target(itemlist) if self.has_target else None,
        )

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
            "--max_features_length",
            type=int,
            default=defaults.MAX_FEATURES_LENGTH,
            help="Maximum features string length. Default: %(default)s.",
        )
        parser.add_argument(
            "--max_target_length",
            type=int,
            default=defaults.MAX_TARGET_LENGTH,
            help="Maximum target string length. Default: %(default)s.",
        )
