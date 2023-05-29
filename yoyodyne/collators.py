"""Collators and related utilities."""

import argparse
from typing import List

import torch

from . import batches, dataconfig, datasets, defaults, util


class LengthError(Exception):
    pass


class Collator:
    """Pads data."""

    pad_idx: int
    has_features: bool
    has_target: bool
    separate_features: bool
    max_source_length: int
    max_target_length: int

    def __init__(
        self,
        pad_idx,
        config: dataconfig.DataConfig,
        arch: str,
        max_source_length=defaults.MAX_SOURCE_LENGTH,
        max_target_length=defaults.MAX_TARGET_LENGTH,
        # FIXME: Handle length.
        # max_features_length=defaults.MAX_FEATURES_LENGTH,
    ):
        """Initializes the collator.

        If max source and/or target length are not specified, they are
        computed dynamically (i.e., for each source and target batch).

        Args:
            pad_idx (int).
            config (dataconfig.DataConfig).
            arch (str).
            max_source_length (int, optional).
            max_target_length (int, optional).
        """
        self.pad_idx = pad_idx
        self.has_features = config.has_features
        self.has_target = config.has_target
        self.separate_features = config.has_features and arch in [
            "pointer_generator_lstm",
            "transducer",
        ]
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # FIXME: Handle length.
        # self.max_features_length = max_features_length

    def _source_length_error(self, padded_length: int):
        """Callback function to raise the error when the padded length of the
        source batch is greater than the `max_source_length` allowed.

        Args:
            padded_length (int): The length of the the padded tensor.

        Raises:
            LengthError.
        """
        if padded_length > self.max_source_length:
            msg = f"The length of a source sample ({padded_length}) "
            msg += "is greater than the allowed `--max_source_length` "
            msg += f"({self.max_source_length})"
            raise LengthError(msg)

    def _target_length_warning(self, padded_length: int):
        """Callback function to log a message when the padded length of the
        target batch is greater than the `max_target_length` allowed.

        Since `max_target_length` truncates during inference, this is only a
        suggestion; it is logged rather than converted into an error.

        Args:
            padded_length (int): The length of the the padded tensor.
        """
        if self.max_target_length and padded_length > self.max_target_length:
            msg = f"The length of a batch ({padded_length}) "
            msg += "is greater than the `--max_target_length` specified "
            msg += f"({self.max_target_length}). This means that "
            msg += "truncation may occur at inference time. "
            msg += "Consider increasing `--max_target_length`."
            util.log_info(msg)

    @staticmethod
    def concatenate_source_and_features(
        itemlist: List[datasets.Item],
    ) -> List[torch.Tensor]:
        """Concatenates source and feature tensors."""
        return [
            (
                torch.cat((item.source, item.features))
                if item.has_features
                else item.source
            )
            for item in itemlist
        ]

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
            self.pad_idx,
            self.max_source_length,
            self._source_length_error,
        )

    def pad_source_features(
        self,
        itemlist: List[datasets.Item],
    ) -> batches.PaddedTensor:
        """Pads concatenated source and features.

        Args:
            itemlist (List[datasets.Item]).

        Returns:
            batches.PaddedTensor.
        """
        return batches.PaddedTensor(
            self.concatenate_source_and_features(itemlist),
            self.pad_idx,
            # FIXME: Handle length.
            self.max_source_length,
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
        return batches.PaddedTensor(
            [item.features for item in itemlist],
            self.pad_idx,
            # FIXME: Handle length.
        )

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
            self.pad_idx,
            self.max_target_length,
            self._target_length_warning,
        )

    def __call__(self, itemlist: List[datasets.Item]) -> batches.PaddedBatch:
        """Pads all elements of an itemlist.

        Args:
            itemlist (List[datasets.Item]).

        Returns:
            batches.PaddedBatch.
        """
        padded_target = self.pad_target(itemlist) if self.has_target else None
        if self.separate_features:
            return batches.PaddedBatch(
                self.pad_source(itemlist),
                features=self.pad_features(itemlist),
                target=padded_target,
            )
        else:
            return batches.PaddedBatch(
                self.pad_source_features(itemlist),
                target=padded_target,
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
            "--max_target_length",
            type=int,
            default=defaults.MAX_TARGET_LENGTH,
            help="Maximum target string length. Default: %(default)s.",
        )
        # FIXME: Handle length.
        # parser.add_argument(
        #    "--max_features_length",
        #    type=int,
        #    default=defaults.MAX_FEATURES_LENGTH,
        #    help="Maximum features string length. Default: %(default)s.",
        # )
