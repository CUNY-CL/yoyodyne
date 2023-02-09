"""Collators and related utilities."""

from typing import List

import torch

from . import batches, dataconfig, datasets, util


class LengthError(Exception):
    pass


class Collator:
    """Base class for other collators.

    Pads according to the longest sequence in a batch of sequences."""

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
        max_source_length: int,
        max_target_length: int,
    ):
        """Initializes the collator.

        Args:
            pad_idx (int).
            config (dataconfig.DataConfig).
            arch (str).
            max_source_length (int).
            max_target_length (int).
        """
        self.pad_idx = pad_idx
        self.has_features = config.has_features
        self.has_target = config.has_target
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.separate_features = config.has_features and arch in [
            "pointer_generator_lstm",
            "transducer",
        ]

    def _check_source_length(self, itemlist: List[datasets.Item]):
        """Checks if a source length in the batch is greater than the
        maximum allowed source length.

        Args:
            itemlist (List[datasets.Item]).

        Raises:
            LengthError.
        """
        batch_max_source_length = max([len(item.source) for item in itemlist])
        if self.max_source_length < batch_max_source_length:
            msg = f"The length of a source sample ({batch_max_source_length}) "
            msg += "is greater than the allowed `--max_source_length` "
            msg += f"({self.max_source_length})"
            raise LengthError(msg)

    def _check_target_length(self, itemlist: List[datasets.Item]):
        """Checks if a target length in the batch is greater than the
        maximum allowed target length for inference. Logs a warning suggesting
        that this probably indicates the max_target_length is too short.

        Args:
            itemlist (List[datasets.Item]).
        """
        batch_max_target_length = max([len(item.target) for item in itemlist])
        if self.max_target_length < batch_max_target_length:
            msg = f"The length of a target sample ({batch_max_target_length}) "
            msg += "is greater than the `--max_target_length` specified "
            msg += f"({self.max_target_length}). This means that "
            msg += "decoding at inference time will likely be truncated. "
            msg += "Consider increasing `--max_target_length` "
            util.log_warning(msg)

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

        Raises:
            LengthError.
        """
        self._check_source_length(itemlist)
        return batches.PaddedTensor(
            [item.source for item in itemlist], self.pad_idx
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

        Raises:
            LengthError.
        """
        self._check_source_length(itemlist)
        return batches.PaddedTensor(
            self.concatenate_source_and_features(itemlist), self.pad_idx
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
            [item.features for item in itemlist], self.pad_idx
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
        self._check_target_length(itemlist)
        return batches.PaddedTensor(
            [item.target for item in itemlist], self.pad_idx
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
