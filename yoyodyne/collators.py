"""Collators and related utilities."""

from typing import List

import torch

from . import batches, dataconfig, datasets


class Collator:
    """Base class for other collators.

    Pads according to the longest sequence in a batch of sequences."""

    pad_idx: int
    has_features: bool
    has_target: bool
    separate_features: bool

    def __init__(self, pad_idx, config: dataconfig.DataConfig, arch: str):
        """Initializes the collator.

        Args:
            pad_idx (int).
            config (dataconfig.DataConfig).
            arch (str).
        """
        self.pad_idx = pad_idx
        self.has_features = config.has_features
        self.has_target = config.has_target
        self.separate_features = config.has_features and arch in [
            "pointer_generator_lstm",
            "transducer",
        ]

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
        """
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
