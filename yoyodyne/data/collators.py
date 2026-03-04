"""Collators and related utilities."""

import dataclasses
from typing import List

from . import batches, datasets


class Error(Exception):
    pass


@dataclasses.dataclass
class Collator:
    """Pads data."""

    has_features: bool
    has_target: bool

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
        )

    def __call__(self, itemlist: List[datasets.Item]) -> batches.Batch:
        """Pads all elements of an itemlist.

        Args:
            itemlist (List[datasets.Item]).

        Returns:
            batches.Batch.
        """
        return batches.Batch(
            self.pad_source(itemlist),
            self.pad_features(itemlist) if self.has_features else None,
            self.pad_target(itemlist) if self.has_target else None,
        )
