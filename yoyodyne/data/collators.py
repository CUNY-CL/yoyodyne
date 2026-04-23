"""Collators and related utilities."""

import dataclasses

from . import batches, datasets


class Error(Exception):
    pass


@dataclasses.dataclass
class Collator:
    """Pads data."""

    has_features: bool
    has_target: bool

    def pad_source(
        self, itemlist: list[datasets.Item]
    ) -> batches.PaddedTensor:
        """Pads source.

        Args:
            itemlist (list[datasets.Item]).

        Returns:
            batches.PaddedTensor.
        """
        return batches.PaddedTensor(
            [item.source for item in itemlist],
        )

    def pad_features(
        self,
        itemlist: list[datasets.Item],
    ) -> batches.PaddedTensor:
        """Pads features.

        Args:
            itemlist (list[datasets.Item]).

        Returns:
            batches.PaddedTensor.
        """
        return batches.PaddedTensor(
            [item.features for item in itemlist],
        )

    def pad_target(
        self, itemlist: list[datasets.Item]
    ) -> batches.PaddedTensor:
        """Pads target.

        Args:
            itemlist (list[datasets.Item]).

        Returns:
            batches.PaddedTensor.
        """
        return batches.PaddedTensor(
            [item.target for item in itemlist],
        )

    def __call__(self, itemlist: list[datasets.Item]) -> batches.Batch:
        """Pads all elements of an itemlist.

        Args:
            itemlist (list[datasets.Item]).

        Returns:
            batches.Batch.
        """
        return batches.Batch(
            self.pad_source(itemlist),
            self.pad_features(itemlist) if self.has_features else None,
            self.pad_target(itemlist) if self.has_target else None,
        )
