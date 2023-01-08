"""Batching, padding, and related utilities."""

import dataclasses
from typing import List, Optional

import torch
from torch.nn import functional

from . import datasets


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


@dataclasses.dataclass
class UnpaddedBatch:
    """Source tensors, with optional features and target tensors."""

    source: List[torch.Tensor]
    features: Optional[List[torch.Tensor]] = None
    target: Optional[List[torch.Tensor]] = None

    @property
    def has_features(self):
        return self.features is not None

    @property
    def has_target(self):
        return self.targets is not None


@dataclasses.dataclass
class PaddedTensor:
    """A tensor and its mask.

    This is ordinarily used for padding a tensor list, so it represents
    one of (source, target, features) for a batch.
    """

    padded: torch.Tensor
    mask: torch.Tensor

    @classmethod
    def from_tensorlist(
        cls, tensorlist: List[torch.Tensor], pad_idx: int
    ) -> "PaddedTensor":
        """Constructs a padded tensor from a tensor list.

        Args:
            tensorlist (List[torch.Tensor]): a list of tensors.
            pad_idx (int): padding index.
        """
        max_len = max(len(tensor) for tensor in tensorlist)
        padded = torch.stack(
            [cls.pad_tensor(tensor, pad_idx, max_len) for tensor in tensorlist]
        )
        mask = padded == pad_idx
        return cls(padded, mask)

    @staticmethod
    def pad_tensor(
        tensor: torch.Tensor, pad_idx: int, pad_max: int
    ) -> torch.Tensor:
        """Pads a tensor.

        Args:
            tensor (torch.Tensor).
            pad_idx (int): padding index.
            pad_max (int): desired tensor length.

        Returns:
            torch.Tensor.
        """
        padding = pad_max - len(tensor)
        return functional.pad(tensor, (0, padding), "constant", pad_idx)


@dataclasses.dataclass
class PaddedBatch:
    """Padded source tensor, with optional padded features and target tensors.

    This represents a padded batch. It is produced by the collator and fed to
    the trainer."""

    source: PaddedTensor
    features: Optional[PaddedTensor] = None
    target: Optional[PaddedTensor] = None

    @property
    def has_features(self):
        return self.features is not None

    @property
    def has_target(self):
        return self.target is not None
