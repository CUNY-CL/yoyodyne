"""Batching, padding, and related utilities.

Anything which has-a tensor should inherit from nn.Module and run the
superclass constructor, as that enables the Trainer to move them to the
appropriate device. Furthermore, we register the tensors as buffers.
"""

from typing import List, Optional

import torch
from torch import nn


class PaddedTensor(nn.Module):
    """A tensor and its mask.

    This is ordinarily used for padding a tensor list, so it represents
    one of (source, target, features) for a batch.
    """

    padded: torch.Tensor
    mask: torch.Tensor

    def __init__(
        self,
        tensorlist: List[torch.Tensor],
        pad_idx: int,
        pad_len: Optional[int] = None,
    ):
        """Constructs the padded tensor from a list of tensors.

        The optional pad_len argument can be used, e.g., to keep all batches
        the exact same length, which improves performance on certain
        accelerators. If not specified, it will be computed using the length
        of the longest input tensor.

        Args:
            tensorlist (List[torch.Tensor]): a list of tensors.
            pad_idx (int): padding index.
            pad_len (int, optional): desired length for padding.

        """
        super().__init__()
        if pad_len is None:
            pad_len = max(len(tensor) for tensor in tensorlist)
        self.register_buffer(
            "padded",
            torch.stack(
                [
                    self.pad_tensor(tensor, pad_idx, pad_len)
                    for tensor in tensorlist
                ],
            ),
        )
        self.register_buffer("mask", self.padded == pad_idx)

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
        return nn.functional.pad(tensor, (0, padding), "constant", pad_idx)

    def __len__(self) -> int:
        return len(self.padded)


class PaddedBatch(nn.Module):
    """Padded source tensor, with optional padded features and target tensors.

    This represents a padded batch. It is produced by the collator and fed to
    the trainer."""

    source: PaddedTensor
    features: Optional[PaddedTensor]
    target: Optional[PaddedTensor]

    def __init__(self, source, features=None, target=None):
        super().__init__()
        self.register_module("source", source)
        self.register_module("target", target)
        self.register_module("features", features)

    @property
    def has_features(self):
        return self.features is not None

    @property
    def has_target(self):
        return self.target is not None

    def __len__(self) -> int:
        return len(self.source)
