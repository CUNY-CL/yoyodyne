"""Batching, padding, and related utilities.

Anything which has a tensor member should inherit from nn.Module, run the
superclass constructor, and register the tensor as a buffer. This enables the
Trainer to move them to the appropriate device."""

from typing import Callable, List, Optional

import torch
from torch import nn


class PaddedTensor(nn.Module):
    """A tensor and its mask.

    This is ordinarily used for padding a tensor list, so it represents
    one of (source, target, features) for a batch."""

    padded: torch.Tensor
    mask: torch.Tensor

    def __init__(
        self,
        tensorlist: List[torch.Tensor],
        pad_idx: int,
        length: int,
        length_msg_callback: Optional[Callable[[int], None]] = None,
    ):
        """Constructs the padded tensor from a list of tensors.

        The optional length argument can be used, e.g., to keep all batches
        the exact same length, which improves performance on certain
        accelerators. If not specified, it will be computed using the length
        of the longest input tensor.

        Args:
            tensorlist (List[torch.Tensor]): a list of tensors.
            pad_idx (int): padding index.
            length (int): desired padded length.
            length_msg_callback (Callable[[int], None]): callback which flags
                invalid tensor lengths.

        """
        super().__init__()
        if length_msg_callback is not None:
            batch_length = max(len(tensor) for tensor in tensorlist)
            length_msg_callback(batch_length)
        self.register_buffer(
            "padded",
            torch.stack(
                [
                    self.pad_tensor(tensor, pad_idx, length)
                    for tensor in tensorlist
                ],
            ),
        )
        self.register_buffer("mask", self.padded == pad_idx)

    @staticmethod
    def pad_tensor(
        tensor: torch.Tensor, pad_idx: int, length: int
    ) -> torch.Tensor:
        """Pads a tensor.

        Args:
            tensor (torch.Tensor).
            pad_idx (int): padding index.
            length (int): desired tensor length.

        Returns:
            torch.Tensor.
        """
        padding = length - len(tensor)
        return nn.functional.pad(tensor, (0, padding), "constant", pad_idx)

    def __len__(self) -> int:
        return len(self.padded)

    def lengths(self) -> torch.Tensor:
        """Computes the lengths of all the strings in the tensor.

        By convention we seem to want this tensor on CPU.

        Returns:
            torch.Tensor.
        """
        return (self.mask == 0).sum(dim=1).cpu()


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
