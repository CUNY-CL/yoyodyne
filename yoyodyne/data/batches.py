"""Batching, padding, and related utilities.

Anything which has a tensor member should inherit from nn.Module, run the
superclass constructor, and register the tensor as a buffer. This enables the
Trainer to move them to the appropriate device."""

from typing import Callable, List, Optional

import torch
from torch import nn

from .. import special


class PaddedTensor(nn.Module):
    """A tensor.

    This is ordinarily used for padding a tensor list, so it represents
    one of (source, target, features) for a batch.

    The optional pad_len argument can be used, e.g., to keep all batches
    the exact same length, which improves performance on certain accelerators.
    If not specified, it will be computed using the length of the longest
    input tensor.

    Mask and string length tensors can be generated as needed.

    Args:
        tensors (List[torch.Tensor]): a list of tensors.
        length_msg_callback (Callable[[int], None]): callback for handling a
            violation of expected tensor length.
        pad_len (int, optional): desired length for padding.
    """

    pad_idx: int
    padded: torch.Tensor

    def __init__(
        self,
        tensors: List[torch.Tensor],
        length_msg_callback: Optional[Callable[[int], None]] = None,
        pad_len: Optional[int] = None,
    ):
        super().__init__()
        if pad_len is None:
            pad_len = max(len(tensor) for tensor in tensors)
        if length_msg_callback is not None:
            length_msg_callback(pad_len)
        self.register_buffer(
            "padded",
            nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=special.PAD_IDX
            ),
        )

    @property
    def mask(self) -> torch.Tensor:
        return self.padded == special.PAD_IDX

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
