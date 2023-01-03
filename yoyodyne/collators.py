"""Collators."""

from typing import Iterable, List, Tuple

import torch
from torch.nn import functional


class Collator:
    """Base class for other collators.

    Pads according to the longest sequence in a batch of sequences."""

    pad_idx: int

    def __init__(self, pad_idx):
        """Initializes the collator.

        Args:
            pad_idx (int).
        """
        self.pad_idx = pad_idx

    @staticmethod
    def max_len(batch: torch.Tensor) -> int:
        """Computes max length for a list of tensors.

        Args:
            batch (List[, torch.Tensor, torch.Tensor]).

        Returns:
            int.
        """
        return max(len(tensor) for tensor in batch)

    @staticmethod
    def concat_tuple(
        b1: Iterable[torch.Tensor], b2: Iterable[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """Concatenates all tensors in b1 to respective tensor in b2.

        For joining source and feature tensors in batches.

        Args:
            b1 (Iterable[torch.Tensor]): Iterable of tensors.
            b2 (Iterable[torch.Tensor]): Iterable of tensors.

        Returns:
            Tuple[torch.Tensor]: the concatenation of
            parallel entries in b1 and b2.
        """
        return tuple(torch.cat((i, j)) for i, j in zip(b1, b2))

    def pad_collate(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads the batch to the maximum sequence length in the batch.

        Args:
            batch torch.Tensor: A batch of samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        batch = torch.stack(
            [self.pad_tensor(tensor, self.max_len(batch)) for tensor in batch]
        )
        batch_mask = batch == self.pad_idx
        return batch, batch_mask

    def pad_tensor(self, tensor: torch.Tensor, pad_max: int) -> torch.Tensor:
        """Pads a tensor.

        Args:
            tensor (torch.Tensor).
            pad_max (int): The desired length for the tensor.

        Returns:
            torch.Tensor.
        """
        padding = pad_max - len(tensor)
        return functional.pad(tensor, (0, padding), "constant", self.pad_idx)

    @property
    def has_features(self) -> bool:
        return False

    @staticmethod
    def is_feature_batch(batch: List[torch.Tensor]) -> bool:
        return False


class SourceCollator(Collator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads source.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        # Checks if the dataloader passed features.
        if self.is_feature_batch(batch):
            source, features, _ = zip(*batch)
            # Concatenates features with source.
            source = self.concat_tuple(source, features)
        else:
            source, _ = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        return source_padded, source_mask

    @staticmethod
    def is_feature_batch(batch: List[torch.Tensor]) -> bool:
        return len(batch[0]) == 3


class SourceTargetCollator(SourceCollator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads source and target.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor].
        """
        if self.is_feature_batch(batch):
            source, features, target = zip(*batch)
            # Concatenates features with source.
            source = self.concat_tuple(source, features)
        else:
            source, target = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        target_padded, target_mask = self.pad_collate(target)
        return source_padded, source_mask, target_padded, target_mask


class SourceFeaturesCollator(Collator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads source and features.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor].
        """
        source, features, _ = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        features_padded, features_mask = self.pad_collate(features)
        return source_padded, source_mask, features_padded, features_mask

    @property
    def has_features(self) -> bool:
        return True

    @staticmethod
    def is_feature_batch(batch: List[torch.Tensor]) -> bool:
        return True


class SourceFeaturesTargetCollator(SourceFeaturesCollator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Pads source, features, and target.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
            ]
        """
        source, features, target = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        target_padded, target_mask = self.pad_collate(target)
        features_padded, features_mask = self.pad_collate(features)
        return (
            source_padded,
            source_mask,
            features_padded,
            features_mask,
            target_padded,
            target_mask,
        )

    @property
    def has_features(self) -> bool:
        return True


def get_collator(
    pad_idx: int, *, arch: str, include_features: bool, include_target: bool
) -> Collator:
    """Collator factory.

    Args:
        pad_idx (int).
        arch (str).
        include_features (bool).
        include_target (bool).

    Returns:
        Collator.
    """
    if include_features and arch in ["pointer_generator_lstm", "transducer"]:
        collator_cls = (
            SourceFeaturesTargetCollator
            if include_target
            else SourceFeaturesCollator
        )
    else:
        collator_cls = (
            SourceTargetCollator if include_target else SourceCollator
        )
    return collator_cls(pad_idx)
