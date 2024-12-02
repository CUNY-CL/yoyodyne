"""Datasets and related utilities."""

import dataclasses

from typing import List, Optional

import torch
from torch import nn
from torch.utils import data

from . import mappers, tsv


class Item(nn.Module):
    """Source tensor, with optional features and target tensors.

    This represents a single item or observation.

    Args:
        source (torch.Tensor).
        features (torch.Tensor, optional).
        target (torch.Tensor, optional).
    """

    source: torch.Tensor
    features: Optional[torch.Tensor]
    target: Optional[torch.Tensor]

    def __init__(self, source, features=None, target=None):
        super().__init__()
        self.register_buffer("source", source)
        self.register_buffer("features", features)
        self.register_buffer("target", target)

    @property
    def has_features(self):
        return self.features is not None

    @property
    def has_target(self):
        return self.target is not None


# TODO: Add an iterable data set object for out-of-core inference.


@dataclasses.dataclass
class Dataset(data.Dataset):
    """Mappable data set.

    This class loads the entire file into memory and is therefore only suitable
    for in-core data sets.
    """

    samples: List[tsv.SampleType]
    mapper: mappers.Mapper
    parser: tsv.TsvParser  # Ditto.

    # Properties.

    @property
    def has_features(self) -> bool:
        return self.parser.has_features

    @property
    def has_target(self) -> bool:
        return self.parser.has_target

    # Required API.

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Item:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Item.
        """
        sample = self.samples[idx]
        if self.has_features:
            if self.has_target:
                source, features, target = sample
                return Item(
                    source=self.mapper.encode_source(source),
                    features=self.mapper.encode_features(features),
                    target=self.mapper.encode_target(target),
                )
            else:
                source, features = sample
                return Item(
                    source=self.mapper.encode_source(source),
                    features=self.mapper.encode_features(features),
                )
        elif self.has_target:
            source, target = sample
            return Item(
                source=self.mapper.encode_source(source),
                target=self.mapper.encode_target(target),
            )
        else:
            source = sample
            return Item(source=self.mapper.encode_source(source))
