"""Datasets and related utilities."""

import abc
import dataclasses
import mmap
from typing import BinaryIO, Iterator, List, Optional

import torch
from torch import nn
from torch.utils import data

from .. import defaults
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
    def has_features(self) -> bool:
        return self.features is not None

    @property
    def has_target(self) -> bool:
        return self.target is not None


@dataclasses.dataclass
class AbstractDataset(abc.ABC):
    """Base class for datasets.

    Args:
        path (str).
        mapper (mappers.Mapper).
        parser (tsv.TsvParser).
    """

    path: str
    mapper: mappers.Mapper
    parser: tsv.TsvParser

    def sample_to_item(self, sample: tsv.SampleType) -> Item:
        """Converts a parsed sample into an Item using the mapper."""
        if self.parser.has_features:
            if self.parser.has_target:
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
        elif self.parser.has_target:
            source, target = sample
            return Item(
                source=self.mapper.encode_source(source),
                target=self.mapper.encode_target(target),
            )
        else:
            source = sample
            return Item(source=self.mapper.encode_source(source))


@dataclasses.dataclass
class IterableDataset(AbstractDataset, data.IterableDataset):
    """Iterable (non-random access) data set."""

    def __iter__(self) -> Iterator[Item]:
        for sample in self.parser.samples(self.path):
            yield self.sample_to_item(sample)


@dataclasses.dataclass
class MappableDataset(AbstractDataset, data.Dataset):
    """Mappable (random access) data set.

    This is implemented with a memory map after making a single pass through
    the file to compute offsets.

    Args:
        sequential (bool, optional): will this data set by used for repeated
            linear access, as is the case for validation data?
    """

    sequential: bool = False

    _offsets: List[int] = dataclasses.field(default_factory=list, init=False)
    _mmap: Optional[mmap.mmap] = dataclasses.field(default=None, init=False)
    _fobj: Optional[BinaryIO] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self._offsets = []
        with open(self.path, "rb") as source:
            offset = 0
            for line in source:
                self._offsets.append(offset)
                offset += len(line)

    def _get_mmap(self) -> mmap.mmap:
        # Makes this safe for use with multiple workers.
        if self._mmap is None:
            self._fobj = open(self.path, "rb")
            if hasattr(mmap, "MAP_POPULATE"):  # Linux-specific.
                flags = mmap.MAP_SHARED
                if not self.sequential:
                    flags |= mmap.MAP_POPULATE
                self._mmap = mmap.mmap(
                    self._fobj.fileno(),
                    0,
                    flags=flags,
                    prot=mmap.PROT_READ,
                )
                if self.sequential:
                    self._mmap.madvise(mmap.MADV_WILLNEED)
                    self._mmap.madvise(mmap.MADV_SEQUENTIAL)
                else:
                    self._mmap.madvise(mmap.MADV_RANDOM)
            else:
                self._mmap = mmap.mmap(
                    self._fobj.fileno(), 0, access=mmap.ACCESS_READ
                )
        return self._mmap

    # Required API.

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Item:
        mm = self._get_mmap()
        start = self._offsets[idx]
        if idx + 1 < len(self._offsets):
            end = self._offsets[idx + 1]
        else:
            end = mm.size()
        line = mm[start:end].decode(defaults.ENCODING).rstrip()
        sample = self.parser.parse_line(line)
        return self.sample_to_item(sample)

    def __del__(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._fobj is not None:
            self._fobj.close()

    # Properties.

    @property
    def has_features(self) -> bool:
        return self.parser.has_features

    @property
    def has_target(self) -> bool:
        return self.parser.has_target
