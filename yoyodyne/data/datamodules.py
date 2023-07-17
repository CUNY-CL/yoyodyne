"""Data modules."""

from typing import Iterator, Optional, Set

import pytorch_lightning as pl
from torch.utils import data

from .. import defaults, util
from . import collators, datasets, indexes, tsv


class DataModule(pl.LightningDataModule):
    """Parses, indexes, collates and loads data."""

    parser: tsv.TsvParser
    index: indexes.Index
    batch_size: int
    collator: collators.Collator

    def __init__(
        self,
        # Paths.
        *,
        train: Optional[str] = None,
        val: Optional[str] = None,
        predict: Optional[str] = None,
        test: Optional[str] = None,
        index_path: Optional[str] = None,
        # TSV parsing arguments.
        source_col: int = defaults.SOURCE_COL,
        features_col: int = defaults.FEATURES_COL,
        target_col: int = defaults.TARGET_COL,
        # String parsing arguments.
        source_sep: str = defaults.SOURCE_SEP,
        features_sep: str = defaults.FEATURES_SEP,
        target_sep: str = defaults.TARGET_SEP,
        # Vocabulary options.
        tied_vocabulary: bool = defaults.TIED_VOCABULARY,
        # Collator options.
        batch_size=defaults.BATCH_SIZE,
        separate_features: bool = False,
        max_source_length: int = defaults.MAX_SOURCE_LENGTH,
        max_target_length: int = defaults.MAX_TARGET_LENGTH,
        # Indexing.
        index: Optional[indexes.Index] = None,
    ):
        super().__init__()
        self.parser = tsv.TsvParser(
            source_col=source_col,
            features_col=features_col,
            target_col=target_col,
            source_sep=source_sep,
            features_sep=features_sep,
            target_sep=target_sep,
        )
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        self.batch_size = batch_size
        self.separate_features = separate_features
        self.index = (
            index if index is not None else self._make_index(tied_vocabulary)
        )
        self.collator = collators.Collator(
            pad_idx=self.index.pad_idx,
            has_features=self.index.has_features,
            has_target=self.index.has_target,
            separate_features=separate_features,
            features_offset=self.index.source_vocab_size
            if self.index.has_features
            else 0,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

    def _make_index(self, tied_vocabulary: bool) -> indexes.Index:
        # Computes index.
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        target_vocabulary: Set[str] = set()
        for path in self.paths:
            if self.parser.has_features:
                if self.parser.has_target:
                    for source, features, target in self.parser.samples(path):
                        source_vocabulary.update(source)
                        features_vocabulary.update(features)
                        target_vocabulary.update(target)
                else:
                    for source, features in self.parser.samples(path):
                        source_vocabulary.update(source)
                        features_vocabulary.update(features)
            elif self.parser.has_target:
                for source, target in self.parser.samples(path):
                    source_vocabulary.update(source)
                    target_vocabulary.update(target)
            else:
                for source in self.parser.samples(path):
                    source_vocabulary.update(source)
            if self.parser.has_target and tied_vocabulary:
                source_vocabulary.update(target_vocabulary)
                target_vocabulary.update(source_vocabulary)
        return indexes.Index(
            source_vocabulary=sorted(source_vocabulary),
            features_vocabulary=sorted(features_vocabulary)
            if features_vocabulary
            else None,
            target_vocabulary=sorted(target_vocabulary)
            if target_vocabulary
            else None,
        )

    # Helpers.

    @property
    def paths(self) -> Iterator[str]:
        if self.train is not None:
            yield self.train
        if self.val is not None:
            yield self.val
        if self.predict is not None:
            yield self.predict
        if self.test is not None:
            yield self.test

    def log_vocabularies(self) -> None:
        """Logs this module's vocabularies."""
        util.log_info(f"Source vocabulary: {self.index.source_map.pprint()}")
        if self.index.has_features:
            util.log_info(
                f"Features vocabulary: {self.index.features_map.pprint()}"
            )
        if self.index.has_target:
            util.log_info(
                f"Target vocabulary: {self.index.target_map.pprint()}"
            )

    def write_index(self, model_dir: str, experiment: str) -> None:
        """Writes the index."""
        self.index.write(model_dir, experiment)

    @property
    def has_features(self) -> int:
        return self.index.has_features

    @property
    def has_target(self) -> int:
        return self.index.has_target

    @property
    def source_vocab_size(self) -> int:
        if self.separate_features:
            return self.index.source_vocab_size
        else:
            return (
                self.index.source_vocab_size + self.index.features_vocab_size
            )

    def _dataset(self, path: str) -> datasets.Dataset:
        return datasets.Dataset(
            list(self.parser.samples(path)),
            self.index,
            self.parser,
        )

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            self._dataset(self.train),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.val is not None, "no val path"
        return data.DataLoader(
            self._dataset(self.val),
            collate_fn=self.collator,
            batch_size=2 * self.batch_size,  # Because no gradients.
            num_workers=1,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        return data.DataLoader(
            self._dataset(self.predict),
            collate_fn=self.collator,
            batch_size=2 * self.batch_size,  # Because no gradients.
            num_workers=1,
        )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            self._dataset(self.test),
            collate_fn=self.collator,
            batch_size=2 * self.batch_size,  # Because no gradients.
            num_workers=1,
        )
