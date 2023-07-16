"""Data modules."""

from typing import Optional, Set

import pytorch_lightning as pl
from torch.utils import data

from .. import defaults, util
from . import collators, datasets, indexes, tsv


class DataModule(pl.LightningDataModule):
    """Parses, indexes, collates and loads data."""

    tsv_parser: tsv.TsvParser
    string_parser: tsv.StringParser
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
    ):
        super().__init__()
        self.tsv_parser = tsv.TsvParser(source_col, features_col, target_col)
        self.string_parser = tsv.StringParser(
            source_sep, features_sep, target_sep
        )
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        # Computes index.
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        target_vocabulary: Set[str] = set()
        for path in [self.train, self.val, self.predict, self.test]:
            if path is None:
                continue
            if self.tsv_parser.has_features:
                if self.tsv_parser.has_target:
                    for source, features, target in self.tsv_parser.samples(
                        path
                    ):
                        source_vocabulary.update(
                            self.string_parser.source_symbols(source)
                        )
                        features_vocabulary.update(
                            self.string_parser.features_symbols(features)
                        )
                        target_vocabulary.update(
                            self.string_parser.target_symbols(target)
                        )
                else:
                    for source, features in self.tsv_parser.samples(path):
                        source_vocabulary.update(
                            self.string_parser.source_symbols(source)
                        )
                        features_vocabulary.update(
                            self.string_parser.features_symbols(features)
                        )
            elif self.tsv_parser.has_target:
                for source, target in self.tsv_parser.samples(path):
                    source_vocabulary.update(
                        self.string_parser.source_symbols(source)
                    )
                    target_vocabulary.update(
                        self.string_parser.target_symbols(target)
                    )
            else:
                for source in self.tsv_parser.samples(path):
                    source_vocabulary.update(
                        self.string_parser.source_symbols(source)
                    )
            if self.tsv_parser.has_target and tied_vocabulary:
                source_vocabulary.update(target_vocabulary)
                target_vocabulary.update(source_vocabulary)
        self.separate_features = separate_features
        self.index = indexes.Index(
            source_vocabulary=sorted(source_vocabulary),
            # These two are stored as nulls if empty.
            features_vocabulary=sorted(features_vocabulary)
            if self.separate_features
            else None,
            target_vocabulary=sorted(target_vocabulary),
        )
        # Stores batch size.
        self.batch_size = batch_size
        # Makes collator.
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

    # Helpers.

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
        index_path = self.index.index_path(model_dir, experiment)
        self.index.write(index_path)
        util.log_info(f"Index path: {index_path}")

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
            list(self.tsv_parser.samples(path)),
            self.index,
            self.string_parser,
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
