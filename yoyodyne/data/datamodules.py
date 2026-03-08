"""Data modules."""

import logging
from typing import Iterable, Optional, Set

import lightning
from torch.utils import data

from .. import defaults
from . import collators, datasets, indexes, mappers, tsv


class Error(Exception):
    pass


class DataModule(lightning.LightningDataModule):
    """Data module.

    This is responsible for indexing the data, collating/padding, and
    generating datasets.

    Args:
        model_dir: Path for checkpoints, indexes, and logs.
        train: Path for training data TSV.
        val: Path for validation data TSV.
        predict: Path for prediction data TSV.
        test: Path for test data TSV.
        source_col: 1-indexed column in TSV containing source strings.
        features_col: 1-indexed column in TSV containing features strings.
        target_col: 1-indexed column in TSV containing target strings.
        source_sep: String used to split source string into symbols; an empty
            string indicates that each Unicode codepoint is its own symbol.
        features_sep: String used to split features string into symbols; an
            empty string indicates that each Unicode codepoint is its own
            symbol.
        target_sep: String used to split target string into symbols; an empty
            string indicates that each Unicode codepoint is its own symbol.
        tie_embeddings: Whether or not source and target embeddings are tied.
            If not, then source symbols are wrapped in {...}.
        batch_size: Desired batch size.
        max_source_length: The maximum length of a source string. An error will
            be raised if any source string exceeds this limit.
        max_features_length: The maximum length of a features string. An error
            will be raised if any features string exceeds this limit.
        max_target_length: The maximum length of a target string. A warning
            will be raised and the strings will be truncated if any target
            exceeds this limit.
    """

    train: Optional[str]
    val: Optional[str]
    predict: Optional[str]
    test: Optional[str]
    parser: tsv.TsvParser
    batch_size: int
    index: indexes.Index
    collator: collators.Collator

    def __init__(
        self,
        # Paths.
        *,
        model_dir: str,
        train: Optional[str] = None,
        val: Optional[str] = None,
        predict: Optional[str] = None,
        test: Optional[str] = None,
        # TSV parsing arguments.
        source_col: int = defaults.SOURCE_COL,
        features_col: int = defaults.FEATURES_COL,
        target_col: int = defaults.TARGET_COL,
        source_sep: str = defaults.SOURCE_SEP,
        features_sep: str = defaults.FEATURES_SEP,
        target_sep: str = defaults.TARGET_SEP,
        tie_embeddings: bool = defaults.TIE_EMBEDDINGS,
        # Other.
        batch_size: int = defaults.BATCH_SIZE,
        max_source_length: int = defaults.MAX_LENGTH,
        max_features_length: int = defaults.MAX_LENGTH,
        max_target_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        self.parser = tsv.TsvParser(
            source_col=source_col,
            features_col=features_col,
            target_col=target_col,
            source_sep=source_sep,
            features_sep=features_sep,
            target_sep=target_sep,
            tie_embeddings=tie_embeddings,
        )
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_features_length = max_features_length
        self.max_target_length = max_target_length
        # If the training data is specified, it is used to create (or recreate)
        # the index; if not specified it is read from the model directory.
        self.index = (
            self._make_index(tie_embeddings)
            if self.train
            else indexes.Index.read(self.model_dir)
        )
        self.log_vocabularies()
        self.collator = collators.Collator(
            has_features=self.has_features,
            has_target=self.has_target,
        )
        self._validate_lengths()

    def _make_index(self, tie_embeddings: bool) -> indexes.Index:
        """Creates the index from a training set."""
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        target_vocabulary: Set[str] = set()
        if self.has_features:
            if self.has_target:
                for source, features, target in self.parser.samples(
                    self.train
                ):
                    source_vocabulary.update(source)
                    features_vocabulary.update(features)
                    target_vocabulary.update(target)
            else:
                for source, features in self.parser.samples(self.train):
                    source_vocabulary.update(source)
                    features_vocabulary.update(features)
        elif self.has_target:
            for source, target in self.parser.samples(self.train):
                source_vocabulary.update(source)
                target_vocabulary.update(target)
        else:
            for source in self.parser.samples(self.train):
                source_vocabulary.update(source)
        index = indexes.Index(
            source_vocabulary=source_vocabulary,
            features_vocabulary=(
                features_vocabulary if features_vocabulary else None
            ),
            target_vocabulary=(
                target_vocabulary if target_vocabulary else None
            ),
            tie_embeddings=tie_embeddings,
        )
        # Writes it to the model directory.
        index.write(self.model_dir)
        return index

    def _validate_lengths(self) -> None:
        """Checks all samples are within the specified length limits.

        Raises:
            Error: X sample length exceeds max_X_length.
        """
        for path in [self.train, self.val, self.test, self.predict]:
            if not path:
                continue
            for lineno, sample in enumerate(self.parser.samples(path), 1):
                if self.has_features and self.has_target:
                    source, features, target = sample
                elif self.has_features:
                    source, features = sample
                elif self.has_target:
                    source, target = sample
                else:
                    source = sample
                if len(source) > self.max_source_length:
                    raise Error(
                        f"Source sample length ({len(source)}) in {path} "
                        f"(line {lineno}) exceeds max_source_length "
                        f"({self.max_source_length})"
                    )
                if (
                    self.has_features
                    and len(features) > self.max_features_length
                ):
                    raise Error(
                        f"Features sample length ({len(features)}) in {path} "
                        f"(line {lineno}) exceeds max_features_length "
                        f"({self.max_features_length})"
                    )
                if self.has_target and len(target) > self.max_target_length:
                    raise Error(
                        f"Target sample length ({len(target)}) in {path} "
                        f"(line {lineno}) exceeds max_target_length "
                        f"({self.max_target_length})"
                    )

    # Logging.

    @staticmethod
    def pprint(vocabulary: Iterable) -> str:
        """Prints the vocabulary."""
        return ", ".join(f"{symbol!r}" for symbol in vocabulary)

    def log_vocabularies(self) -> None:
        """Logs this module's vocabularies."""
        logging.info(
            "Source vocabulary (%d): %s",
            len(self.index.source_vocabulary),
            self.pprint(self.index.source_vocabulary),
        )
        if self.has_features:
            logging.info(
                "Features vocabulary (%d): %s",
                len(self.index.features_vocabulary),
                self.pprint(self.index.features_vocabulary),
            )
        if self.has_target:
            logging.info(
                "Target vocabulary (%d): %s",
                len(self.index.target_vocabulary),
                self.pprint(self.index.target_vocabulary),
            )

    # Properties.

    # has_source is always true.

    @property
    def has_features(self) -> bool:
        return self.parser.has_features

    @property
    def has_target(self) -> bool:
        return self.parser.has_target

    @property
    def target_sep(self) -> str:
        return self.parser.target_sep

    @property
    def target_vocab_size(self) -> int:
        return self.index.target_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.index.vocab_size

    # Required API.

    # The training set uses the mappable dataset because of shuffling, and
    # the validation set uses it because it is accesssed repeatedly under
    # normal conditions. Other dataloaders use the iterable dataset.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        # This uses the mappable dataset because of shuffling. Other
        # dataloaders use the iterable dataset.
        return data.DataLoader(
            datasets.MappableDataset(
                self.train, mappers.Mapper(self.index), self.parser
            ),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.val is not None, "no val path"
        return data.DataLoader(
            datasets.MappableDataset(
                self.val,
                mappers.Mapper(self.index),
                self.parser,
                sequential=True,
            ),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        return data.DataLoader(
            datasets.IterableDataset(
                self.predict, mappers.Mapper(self.index), self.parser
            ),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            datasets.IterableDataset(
                self.test, mappers.Mapper(self.index), self.parser
            ),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )
