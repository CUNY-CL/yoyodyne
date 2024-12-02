"""Data modules."""

from typing import Iterable, Optional, Set

import lightning
from torch.utils import data

from .. import defaults, util
from . import collators, datasets, indexes, mappers, tsv


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
        separate_features: Whether or not a separate encoder should be used
            for features.
        tie_embeddings: Whether or not source and target embeddings are tied.
            If not, then source symbols are wrapped in {...}.
        batch_size: Desired batch size.
        max_source_length: The maximum length of a source string; this includes
            concatenated feature strings if not using separate features. An
            error will be raised if any source exceeds this limit.
        max_target_length: The maximum length of a target string. A warning
            will be raised and the target strings will be truncated if any
            target exceeds this limit.
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
        train=None,
        val=None,
        predict=None,
        test=None,
        # TSV parsing arguments.
        source_col: int = defaults.SOURCE_COL,
        features_col: int = defaults.FEATURES_COL,
        target_col: int = defaults.TARGET_COL,
        source_sep: str = defaults.SOURCE_SEP,
        features_sep: str = defaults.FEATURES_SEP,
        target_sep: str = defaults.TARGET_SEP,
        # Modeling options.
        separate_features: bool = False,
        tie_embeddings: bool = defaults.TIE_EMBEDDINGS,
        # Other.
        batch_size: int = defaults.BATCH_SIZE,
        max_source_length: int = defaults.MAX_SOURCE_LENGTH,
        max_target_length: int = defaults.MAX_TARGET_LENGTH,
    ):
        super().__init__()
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
        # If the training data is specified, it is used to create (or recreate)
        # the index; if not specified it is read from the model directory.
        self.index = (
            self._make_index(model_dir, tie_embeddings)
            if self.train
            else indexes.Index.read(model_dir)
        )
        self.collator = collators.Collator(
            has_features=self.has_features,
            has_target=self.has_target,
            separate_features=separate_features,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

    def _make_index(
        self, model_dir: str, tie_embeddings: bool
    ) -> indexes.Index:
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
            target_vocabulary=target_vocabulary if target_vocabulary else None,
            tie_embeddings=tie_embeddings,
        )
        # Writes it to the model directory.
        index.write(model_dir)
        return index

    # Logging.

    @staticmethod
    def pprint(vocabulary: Iterable) -> str:
        """Prints the vocabulary for debugging dnd logging purposes."""
        return ", ".join(f"{symbol!r}" for symbol in vocabulary)

    def log_vocabularies(self) -> None:
        """Logs this module's vocabularies."""
        util.log_info(
            f"Source vocabulary: {self.pprint(self.index.source_vocabulary)}"
        )
        if self.has_features:
            util.log_info(
                f"Features vocabulary: "
                f"{self.pprint(self.index.features_vocabulary)}"
            )
        if self.has_target:
            util.log_info(
                f"Target vocabulary: "
                f"{self.pprint(self.index.target_vocabulary)}"
            )

    # Properties.

    @property
    def has_features(self) -> bool:
        return self.parser.has_features

    @property
    def has_target(self) -> bool:
        return self.parser.has_target

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            self._dataset(self.train),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.val is not None, "no val path"
        return data.DataLoader(
            self._dataset(self.val),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        return data.DataLoader(
            self._dataset(self.predict),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            self._dataset(self.test),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def _dataset(self, path: str) -> datasets.Dataset:
        return datasets.Dataset(
            list(self.parser.samples(path)),
            mappers.Mapper(self.index),
            self.parser,
        )
