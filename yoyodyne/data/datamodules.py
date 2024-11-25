"""Data modules."""

from typing import Iterable, Optional

import lightning
from torch.utils import data

from .. import defaults, util
from . import collators, datasets, indexes, mappers, tsv


class DataModule(lightning.LightningDataModule):
    """Data module.

    This class is initialized by the LightningCLI interface. It manages all
    data loading steps.

    Args:
        model_dir: Path for checkpoints, indexes, and logs.
        predict

    """

    predict: Optional[str]
    test: Optional[str]
    train: Optional[str]
    val: Optional[str]
    parser: tsv.TsvParser
    separate_features: bool
    batch_size: int
    index: indexes.Index
    collator: collators.Collator

    def __init__(
        self,
        # Paths.
        *,
        model_dir: str,
        predict=None,
        train=None,
        test=None,
        val=None,
        # TSV parsing options.
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
        self.separate_features = separate_features
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
        source_vocabulary = set()
        features_vocabulary = set() if self.has_features else None
        target_vocabulary = set() if self.has_target else None
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
        index.write(model_dir)
        return index

    # Logging.

    @staticmethod
    def pprint(vocabulary: Iterable) -> str:
        """Prints the vocabulary for debugging adn logging purposes."""
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

    @property
    def source_vocab_size(self) -> int:
        if self.separate_features:
            return self.index.source_vocab_size
        else:
            return (
                self.index.source_vocab_size + self.index.features_vocab_size
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
