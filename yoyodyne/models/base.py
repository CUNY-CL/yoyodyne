"""Base model class, with PL integration."""

import abc
import logging
from typing import Callable, Optional, Tuple, Union

import lightning
from lightning.pytorch import cli

import torch
from torch import nn, optim

from .. import data, defaults, metrics, special
from . import modules


class Error(Exception):
    pass


class BaseModel(abc.ABC, lightning.LightningModule):
    """Abstract base class for models handling Lightning integration.

    The following are defaults, but can be overriden by individual models:

    * The forward method returns a tensor of shape B x target_vocab_size x
      seq_length for compatibility with loss and evaluation functions unless
      beam search is enabled.
    * Cross-entropy loss is the loss function.
    * One or more predictions tensor(s) are returned by predict_step.
    * Loss is returned by training_step.
    * Evaluation metrics are tracked by test_step; nothing is returned.
    * Validation loss and evaluation metrics are tracked by validation_step;
      nothing is returned.
    * If features_encoder is True, the source encoder will be reused as the
      features encoder and if False (the default), no features encoder will be
      used.

    Unknown positional or keyword args from the superclass are ignored.

    Args:
        source_encoder (modules.BaseModule).
        decoder_hidden_size (int, optional): dimensionality of decoder layers.
        decoder_layers (int, optional): number of decoder layers.
        decoder_dropout (float, optional): dropout probability.
        embedding_size (int, optional): dimensionality of embedding.
        features_encoder (modules.BaseModule, optional).
        label_smoothing (float, optional): label smoothing coefficient.
        max_target_length (int, optional): maximum target length.
    """

    beam_width: int
    decoder_layers: int
    decoder_hidden_size: int
    decoder_dropout: float
    label_smoothing: float
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    source_encoder: modules.BaseModule
    features_encoder: Optional[modules.BaseModule]
    accuracy: Optional[metrics.Accuracy]
    ser: Optional[metrics.SER]
    decoder: modules.BaseModule
    embedding: nn.Embedding
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        source_encoder: modules.BaseModule,
        *args,  # Ignored here.
        beam_width: int = defaults.BEAM_WIDTH,
        compute_accuracy: bool = True,
        compute_ser: bool = False,
        decoder_hidden_size: int = defaults.HIDDEN_SIZE,
        decoder_layers: int = defaults.LAYERS,
        decoder_dropout: float = defaults.DROPOUT,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        features_encoder: Union[modules.BaseModule, bool] = False,
        label_smoothing: float = defaults.LABEL_SMOOTHING,
        max_target_length: int = defaults.MAX_LENGTH,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        target_vocab_size: int = -1,  # Dummy value filled in via link.
        vocab_size: int = -1,  # Dummy value filled in via link.
        **kwargs,  # Ignored here.
    ):
        super().__init__()
        self.beam_width = beam_width
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.decoder_dropout = decoder_dropout
        self.embedding_size = embedding_size
        self.label_smoothing = label_smoothing
        self.max_target_length = max_target_length
        self.num_embeddings = vocab_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.target_vocab_size = target_vocab_size
        self.accuracy = (
            metrics.Accuracy(self.target_vocab_size)
            if compute_accuracy
            else None
        )
        self.ser = metrics.SER() if compute_ser else None
        self.embeddings = self.init_embeddings(
            self.num_embeddings, self.embedding_size
        )
        if source_encoder.embedding_size != self.embedding_size:
            raise Error(
                "Source embedding size "
                f"({source_encoder.embedding_size}) != "
                "model embedding size "
                f"({self.embedding_size})"
            )
        self.source_encoder = source_encoder
        if features_encoder is True:
            self.features_encoder = self.source_encoder
            self.has_features_encoder = True
        elif features_encoder is False:
            self.feature_encoder = None
            self.has_features_encoder = False
        else:
            if features_encoder.embedding_size != self.embedding_size:
                raise Error(
                    "Features embedding size "
                    f"({features_encoder.embedding_size}) != "
                    "model embedding size "
                    f"({self.embedding_size})"
                )
            self.features_encoder = features_encoder
            self.has_features_encoder = True
        self.decoder = self.get_decoder()
        self.loss_func = self._get_loss_func()
        self.save_hyperparameters(
            ignore=[
                "source_encoder",
                "features_encoder",
                "decoder",
            ]
        )
        self._log_model()

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        This is overridden by certain classes which compute loss as a side
        effect of training and/or inference.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        return nn.CrossEntropyLoss(
            ignore_index=special.PAD_IDX,
            label_smoothing=self.label_smoothing,
        )

    def _log_model(self) -> None:
        logging.info("Model: %s", self.name)
        if self.has_features_encoder:
            if self.source_encoder == self.features_encoder:
                logging.info(
                    "Source/features encoder: %s", self.source_encoder.name
                )
            else:
                logging.info("Source encoder: %s", self.source_encoder.name)
                logging.info(
                    "Features encoder: %s", self.features_encoder.name
                )
        else:
            logging.info("Encoder: %s", self.source_encoder.name)
        logging.info("Decoder: %s", self.decoder.name)

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

    @abc.abstractmethod
    def get_decoder(self) -> modules.BaseModule: ...

    @staticmethod
    @abc.abstractmethod
    def init_embeddings(
        num_embeddings: int, embedding_size: int
    ) -> nn.Embedding: ...

    # TODO: update with new metrics as they become available.

    @property
    def has_accuracy(self) -> bool:
        return self.accuracy is not None

    @property
    def has_ser(self) -> bool:
        return self.ser is not None

    def start_symbol(self, batch_size: int) -> torch.Tensor:
        """Generates a tensor of start symbols for the batch."""
        return torch.tensor([special.START_IDX], device=self.device).repeat(
            batch_size, 1
        )

    def predict_step(
        self,
        batch: data.Batch,
        batch_idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Runs one predict step.

        If beam_width > 1 this invokes the forward method directly and returns
        the predictions and scores; otherwise it returns the predictions.

        No metrics are tracked.

        Args:
            batch (data.Batch).
            batch_idx (int).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor].
        """
        if self.beam_width > 1:
            return self(batch)
        else:
            return torch.argmax(self(batch), dim=1)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        """Runs one step of training.

        Training loss is tracked.

        Args:
            batch (data.Batch)
            batch_idx (int).

        Returns:
            torch.Tensor: training loss.
        """
        predictions = self(batch)
        loss = self.loss_func(predictions, batch.target.padded)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def on_test_epoch_start(self) -> None:
        self._reset_metrics()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        """Runs one test step.

        Evaluation metrics are tracked.

        Args:
            batch (data.Batch).
            batch_idx (int).
        """
        if self.beam_width > 1:
            # TODO: make this workable with `yoyodyne test`.
            predictions, _ = self(batch)
        else:
            predictions = torch.argmax(self(batch), dim=1)
        self._update_metrics(predictions, batch.target.padded)

    def on_test_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("test")

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def validation_step(
        self,
        batch: data.Batch,
        batch_idx: int,
    ) -> None:
        """Runs one validation step.

        Validation loss and evaluation metrics are tracked.

        Args:
            batch (data.Batch).
            batch_idx (int).
        """
        predictions = self(batch)
        loss = self.loss_func(predictions, batch.target.padded)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        self._update_metrics(predictions, batch.target.padded)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("val")

    # Helpers to make it run.

    def _reset_metrics(self) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.accuracy.reset()
        if self.has_ser:
            self.ser.reset()

    def _update_metrics(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.accuracy.update(predictions, target)
        if self.has_ser:
            self.ser.update(predictions, target)

    def _log_metrics_on_epoch_end(self, subset: str) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.log(
                f"{subset}_accuracy",
                self.accuracy.compute(),
                logger=True,
                on_epoch=True,
                prog_bar=True,
            )
        if self.has_ser:
            self.log(
                f"{subset}_ser",
                self.ser.compute(),
                logger=True,
                on_epoch=True,
                prog_bar=True,
            )
