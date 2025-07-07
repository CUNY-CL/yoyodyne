"""Base model class, with PL integration."""

import abc
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning
import torch
from torch import nn, optim

from .. import data, defaults, metrics, optimizers, schedulers, special, util
from . import modules


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
    """

    #  TODO: clean up type checking here.
    # Sizes.
    vocab_size: int
    features_vocab_size: int
    target_vocab_size: int
    # Optimizer arguments.
    beta1: float
    beta2: float
    optimizer: str
    scheduler: Optional[str]
    scheduler_kwargs: Optional[Dict]
    # Regularization arguments.
    dropout: float
    label_smoothing: float
    teacher_forcing: bool
    # Decoding arguments.
    beam_width: int
    max_source_length: int
    max_target_length: int
    # Model arguments.
    embedding_size: int
    encoder_layers: int
    decoder_layers: int
    features_encoder_cls: Optional[modules.BaseModule]
    hidden_size: int
    source_encoder_cls: modules.BaseModule
    # Loss and evaluation objects.
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # TODO: update with new metrics as they become available.
    accuracy: Optional[metrics.Accuracy]
    ser: Optional[metrics.SER]

    def __init__(
        self,
        *,
        beta1=defaults.BETA1,
        beta2=defaults.BETA2,
        features_vocab_size,
        source_encoder_cls,
        target_vocab_size,
        vocab_size,
        # All of these have keyword defaults.
        beam_width=defaults.BEAM_WIDTH,
        compute_accuracy=True,
        compute_ser=False,
        decoder_layers=defaults.DECODER_LAYERS,
        dropout=defaults.DROPOUT,
        embedding_size=defaults.EMBEDDING_SIZE,
        encoder_layers=defaults.ENCODER_LAYERS,
        features_encoder_cls=None,
        hidden_size=defaults.HIDDEN_SIZE,
        label_smoothing=defaults.LABEL_SMOOTHING,
        learning_rate=defaults.LEARNING_RATE,
        max_source_length=defaults.MAX_SOURCE_LENGTH,
        max_target_length=defaults.MAX_TARGET_LENGTH,
        optimizer=defaults.OPTIMIZER,
        scheduler=None,
        scheduler_kwargs=None,
        teacher_forcing=defaults.TEACHER_FORCING,
        **kwargs,
    ):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.features_vocab_size = features_vocab_size
        self.target_vocab_size = target_vocab_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.hidden_size = hidden_size
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.teacher_forcing = teacher_forcing
        self.embeddings = self.init_embeddings(
            self.vocab_size, self.embedding_size
        )
        # Instantiates loss and evaluation objects.
        self.loss_func = self._get_loss_func()
        # TODO: update with new metrics as they become available.
        self.accuracy = (
            metrics.Accuracy(self.target_vocab_size)
            if compute_accuracy
            else None
        )
        self.ser = metrics.SER() if compute_ser else None
        # Instantiates encoder(s).
        self.source_encoder = source_encoder_cls(
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            features_vocab_size=features_vocab_size,
            hidden_size=self.hidden_size,
            layers=self.encoder_layers,
            max_source_length=max_source_length,
            num_embeddings=self.vocab_size,
            **kwargs,
        )
        self.features_encoder = (
            features_encoder_cls(
                dropout=self.dropout,
                embedding_size=self.embedding_size,
                embeddings=self.embeddings,
                hidden_size=self.hidden_size,
                layers=self.encoder_layers,
                max_source_length=max_source_length,
                num_embeddings=self.vocab_size,
                **kwargs,
            )
            if features_encoder_cls is not None
            else None
        )
        # Instantiates decoder.
        self.decoder = self.get_decoder()
        # Saves hyperparameters for PL checkpointing.
        self.save_hyperparameters(
            ignore=["source_encoder", "decoder", "features_encoder"],
        )
        # Logs the module names.
        util.log_info(f"Model: {self.name}")
        if self.features_encoder is not None:
            util.log_info(f"Source encoder: {self.source_encoder.name}")
            util.log_info(f"Features encoder: {self.features_encoder.name}")
        else:
            util.log_info(f"Encoder: {self.source_encoder.name}")
        util.log_info(f"Decoder: {self.decoder.name}")

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

    def configure_optimizers(
        self,
    ) -> Union[
        optim.Optimizer,
        Tuple[List[optim.Optimizer], List[Dict[str, Any]]],
    ]:
        """Gets the configured torch optimizer and scheduler.

        Returns:
            Union[optim.Optimizer,
                  Tuple[List[optim.Optimizer], List[Dict[str, Any]]].
        """
        optimizer = self._get_optimizer()
        scheduler_cfg = self._get_lr_scheduler(optimizer)
        if scheduler_cfg:
            return [optimizer], [scheduler_cfg]
        else:
            return optimizer

    def _get_optimizer(self) -> optim.Optimizer:
        """Factory for selecting the optimizer.

        Returns:
            optim.Optimizer: optimizer for training.

        Raises:
            NotImplementedError: Optimizer not found.
        """
        return optimizers.get_optimizer_cfg(
            self.optimizer,
            self.parameters(),
            self.learning_rate,
            self.beta1,
            self.beta2,
        )

    def _get_lr_scheduler(self, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """Factory for selecting the scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.

        Returns:
            Dict: LR scheduler configuration dictionary.
        """
        if not self.scheduler:
            return {}
        return schedulers.get_scheduler_cfg(
            self.scheduler, optimizer, **dict(self.scheduler_kwargs)
        )

    @abc.abstractmethod
    def get_decoder(self): ...

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

    @property
    def has_features_encoder(self) -> bool:
        return self.features_encoder is not None

    @property
    def num_parameters(self) -> int:
        return sum(part.numel() for part in self.parameters())

    def start_symbol(self, batch_size: int) -> torch.Tensor:
        """Generates a tensor of start symbols for the batch."""
        return torch.tensor([special.START_IDX], device=self.device).repeat(
            batch_size, 1
        )

    def predict_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Runs one predict step.

        If beam_width > 1 this invokes the forward method directly and returns
        the predictions and scores; otherwise it returns the predictions.

        No metrics are tracked.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor].
        """
        if self.beam_width > 1:
            return self(batch)
        else:
            return torch.argmax(self(batch), dim=1)

    def training_step(
        self, batch: data.PaddedBatch, batch_idx: int
    ) -> torch.Tensor:
        """Runs one step of training.

        Training loss is tracked.

        Args:
            batch (data.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: training loss.
        """
        predictions = self(batch)
        return self.loss_func(predictions, batch.target.padded)

    def on_test_epoch_start(self) -> None:
        self._reset_metrics()

    def test_step(self, batch: data.PaddedBatch, batch_idx: int) -> None:
        """Runs one test step.

        Evaluation metrics are tracked.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).
        """
        if self.beam_width > 1:
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
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> None:
        """Runs one validation step.

        Validation loss and evaluation metrics are tracked.

        Args:
            batch (data.PaddedBatch).
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


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared configuration options to the argument parser.

    These are only needed at training time.

    Args:
        parser (argparse.ArgumentParser).
    """
    # Optimizer arguments.
    parser.add_argument(
        "--beta1",
        type=float,
        default=defaults.BETA1,
        help="beta_1 (Adam optimizer only). Default: %(default)s.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=defaults.BETA2,
        help="beta_2 (Adam optimizer only). Default: %(default)s.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=defaults.LEARNING_RATE,
        help="Learning rate. Default: %(default)s.",
    )
    parser.add_argument(
        "--optimizer",
        choices=optimizers.OPTIMIZERS,
        default=defaults.OPTIMIZER,
        help="Optimizer. Default: %(default)s.",
    )
    parser.add_argument(
        "--scheduler",
        choices=schedulers.SCHEDULERS,
        help="Learning rate scheduler.",
    )
    # Regularization arguments.
    parser.add_argument(
        "--dropout",
        type=float,
        default=defaults.DROPOUT,
        help="Dropout probability. Default: %(default)s.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=defaults.LABEL_SMOOTHING,
        help="Coefficient for label smoothing. Default: %(default)s.",
    )
    # Model arguments.
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=defaults.DECODER_LAYERS,
        help="Number of decoder layers. Default: %(default)s.",
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=defaults.ENCODER_LAYERS,
        help="Number of encoder layers. Default: %(default)s.",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=defaults.EMBEDDING_SIZE,
        help="Dimensionality of embeddings. Default: %(default)s.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=defaults.HIDDEN_SIZE,
        help="Dimensionality of the hidden layer(s). " "Default: %(default)s.",
    )
