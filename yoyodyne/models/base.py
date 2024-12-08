"""Base model class, with PL integration."""

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning
import torch
from torch import nn, optim

from .. import (
    data,
    defaults,
    metrics,
    optimizers,
    schedulers,
    special,
    util,
)
from . import modules


class BaseModel(lightning.LightningModule):
    """Base class, handling Lightning integration."""

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
    features_encoder_cls: Optional[modules.base.BaseModule]
    hidden_size: int
    source_encoder_cls: modules.base.BaseModule
    # Evaluation objects.
    accuracy: Optional[metrics.Accuracy]
    ser: Optional[metrics.SymbolErrorRate]
    # Model parameters.
    dropout_layer: nn.Dropout
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

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
        compute_accuracy=False,
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
        # Symbol processing.
        self.vocab_size = vocab_size
        self.features_vocab_size = features_vocab_size
        self.target_vocab_size = target_vocab_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.loss_func = self._get_loss_func()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.teacher_forcing = teacher_forcing
        self.beam_width = beam_width
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.decoder_layers = decoder_layers
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.hidden_size = hidden_size
        self.embeddings = self.init_embeddings(
            self.vocab_size, self.embedding_size
        )
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
        # Checks compatibility with feature encoder and dataloader.
        modules.check_encoder_compatibility(
            source_encoder_cls, features_encoder_cls
        )
        # Instantiates encoder(s).
        self.source_encoder = source_encoder_cls(
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            features_vocab_size=self.features_vocab_size,
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
        # Logs the module names.
        util.log_info(f"Model: {self.name}")
        if self.features_encoder is not None:
            util.log_info(f"Source encoder: {self.source_encoder.name}")
            util.log_info(f"Features encoder: {self.features_encoder.name}")
        else:
            util.log_info(f"Encoder: {self.source_encoder.name}")
        util.log_info(f"Decoder: {self.decoder.name}")
        # Instantiates metrics objects.
        self.accuracy = (
            metrics.Accuracy(self.target_vocab_size)
            if compute_accuracy
            else None
        )
        self.ser = metrics.SymbolErrorRate() if compute_ser else None
        # Saves hyperparameters for PL checkpointing.
        self.save_hyperparameters(
            ignore=["source_encoder", "decoder", "features_encoder"],
        )

    @staticmethod
    def init_embeddings(num_embed: int, embed_size: int) -> nn.Embedding:
        """Method interface for initializing the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.

        Raises:
            NotImplementedError: This method needs to be overridden.

        Returns:
            nn.Embedding: embedding layer.
        """
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    # Properties.

    @property
    def compute_accuracy(self) -> bool:
        return self.accuracy is not None

    @property
    def compute_ser(self) -> bool:
        return self.ser is not None

    @property
    def has_features_encoder(self) -> bool:
        return self.features_encoder is not None

    @property
    def num_parameters(self) -> int:
        return sum(part.numel() for part in self.parameters())

    # Implemented Lightning interface.

    def training_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs one step of training.

        This tracks training loss but no other metric.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int): ignored.

        Returns:
            torch.Tensor: loss.
        """
        # -> B x seq_len x target_vocab_size.
        predictions = self(batch)
        return self._log_loss(predictions, batch.target.padded, "train")

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def validation_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> None:
        """Runs one validation step.

        This tracks validation loss and any other metrics enabled.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).
        """
        predictions = self(batch)
        self._log_loss(predictions, batch.target.padded, "val")
        self._update_metrics(
            self._get_predicted(predictions), batch.target.padded
        )

    def on_validation_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("val")

    def on_test_epoch_start(self) -> None:
        self._reset_metrics()

    def test_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> None:
        """Runs one test step.

        This tracks test loss and any other metrics enabled.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).
        """
        predictions = self(batch)
        self._log_loss(predictions, batch.target.padded, "test")
        self._update_metrics(
            self._get_predicted(predictions), batch.target.padded
        )

    def on_test_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("test")

    def _log_loss(
        self, predictions: torch.Tensor, target: torch.Tensor, subset: str
    ) -> torch.Tensor:
        predictions = predictions.transpose(1, 2)
        # Truncates predictions to the size of the target for loss computation.
        predictions = torch.narrow(predictions, 2, 0, target.size(1))
        loss = self.loss_func(predictions, target)
        self.log(
            f"{subset}_loss",
            loss,
            batch_size=len(target),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def _reset_metrics(self) -> None:
        # Loss is automatically reset between batches.
        if self.compute_accuracy:
            self.accuracy.reset()
        if self.compute_ser:
            self.ser.reset()

    def _update_metrics(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> None:
        if self.compute_accuracy:
            self.accuracy.update(predictions, target)
        if self.compute_ser:
            self.ser.update(predictions, target)

    def _log_metrics_on_epoch_end(self, subset: str) -> None:
        if self.compute_accuracy:
            self.log(
                f"{subset}_accuracy",
                self.accuracy.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        if self.compute_ser:
            self.log(
                f"{subset}_ser",
                self.ser.compute(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

    def predict_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Runs one predict step.

        This doesn't track any metrics.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: if
                using beam search, the predictions and scores as a tuple of
                tensors; if using greedy search, the predictions as a tensor.
        """
        if self.beam_width > 1:
            predictions, scores = self(batch)
            # Calling `_get_predicted` is not necessary as it is called
            # inside the beam search.
            return predictions, scores
        else:
            return self._get_predicted(self(batch))

    @staticmethod
    def _get_predicted(predictions: torch.Tensor) -> torch.Tensor:
        """Picks the best index from the vocabulary.

        Args:
            predictions (torch.Tensor): B x seq_len x target_vocab_size.

        Returns:
            torch.Tensor: indices of the argmax at each timestep.
        """
        assert predictions.ndim == 3
        return torch.argmax(predictions, dim=2)

    def configure_optimizers(
        self,
    ) -> Union[
        optim.Optimizer, Tuple[List[optim.Optimizer], List[Dict[str, Any]]]
    ]:
        """Gets the configured torch optimizer and scheduler.

        This is called by the PL Trainer.

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

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        This is overriden by certain classes which compute loss as a side
        effect of training or inference.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        return nn.CrossEntropyLoss(
            ignore_index=special.PAD_IDX,
            label_smoothing=self.label_smoothing,
        )

    @staticmethod
    def add_predict_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds shared configuration options to the argument parser.

        These are only needed at prediction time.

        Args:
            parser (argparse.ArgumentParser).
        """
        # Beam search arguments.
        parser.add_argument(
            "--beam_width",
            type=int,
            required=False,
            help="Size of the beam for beam search. Default: %(default)s.",
        )

    @staticmethod
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
            help="Dimensionality of the hidden layer(s). "
            "Default: %(default)s.",
        )
