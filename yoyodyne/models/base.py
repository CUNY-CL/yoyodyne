"""Base model class, with PL integration."""

import abc
import argparse
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import lightning
import torch
from torch import nn, optim

from .. import (
    data,
    defaults,
    evaluators,
    optimizers,
    schedulers,
    special,
    util,
)
from . import modules


class BaseModel(abc.ABC, lightning.LightningModule):
    """Abstract base class for models handling Lightning integration."""

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
    # Other stuff.
    eval_metrics: Set[evaluators.Evaluator]
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

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
        decoder_layers=defaults.DECODER_LAYERS,
        dropout=defaults.DROPOUT,
        embedding_size=defaults.EMBEDDING_SIZE,
        encoder_layers=defaults.ENCODER_LAYERS,
        eval_metrics=defaults.EVAL_METRICS,
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
        self.eval_metrics = eval_metrics
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
        self.evaluators = [
            evaluators.get_evaluator(eval_metric)()
            for eval_metric in self.eval_metrics
        ]
        # Checks compatibility with feature encoder and dataloader.
        modules.check_encoder_compatibility(
            source_encoder_cls, features_encoder_cls
        )
        # Instantiates encoders class.
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

    @abc.abstractmethod
    def get_decoder(self): ...

    @staticmethod
    @abc.abstractmethod
    def init_embeddings(
        num_embeddings: int, embedding_size: int
    ) -> nn.Embedding: ...

    @property
    def has_features_encoder(self):
        return self.features_encoder is not None

    @property
    def num_parameters(self) -> int:
        return sum(part.numel() for part in self.parameters())

    def start_symbol(self, batch_size: int) -> torch.Tensor:
        """Generates a tensor of start symbols for the batch."""
        return torch.tensor([special.START_IDX], device=self.device).repeat(
            batch_size, 1
        )

    def training_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        # -> B x seq_len x target_vocab_size.
        predictions = self(batch)
        target_padded = batch.target.padded
        # -> B x target_vocab_size x seq_len. For loss.
        predictions = predictions.transpose(1, 2)
        loss = self.loss_func(predictions, target_padded)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_epoch_end(
        self, validation_step_outputs: Dict
    ) -> Dict[str, float]:
        """Computes average loss and average accuracy.

        Args:
            validation_step_outputs (Dict).

        Returns:
            Dict[str, float]: averaged metrics over all validation steps.
        """
        avg_val_loss = torch.tensor(
            [v["val_loss"] for v in validation_step_outputs]
        ).mean()
        # Gets requested metrics.
        metrics = {
            metric_name: sum(
                (v[metric_name] for v in validation_step_outputs),
                start=evaluators.EvalItem([]),
            ).metric
            for metric_name in self.eval_metrics
        }
        # Always logs validation loss.
        metrics.update({"loss": avg_val_loss})
        # del validation_step_outputs
        for metric, value in metrics.items():
            self.log(f"val_{metric}", value, prog_bar=True)
        return metrics

    def validation_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> Dict[str, float]:
        """Runs one validation step.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).

        Returns:
            Dict[str, float]: validation metrics.
        """
        # Greedy decoding.
        # -> B x seq_len x target_vocab_size.
        target_padded = batch.target.padded
        greedy_predictions = self(batch)
        # Gets a dict of all eval metrics for this batch.
        val_eval_items_dict = {
            evaluator.name: evaluator.evaluate(
                greedy_predictions, target_padded
            )
            for evaluator in self.evaluators
        }
        # -> B x target_vocab_size x seq_len. For loss.
        greedy_predictions = greedy_predictions.transpose(1, 2)
        # Truncates predictions to the size of the target.
        greedy_predictions = torch.narrow(
            greedy_predictions, 2, 0, target_padded.size(1)
        )
        loss = self.loss_func(greedy_predictions, target_padded)
        val_eval_items_dict.update({"val_loss": loss})
        return val_eval_items_dict

    def predict_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Runs one predict step.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: if
                using beam search, the predictions and scores as a tuple of
                tensors; if using greedy search, the predictions as a tensor.
        """
        if self.beam_width > 1:
            return self(batch)
        else:
            return torch.argmax(self(batch), dim=2)


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
