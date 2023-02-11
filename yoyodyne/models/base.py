"""Base model class, with PL integration.

This also includes init_embeddings, which has to go somewhere.
"""

import argparse
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn, optim

from .. import batches, defaults, evaluators, schedulers, util


class BaseEncoderDecoder(pl.LightningModule):
    # Indices.
    pad_idx: int
    start_idx: int
    end_idx: int
    # Sizes.
    vocab_size: int
    output_size: int
    # Optimizer arguments.
    beta1: float
    beta2: float
    optimizer: str
    scheduler: Optional[str]
    scheduler_kwargs: Optional[Dict]
    # Regularization arguments.
    dropout: float
    label_smoothing: Optional[float]
    # Decoding arguments.
    beam_width: int
    max_target_length: int
    # Model arguments.
    decoder_layers: int
    embedding_size: int
    encoder_layers: int
    hidden_size: int
    # Constructed inside __init__.
    dropout_layer: nn.Dropout
    evaluator: evaluators.Evaluator
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        *,
        pad_idx,
        start_idx,
        end_idx,
        vocab_size,
        output_size,
        beta1=defaults.BETA1,
        beta2=defaults.BETA2,
        learning_rate=defaults.LEARNING_RATE,
        optimizer=defaults.OPTIMIZER,
        scheduler=None,
        scheduler_kwargs=None,
        dropout=defaults.DROPOUT,
        label_smoothing=None,
        beam_width=defaults.BEAM_WIDTH,
        max_target_length=defaults.MAX_TARGET_LENGTH,
        decoder_layers=defaults.DECODER_LAYERS,
        embedding_size=defaults.EMBEDDING_SIZE,
        encoder_layers=defaults.ENCODER_LAYERS,
        hidden_size=defaults.HIDDEN_SIZE,
        **kwargs,  # Ignored.
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.beam_width = beam_width
        self.max_target_length = max_target_length
        self.decoder_layers = decoder_layers
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.hidden_size = hidden_size
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
        self.evaluator = evaluators.Evaluator()
        self.loss_func = self._get_loss_func("mean")
        # Saves hyperparameters for PL checkpointing.
        self.save_hyperparameters()

    @staticmethod
    def _xavier_embedding_initialization(
        num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embeddings layer using Xavier initialization.

        The pad embeddings are also zeroed out.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        embedding_layer = nn.Embedding(num_embeddings, embedding_size)
        # Xavier initialization.
        nn.init.normal_(
            embedding_layer.weight, mean=0, std=embedding_size**-0.5
        )
        # Zeroes out pad embeddings.
        if pad_idx is not None:
            nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
        return embedding_layer

    @staticmethod
    def _normal_embedding_initialization(
        num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embeddings layer from a normal distribution.

        The pad embeddings are also zeroed out.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        embedding_layer = nn.Embedding(num_embeddings, embedding_size)
        # Zeroes out pad embeddings.
        if pad_idx is not None:
            nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
        return embedding_layer

    @staticmethod
    def init_embeddings(
        num_embed: int, embed_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Method interface for initializing the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Raises:
            NotImplementedError: This method needs to be overridden.

        Returns:
            nn.Embedding: embedding layer.
        """
        raise NotImplementedError

    def training_step(
        self,
        batch: batches.PaddedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (batches.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        # -> B x seq_len x output_size.
        predictions = self(batch)
        target_padded = batch.target.padded
        # -> B x output_size x seq_len. For loss.
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

    def validation_step(
        self,
        batch: batches.PaddedBatch,
        batch_idx: int,
    ) -> Dict:
        """Runs one validation step.

        This is called by the PL Trainer.

        Args:
            batch (batches.PaddedBatch).
            batch_idx (int).

        Returns:
            Dict[str, float]: validation metrics.
        """
        # Greedy decoding.
        # -> B x seq_len x output_size.
        predictions = self(batch)
        target_padded = batch.target.padded
        accuracy = self.evaluator.val_accuracy(
            predictions, target_padded, self.end_idx, self.pad_idx
        )
        # We rerun the model with teacher forcing so we can compute loss.
        # TODO: Update to run the model only once.
        forced_predictions = self(batch)
        # -> B x output_size x seq_len. For loss.
        forced_predictions = forced_predictions.transpose(1, 2)
        loss = self.loss_func(forced_predictions, target_padded)
        return {"val_accuracy": accuracy, "val_loss": loss}

    def validation_epoch_end(self, validation_step_outputs: Dict) -> Dict:
        """Computes average loss and average accuracy.

        Args:
            validation_step_outputs (Dict).

        Returns:
            Dict: averaged metrics over all validation steps.
        """
        keys = validation_step_outputs[0].keys()
        # Shortens name to do comprehension below.
        # TODO: there has to be a more elegant way to do this.
        V = validation_step_outputs
        metrics = {k: sum([v[k] for v in V]) / len(V) for k in keys}
        for metric, value in metrics.items():
            self.log(metric, value, prog_bar=True)
        return metrics

    def predict_step(
        self,
        batch: batches.PaddedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs one predict step.

        This is called by the PL Trainer.

        Args:
            batch (batches.PaddedBatch).
            batch_idx (int).

        Returns:
            torch.Tensor: indices of the argmax at each timestep.
        """
        predictions = self(batch)
        # -> B x seq_len x 1.
        greedy_predictions = self._get_predicted(predictions)
        return greedy_predictions

    def _get_predicted(self, predictions: torch.Tensor) -> torch.Tensor:
        """Picks the best index from the vocabulary.

        Args:
            predictions (torch.Tensor): B x seq_len x output_size.

        Returns:
            torch.Tensor: indices of the argmax at each timestep.
        """
        assert len(predictions.size()) == 3
        _, indices = torch.max(predictions, dim=2)
        return indices

    def configure_optimizers(self) -> optim.Optimizer:
        """Gets the configured torch optimizer.

        This is called by the PL Trainer.

        Returns:
            optim.Optimizer: optimizer for training.
        """
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer[0])
        if scheduler:
            util.log_info("Scheduler details:")
            util.log_info(scheduler)
        else:
            util.log_info("Optimizer details:")
            util.log_info(optimizer)
        return optimizer, scheduler

    def _get_optimizer(self) -> optim.Optimizer:
        """Factory for selecting the optimizer.

        Returns:
            optim.Optimizer: optimizer for training.
        """
        optim_fac = {
            "adadelta": optim.Adadelta,
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }
        optimizer = optim_fac[self.optimizer]
        kwargs = {"lr": self.learning_rate}
        if self.optimizer == "adam":
            kwargs["betas"] = self.beta1, self.beta2
        return [optimizer(self.parameters(), **kwargs)]

    def _get_lr_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler:
        """Factory for selecting the scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.

        Returns:
            optim.lr_scheduler: LR scheduler for training.
        """
        if self.scheduler is None:
            return []
        scheduler_fac = {
            "warmupinvsqrt": schedulers.WarmupInverseSquareRootSchedule,
            "lineardecay": schedulers.LinearDecay,
        }
        scheduler_cls = scheduler_fac[self.scheduler]
        scheduler = scheduler_cls(
            **dict(self.scheduler_kwargs, optimizer=optimizer)
        )
        scheduler_cfg = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [scheduler_cfg]

    def _get_loss_func(
        self, reduction: str
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        Args:
            reduction (str): reduction for the loss function (e.g., "mean").

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        if self.label_smoothing is None:
            return nn.NLLLoss(ignore_index=self.pad_idx, reduction=reduction)
        else:
            return self._smooth_nllloss

    def _smooth_nllloss(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """After:

            https://github.com/NVIDIA/DeepLearningExamples/blob/
            8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/
            ConvNets/image_classification/smoothing.py#L18

        Args:
            predictions (torch.Tensor): tensor of prediction
                distribution of shape B x output_size x seq_len.
            target (torch.Tensor): tensor of golds of shape
                B x seq_len.

        Returns:
            torch.Tensor: loss.
        """
        # -> (B * seq_len) x output_size.
        predictions = predictions.transpose(1, 2).reshape(-1, self.output_size)
        # -> (B * seq_len) x 1.
        target = target.view(-1, 1)
        non_pad_mask = target.ne(self.pad_idx)
        # Gets the ordinary loss.
        nll_loss = -predictions.gather(dim=-1, index=target)[
            non_pad_mask
        ].mean()
        # Gets the smoothed loss.
        smooth_loss = -predictions.sum(dim=-1, keepdim=True)[
            non_pad_mask
        ].mean()
        smooth_loss = smooth_loss / self.output_size
        # Combines both according to label smoothing weight.
        loss = (1.0 - self.label_smoothing) * nll_loss
        loss += self.label_smoothing * smooth_loss
        return loss

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
            choices=["adadelta", "adam", "sgd"],
            default=defaults.OPTIMIZER,
            help="Optimizer. Default: %(default)s.",
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
            help="Coefficient for label smoothing.",
        )
        # TODO: add --beam_width.
        # Model arguments.
        parser.add_argument(
            "--decoder_layers",
            type=int,
            default=defaults.DECODER_LAYERS,
            help="Number of decoder layers. Default: %(default)s.",
        )
        parser.add_argument(
            "--embedding_size",
            type=int,
            default=defaults.EMBEDDING_SIZE,
            help="Dimensionality of embeddings. Default: %(default)s.",
        )
        parser.add_argument(
            "--encoder_layers",
            type=int,
            default=defaults.ENCODER_LAYERS,
            help="Number of encoder layers. Default: %(default)s.",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=defaults.HIDDEN_SIZE,
            help="Dimensionality of the hidden layer(s). "
            "Default: %(default)s.",
        )
