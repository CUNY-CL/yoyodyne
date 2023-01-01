"""Base model class, with PL integration.

This also includes init_embeddings, which has to go somewhere.
"""

from typing import Callable, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn, optim

from .. import schedulers, util

# TODO: Eliminate this in #33.
Batch = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]


class BaseEncoderDecoder(pl.LightningModule):
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
            nn.init.constant_(embedding_layer.weight[pad_idx], 0)
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
            nn.init.constant_(embedding_layer.weight[pad_idx], 0)
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
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (Batch): tuple of src, src_mask, target, target_mask.
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        self.train()
        targets = batch[-2]
        preds = self(batch)
        loss = self.loss_func(preds, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> Dict:
        """Runs one validation step.

        This is called by the PL Trainer.

        Args:
            batch (Batch): tuple of src, src_mask, target, target_mask.
            batch_idx (int).

        Returns:
            Dict[str, float]: validation metrics.
        """
        self.eval()
        y = batch[-2]
        # Greedy decoding
        preds = self(batch[:-2])
        accuracy = self.evaluator.val_accuracy(
            preds, y, self.end_idx, self.pad_idx
        )
        # We rerun the model with teacher forcing so we can compute a loss....
        # TODO: Update to run the model only once.
        loss = self.loss_func(self(batch), y)
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
        # TODO(kbg): there has to be a more elegant way to do this.
        V = validation_step_outputs
        metrics = {k: sum([v[k] for v in V]) / len(V) for k in keys}
        for metric, value in metrics.items():
            self.log(metric, value, prog_bar=True)
        return metrics

    def _get_predicted(self, preds: torch.Tensor) -> torch.Tensor:
        """Picks the best index from the vocabulary.

        Args:
            preds (torch.Tensor): B x seq_len x vocab_size.

        Returns:
            indices (torch.Tensor): indices of the argmax at each timestep.
        """
        assert len(preds.size()) == 3
        vals, indices = torch.max(preds, dim=2)
        return indices

    def configure_optimizers(self) -> optim.Optimizer:
        """Gets the configured torch optimizer.

        This is called by the PL Trainer.

        Returns:
            optim.Optimizer: optimizer for training.
        """
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer[0])
        util.log_info("Optimizer details:")
        util.log_info(optimizer)
        if scheduler:
            util.log_info("Scheduler details:")
            util.log_info(scheduler)
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
        # TODO: Implement multiple options.
        scheduler_fac = {
            "warmupinvsqr": schedulers.WarmupInverseSquareRootSchedule
        }
        scheduler = scheduler_fac[self.scheduler](
            optimizer=optimizer, warmup_steps=self.warmup_steps
        )
        scheduler_cfg = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [scheduler_cfg]

    def get_loss_func(
        self, reduction: str
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        At training time, this is be called to get the loss function.

        Args:
            reduction (str): reduction for the loss function (e.g., "mean").

        Returns:
            loss_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
                configured loss function.
        """
        if self.label_smoothing is None:
            return nn.NLLLoss(ignore_index=self.pad_idx, reduction=reduction)
        else:

            def _smooth_nllloss(
                predict: torch.Tensor, target: torch.Tensor
            ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                """After:
                    https://github.com/NVIDIA/DeepLearningExamples/blob/
                    8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/
                    ConvNets/image_classification/smoothing.py#L18

                Args:
                    predict (torch.Tensor): tensor of prediction distribution
                        of shape B x vocab_size x seq_len.
                    target (torch.Tensor): tensor of golds of shape
                        B x seq_len.

                Returns:
                    torch.Tensor: loss.
                """
                # -> (B * seq_len) x output_size
                predict = predict.transpose(1, 2).reshape(-1, self.output_size)
                # -> (B * seq_len) x 1
                target = target.view(-1, 1)
                non_pad_mask = target.ne(self.pad_idx)
                # Gets the ordinary loss.
                nll_loss = -predict.gather(dim=-1, index=target)[
                    non_pad_mask
                ].mean()
                # Gets the smoothed loss.
                smooth_loss = -predict.sum(dim=-1, keepdim=True)[
                    non_pad_mask
                ].mean()
                smooth_loss = smooth_loss / self.output_size
                # Combines both according to label smoothing weight.
                loss = (1.0 - self.label_smoothing) * nll_loss
                loss += self.label_smoothing * smooth_loss
                return loss

            return _smooth_nllloss
