"""Pointer-generator model base."""

from typing import Callable

import torch
from torch import nn

from ... import special
from .. import base


class PointerGeneratorModel(base.BaseModel):
    """Base class for pointer-generator models."""

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        This overrides the loss function behavior in
        models.base.BaseModel because we need to use NLLLoss in
        order to factor the addition of two separate probability
        distributions. An NLLLoss-compatible implementation of label smoothing
        is also provided when label smoothing is requested.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        if not self.label_smoothing:
            return nn.NLLLoss(ignore_index=special.PAD_IDX)
        else:
            return self._smooth_nllloss

    def _smooth_nllloss(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the NLLLoss with a smoothing factor such that some
        proportion of the output distribution is replaced with a
        uniform distribution.

        After:
            https://github.com/NVIDIA/DeepLearningExamples/blob/
            8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/
            ConvNets/image_classification/smoothing.py#L18

        Args:
            predictions (torch.Tensor): tensor of prediction
                distribution of shape B x target_vocab_size x seq_len.
            target (torch.Tensor): tensor of golds of shape
                B x seq_len.

        Returns:
            torch.Tensor: loss.
        """
        # -> (B * seq_len) x target_vocab_size.
        predictions = predictions.transpose(1, 2).reshape(
            -1, self.target_vocab_size
        )
        # -> (B * seq_len) x 1.
        target = target.view(-1, 1)
        non_pad_mask = target.ne(special.PAD_IDX)
        # Gets the ordinary loss.
        nll_loss = -predictions.gather(dim=-1, index=target)[
            non_pad_mask
        ].mean()
        # Gets the smoothed loss.
        smooth_loss = -predictions.sum(dim=-1, keepdim=True)[
            non_pad_mask
        ].mean()
        smooth_loss = smooth_loss / self.target_vocab_size
        # Combines both according to label smoothing weight.
        loss = (1.0 - self.label_smoothing) * nll_loss
        loss.add_(self.label_smoothing * smooth_loss)
        return loss
