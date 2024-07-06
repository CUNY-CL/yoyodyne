"""Evaluators."""

from __future__ import annotations

import abc
import argparse
import dataclasses
from typing import List

import numpy
import torch
from torch.nn import functional

from . import defaults, util


class Error(Exception):
    pass


@dataclasses.dataclass
class EvalItem:
    per_sample_metrics: List[float]

    @property
    def metric(self) -> float:
        """Computes the micro-average of the metric."""
        return numpy.mean(self.per_sample_metrics)

    def __add__(self, other_eval: EvalItem) -> EvalItem:
        """Adds two EvalItem by concatenating the list of individual metrics.

        Args:
            other_eval (EvalItem): The other eval item to add to self.

        Returns:
            EvalItem.
        """
        return EvalItem(
            self.per_sample_metrics + other_eval.per_sample_metrics
        )


class Evaluator(abc.ABC):
    """Evaluator interface."""

    def evaluate(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        predictions_finalized: bool = False,
    ) -> EvalItem:
        """Computes the evaluation metric.

        This is the top-level public method that should be called by
        during evaluation.

        Args:
            predictions (torch.Tensor): B x seq_len x vocab_size.
            golds (torch.Tensor): B x seq_len x 1.
            end_idx (int): end of sequence index.
            pad_idx (int): padding index.
            predictions_finalized (bool, optional): have the predictions
                been arg-maxed and padded?

        Returns:
            EvalItem.
        """
        if predictions.size(0) != golds.size(0):
            raise Error(
                f"Preds batch size ({predictions.size(0)}) and "
                f"golds batch size ({golds.size(0)} do not match"
            )
        if not predictions_finalized:
            # Gets the max value at each dim2 in predictions.
            _, predictions = torch.max(predictions, dim=2)
            # Finalizes the predictions.
            predictions = self.finalize_predictions(
                predictions, end_idx, pad_idx
            )
        golds = self.finalize_golds(golds, end_idx, pad_idx)
        return self.get_eval_item(predictions, golds, pad_idx)

    def get_eval_item(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> EvalItem:
        raise NotImplementedError

    def finalize_predictions(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def finalize_golds(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class AccuracyEvaluator(Evaluator):
    """Evaluates accuracy."""

    def get_eval_item(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> EvalItem:
        if predictions.size(1) > golds.size(1):
            predictions = predictions[:, : golds.size(1)]
        elif predictions.size(1) < golds.size(1):
            num_pads = (0, golds.size(1) - predictions.size(1))
            predictions = functional.pad(
                predictions, num_pads, "constant", pad_idx
            )
        # Gets the count of exactly matching tensors in the batch.
        # -> B.
        accs = (predictions.to(golds.device) == golds).all(dim=1).tolist()
        return EvalItem(accs)

    def finalize_predictions(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> torch.Tensor:
        """Finalizes predictions.

        Cuts off tensors at the first end_idx, and replaces the rest of the
        predictions with pad_idx, as these are erroneously decoded while the
        rest of the batch is finishing decoding.

        Args:
            predictions (torch.Tensor): prediction tensor.
            end_idx (int).
            pad_idx (int).

        Returns:
            torch.Tensor: finalized predictions.
        """
        return util.pad_tensor_after_eos(
            predictions,
            end_idx,
            pad_idx,
        )

    def finalize_golds(
        self,
        golds: torch.Tensor,
        *args,
        **kwargs,
    ):
        return golds

    @property
    def name(self) -> str:
        return "accuracy"


class SEREvaluator(Evaluator):
    """Evaluates symbol error rate.

    Here, a symbol is defined by the user specified tokenization."""

    def _compute_ser(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        errors = self._edit_distance(preds, target)
        total = len(target)
        return errors / total

    @staticmethod
    def _edit_distance(x: torch.Tensor, y: torch.Tensor) -> int:
        idim = len(x) + 1
        jdim = len(y) + 1
        table = numpy.zeros((idim, jdim), dtype=numpy.uint16)
        table[:, 0] = range(idim)
        table[0, :] = range(jdim)
        for i in range(1, idim):
            for j in range(1, jdim):
                if x[i - 1] == y[j - 1]:
                    table[i][j] = table[i - 1][j - 1]
                else:
                    c1 = table[i - 1][j]
                    c2 = table[i][j - 1]
                    c3 = table[i - 1][j - 1]
                    table[i][j] = min(c1, c2, c3) + 1
        return int(table[-1][-1])

    def get_eval_item(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> EvalItem:
        sers = [self._compute_ser(p, g) for p, g in zip(predictions, golds)]
        return EvalItem(sers)

    def _finalize_tensor(
        self,
        tensor: torch.Tensor,
        end_idx: int,
    ) -> List[torch.Tensor]:
        """Finalizes each tensor.

        Truncates at EOS for each prediction and returns a List of predictions.
        This does basically the same as util.pad_after_eos, but does
        not actually pad since we do not need to return a well-formed tensor.

        Args:
            tensor (torch.Tensor): _description_
            end_idx (int): _description_

        Returns:
            List[torch.Tensor]: _description_
        """
        # Not necessary if batch size is 1.
        if tensor.size(0) == 1:
            return [tensor]
        out = []
        for prediction in tensor:
            # Gets first instance of EOS.
            eos = (prediction == end_idx).nonzero(as_tuple=False)
            if len(eos) > 0 and eos[0].item() < len(prediction):
                # If an EOS was decoded and it is not the last one in the
                # sequence.
                eos = eos[0]
            else:
                # Leaves tensor[i] alone.
                out.append(prediction)
                continue
            # Hack in case the first prediction is EOS. In this case
            # torch.split will result in an error, so we change these 0's to
            # 1's, which will make the entire sequence EOS as intended.
            eos[eos == 0] = 1
            symbols, *_ = torch.split(prediction, eos)
            out.append(symbols)
        return out

    def finalize_predictions(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        *args,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Finalizes predictions.

        Args:
            predictions (torch.Tensor): prediction tensor.
            end_idx (int).
            pad_idx (int).

        Returns:
            List[torch.Tensor]: finalized predictions.
        """
        return self._finalize_tensor(predictions, end_idx)

    def finalize_golds(
        self,
        golds: torch.Tensor,
        end_idx: int,
        *args,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Finalizes predictions.

        Args:
            predictions (torch.Tensor): prediction tensor.
            end_idx (int).
            pad_idx (int).

        Returns:
            List[torch.Tensor]: finalized predictions.
        """
        return self._finalize_tensor(golds, end_idx)

    @property
    def name(self) -> str:
        return "ser"


_eval_factory = {
    "accuracy": AccuracyEvaluator,
    "ser": SEREvaluator,
}


def get_evaluator(eval_metric: str) -> Evaluator:
    """Gets the requested Evaluator given the specified metric.

    Args:
        eval_metric (str).

    Raises:
        Error.

    Returns:
        Evaluator.
    """
    try:
        return _eval_factory[eval_metric]
    except KeyError:
        raise Error(f"No evaluation metric {eval_metric}")


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds LSTM configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--eval_metric",
        action=util.UniqueAddAction,
        choices=_eval_factory.keys(),
        default=defaults.EVAL_METRICS,
        help="Additional metrics to compute. Default: none.",
    )
