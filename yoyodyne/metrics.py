"""Metadata for validation metrics.

The computation of loss is built into the models. A slight modification of
a built-in class from torchmetrics is used to compute exact match accuracy. A
novel symbol error rate (SER) implementation is also provided.


Adding additional metrics is relatively easy, though there are a lot of steps.
Suppose one wants to add a metric called Wham. Then one must:

* Implement `Wham(torchmetrics.Metric)` in this module.
* Register `Wham` in `_metric_fac` in this module.
* Add the following to `get_model_from_argparse_args` in `train.py`:

    compute_wham=metrics.compute_metric(args, "wham")

* Add the following to the `BaseModel` in `models/base.py`:
    - add `wham: Optional[metrics.Wham]` to the member type declarations
    - add `compute_wham=False` to the constructor's arguments
    - add `self.wham = metric.Wham(...) if compute_wham else None` to the
      body of the constructor
    - add the following property:

        @property
        def has_wham(self) -> bool:
            return self.wham is not None

    - add the following to the body of `_reset_metrics`:

        if self.has_wham:
            self.wham.reset()

    - add the following to the body of `_update_metrics`:

        if self.has_wham:
            self.wham.update(predictions, target)

    - add the following to the body of `_log_metrics_on_epoch_end`:

        if self.has_wham:
            self.log(
                f"{subset}_wham",
                self.wham.compute(),
                logger=True,
                on_epoch=True,
                prog_bar=True,
            )
"""

import argparse
import collections

import numpy
import torch
import torchmetrics

from . import defaults, special, util


class Error(Exception):
    pass


# This maps metrics, as specified as command-line strings, onto three
# pieces of data:
#
# * `filename` is an f-string template (note it is not an f-string literal)
#   which is used when that metric is used for checkpointing.
#   It includes the integer `epoch` and the value of `monitor` (see below)
#   in the template.
# * `mode` is either "max" or "min" and indicates whether we want to
#   maximize or minimize the metric.
# * `monitor` is the name of the metric with a with `val_` prefix.


Metric = collections.namedtuple("Metric", ["filename", "mode", "monitor"])


_metric_fac = {
    "accuracy": Metric(
        "model-{epoch:03d}-{val_accuracy:.3f}", "max", "val_accuracy"
    ),
    "loss": Metric(
        "model-{epoch:03d}-{val_loss:.3f}",
        "min",
        "val_loss",
    ),
    "ser": Metric(
        "model-{epoch:03d}-{val_ser:.3f}",
        "min",
        "val_ser",
    ),
}


def get_metric(name: str) -> Metric:
    return _metric_fac[name]


# Implements metrics.


class Accuracy(torchmetrics.classification.MulticlassExactMatch):
    """Exact match string accuracy ignoring padding symbols."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, ignore_index=special.PAD_IDX, **kwargs)


class SER(torchmetrics.Metric):
    r"""Defines a symbol error rate metric.

    Symbol error rate is essentially minimum edit distance, in "symbols" (not
    characters, as in the `torchmetrics.text` module) scaled by the length
    of the gold hypotheses. Theoretically its range is $[0, \infty]$ but in
    practice it is usually in [0, 1]; smaller is better.

    Some definitions multiple this by 100; we don't bother here.

    We assume tensors of shape B x seq_len as input. For reasons documented
    below, seq_len must be $< 2^16$.

    This is intended to be a corpus-level statistic, so the number of edits and
    the lengths of strings are stored separately and are only combined as
    needed.

    It is not obvious whether the dynamic programming table ought to be stored
    on CPU, using a Numpy array, or in a 2d tensor on the accelerator. Since
    there is no need to track gradients and since it is accessed in a way that
    is likely to make good use of a CPU cache, the current implementation
    assumes the former, but this can be re-evaluated in light of profiling.

    One can imagine imposing different user-specified costs for the different
    edit operations, but rule this out for YAGNI reasons.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("edits", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("length", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, hypo: torch.Tensor, gold: torch.Tensor) -> None:
        """Accumulates edit distance sufficient statistics for a batch.

        This also performs all the necessary data validation work:

        Args:
            hypo (torch.Tensor): a tensor of hypothesis data of shape
                B x target_vocab_size x seq_len or B x seq_len.
            gold (torch.Tensor): a tensor of gold data of shape B x seq_len.

        Raises:
            Error: Hypothesis tensor is not 2d or 3d.
            Error: Gold tensor is not 2d.
            Error: Hypothesis and gold batch sizes do not match.
            Error: Hypothesis string lengths exceeds precision.
            Error: Gold string lengths exceeds precision.
        """
        if hypo.ndim < 2 or hypo.ndim > 3:
            raise Error(f"Hypothesis tensor is not 2d or 3d ({hypo.ndim})")
        if hypo.ndim == 3:
            hypo = torch.argmax(hypo, dim=1)
        if gold.ndim != 2:
            raise Error(f"Gold tensor is not 2d ({gold.ndim})")
        if hypo.size(0) != gold.size(0):
            raise Error(
                "Hypothesis and gold batch sizes do not match "
                f"({gold.size(0)} != {hypo.size(0)})"
            )
        # uint16 is used for the dynamic programming table, so this
        # implementation is not necessarily correct for strings longer than
        # $2^16 = 65536$. This is not much of a limitation in practice because
        # quadratic growth makes the computation infeasible at that length
        # anyways. This checks the length of the second dimension to ensure it
        # does not exceed this length.
        max_size = numpy.iinfo(numpy.uint16).max
        if hypo.size(-1) > max_size:
            raise Error(
                "Hypothesis string lengths exceeds precision "
                f"({hypo.size(-1)} > {max_size})"
            )
        if gold.size(-1) > max_size:
            raise Error(
                "Gold string lengths exceeds precision "
                f"({gold.size(-1)} > {max_size})"
            )
        # Iterates over every element in batch.
        for hypo_row, gold_row in zip(hypo, gold):
            self._row_edit_distance(hypo_row, gold_row)

    def _row_edit_distance(
        self,
        hypo: torch.Tensor,
        gold: torch.Tensor,
    ) -> None:
        """Computes edit distance sufficient statistics for single tensors.

        This makes the following assumptions about the input tensors:

        * They are 1d and can be interpreted as single strings.
        * The end of the string is delimited by END_IDX; all indices before
          the first END_IDX are part of the string and all indices after it
          are not.

        Args:
            hypo (torch.Tensor): hypothesis 1d tensor.
            gold (torch.Tensor): gold 1d tensor.
        """
        # The - 1 term reflects that `END_IDX` is not part of the string
        # with respect to edit distance. This also cannot fail.
        gold_length = torch.nonzero(gold == special.END_IDX)[0].item() - 1
        self.length += gold_length
        try:
            hypo_length = torch.nonzero(hypo == special.END_IDX)[0].item() - 1
        except IndexError:
            hypo_length = -1
        if hypo_length < 0:
            # If END_IDX isn't present, we'll consider this is a "total loss"
            # with an edit distance equivalent to the gold length. One can
            # imagine more elaborate strategies but this oughta do.
            self.edits += gold_length
            return
        table = numpy.zeros(
            (hypo_length + 1, gold_length + 1), dtype=numpy.uint16
        )
        table[:, 0] = range(hypo_length + 1)
        table[0, :] = range(gold_length + 1)
        for i in range(1, hypo_length + 1):
            for j in range(1, gold_length + 1):
                if hypo[i - 1] == gold[j - 1]:
                    table[i, j] = table[i - 1, j - 1]
                else:
                    table[i, j] = (
                        min(
                            table[i - 1, j],
                            table[i, j - 1],
                            table[i - 1, j - 1],
                        )
                        + 1
                    )
        self.edits += table[-1, -1]

    def compute(self) -> torch.Tensor:
        return self.edits / self.length


# Helper function to determine whether we need to compute a metric.


def compute_metric(args: argparse.Namespace, metric: str) -> bool:
    """Tests whether a metric needs to be tracked.

    Args:
        args (argparse.Namespace).
        metric (str)

    Return:
        True iff the metric needs to be tracked.
    """
    return (
        metric == args.checkpoint_metric
        or metric == args.reduceonplateau_metric
        or metric == args.patience_metric
        or metric in args.eval_metric
    )


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared configuration options to the argument parser.

    These are only needed at training time.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--checkpoint_metric",
        choices=_metric_fac.keys(),
        default=defaults.CHECKPOINT_METRIC,
        help="Selects checkpoints to maximize validation `accuracy`, "
        "or to minimize validation `loss` or `ser`. "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--eval_metric",
        action=util.UniqueAddAction,
        choices=_metric_fac.keys(),
        default=defaults.EVAL_METRICS,
        help="Additional metrics to compute.",
    )
    parser.add_argument(
        "--patience_metric",
        choices=_metric_fac.keys(),
        default=defaults.PATIENCE_METRIC,
        help="Stops early when validation `accuracy` does not increase or "
        "when validation `loss` or `ser` does not decrease. "
        "Default: %(default)s.",
    )
