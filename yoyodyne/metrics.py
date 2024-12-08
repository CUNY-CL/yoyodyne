"""Metadata for validation metrics.

The computation of loss is built into the models. A slight modification of
a built-in class from torchmetrics is used to compute exact match accuracy. A
novel symbol error rate (SER) implementation is also provided."""

import argparse
import collections

import numpy
import torch
import torchmetrics

from . import defaults, special


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
        super().__init__(self, *args, ignore_index=special.PAD_IDX, **kwargs)


class SymbolErrorRate(torchmetrics.Metric):
    r"""Defines a symbol error rate metric.

    Symbol error rate is essentially minimum edit distance, in "symbols" (not
    characters, as in the `torchmetrics.text` module) scaled by the length
    of the gold hypotheses. Theoretically its range is $[0, \infty]$ but in
    practice it is usually in [0, 1]; smaller is better.

    We assume tensors of shape B x seq_len as input. For reasons documented
    below, seq_len must be $< 2^16$.

    This is intended to be a corpus-level description, not a micro-average, so
    the number of edits and the lengths of strings are stored separately.

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
        self.add_state("edits", default=0, dist_reduce_fx="sum")
        self.add_state("length", default=0, dist_reduce_fx="sum")

    def update(self, gold: torch.Tensor, hypo: torch.Tensor) -> None:
        """Accumulates edit distance sufficient statistics for a batch.

        This also performs all the necessary data validation work:

        * The first (batch size) dimension of the two tensors must match.
        * The second (sequence length) dimensions of the two tensors need not
          much, but neither can exceed $2^16$ for precision reasons.

        Args:
            gold (torch.Tensor): a tensor of gold data of shape B x seq_len.
            hypo (torch.Tensor): a tensor of hypothesis data of shape
                B x seq_len.

        Raises:
            Error: Gold tensor is not 2d.
            Error: Hypothesis tensor is not 2d.
            Error: Gold string lengths exceeds precision.
            Error: Hypothesis string lengths exceeds precision.
            Error: Gold and hypothesis batch sizes do not match.
        """
        if gold.ndim != 2:
            raise Error(f"Gold tensor is not 2d ({gold.ndim})")
        if hypo.ndim != 2:
            raise Error(f"Hypothesis tensor is not 2d ({hypo.ndim})")
        # uint16 is used for the dynamic programming table, so this
        # implementation is not necessarily correct for strings longer than
        # $2^16 = 65536$. This is not much of a limitation in practice because
        # quadratic growth makes the computation infeasible at that length
        # anyways. This checks the length of the second dimension to ensure it
        # does not exceed this length.
        sixteen_bits = 2**16
        if gold.shape(1) > sixteen_bits:
            raise Error(
                "Gold string lengths exceeds precision "
                f"({gold.shape(1)} > {sixteen_bits})"
            )
        if hypo.shape(1) > sixteen_bits:
            raise Error(
                "Hypothesis string lengths exceeds precision "
                f"({hypo.shape(1)} > {sixteen_bits})"
            )
        if gold.shape(0) != hypo.shape(1):
            raise Error(
                "Gold and hypothesis batch shapes do not match "
                f"({gold.shape(0)} != {hypo.shape(0)})"
            )
        for gold_row, hypo_row in zip(gold, hypo):
            self._row_edit_distance(gold_row, hypo_row)

    def _row_edit_distance(
        self, gold: torch.Tensor, hypo: torch.Tensor
    ) -> None:
        """Computes edit distance sufficient statistics for single tensors.

        This makes the following assumptions about the input tensors:

        * They are 1d and can be interpreted as single strings.
        * The end of the string is delimited by END_IDX; all indices before
          the first END_IDX are part of the string and all indices after it
          are not.

        Args:
            gold (torch.Tensor): gold 1d tensor viewed as string.
            hypo (torch.Tensor): gold 1d tensor viewed as string.
        """
        # This has a desired "add-one" effect; it will fail if there is no
        # END_IDX present.
        gold_len = torch.where(gold == special.END_IDX)[0].item()
        hypo_len = torch.where(hypo == special.END_IDX)[0].item()
        table = numpy.zeros((gold_len, hypo_len), dtype=numpy.uint16)
        table[:, 0] = range(gold_len)
        table[0, :] = range(hypo_len)
        for i in range(1, gold_len):
            for j in range(1, hypo_len):
                if gold[i - 1] == hypo[j - 1]:
                    table[i, j] = table[i - 1, j - 1]
                else:
                    table[i][j] = min(
                        table[i - 1, j],
                        table[i, j - 1],
                        table[i - 1, j - 1],
                    )
        self.edits += table[-1, -1]
        self.length += gold_len

    def compute(self) -> torch.Tensor:
        # TODO: this returns a scalar as a tensor since that's what the
        # documentation suggests. Test whether this is necessary.
        return torch.Tensor(self.edits / self.length)


# Helper function to determine whether we need to compute a metric.


def compute_metric(args: argparse.Namespace, metric: str) -> bool:
    """Tests whether a metric needs to be tracked.

    Args:
        args (argparse.Namespace).
        metric (str)

    Return:
        True iff the metric needs to be tracked.
    """
    if args.checkpoint_metric == metric:
        return True
    if args.patience_metric == metric:
        return True
    return metric in args.eval_metric


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
        "--patience_metric",
        choices=_metric_fac.keys(),
        default=defaults.PATIENCE_METRIC,
        help="Stops early when validation `accuracy` does not increase or "
        "when validation `loss` or `ser` does not decrease. "
        "Default: %(default)s.",
    )
