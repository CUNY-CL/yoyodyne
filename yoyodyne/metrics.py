"""Validation metrics.

The computation of loss is built into the models. A slight modification of
a built-in class from torchmetrics is used to compute exact match accuracy. A
novel symbol error rate (SER) implementation is also provided.

Adding additional metrics is relatively easy, though there are a lot of steps.
Suppose one wants to add a metric called Wham. Then one must:

* Implement `Wham(torchmetrics.Metric)` in this module.
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

import torch
import torchmetrics

from . import special


class Error(Exception):
    pass


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

    This is a corpus-level statistic; the number of edits and total lengths
    are stored separately and combined when computing. Some definitions
    multiply this by 100; we don't bother here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state(
            "edits",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "length",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, hypo: torch.Tensor, gold: torch.Tensor) -> None:
        """Accumulates edit distance sufficient statistics for a batch.

        Args:
            hypo (torch.Tensor): a tensor of hypothesis data of shape
                B x target_vocab_size x seq_len or B x seq_len.
            gold (torch.Tensor): a tensor of gold data of shape B x seq_len.

        Raises:
            Error: Hypothesis tensor is not 2d or 3d.
            Error: Gold tensor is not 2d.
            Error: Hypothesis and gold batch sizes do not match.
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
        batch_size = hypo.size(0)
        hypo_length = self._get_length(hypo)
        gold_length = self._get_length(gold)
        self.length += gold_length.sum()
        max_hypo = hypo_length.max().item()
        max_gold = gold_length.max().item()
        prev_row = torch.arange(
            max_gold + 1, device=self.device, dtype=torch.int32
        ).repeat(batch_size, 1)
        for i in range(1, max_hypo + 1):
            curr_row = torch.zeros(
                (batch_size, max_gold + 1),
                device=self.device,
                dtype=torch.int32,
            )
            curr_row[:, 0] = i
            # Slice for comparison.
            hypo_sym = hypo[:, i - 1].unsqueeze(dim=1)
            for j in range(1, max_gold + 1):
                # Substitution cost.
                cost = (hypo_sym != gold[:, j - 1].unsqueeze(dim=1)).to(
                    torch.int32
                )
                options = torch.stack(
                    [
                        prev_row[:, j] + 1,  # Deletion.
                        curr_row[:, j - 1] + 1,  # Insertion.
                        prev_row[:, j - 1] + cost.squeeze(),  # Substitution.
                    ]
                )
                curr_row[:, j] = options.amin(dim=0)
            # Only updates rows for batch items that haven't reached hypo_length.
            mask = (i <= hypo_length).unsqueeze(dim=1)
            prev_row = torch.where(mask, curr_row, prev_row)
        batch_edits = prev_row.gather(
            1, gold_length.unsqueeze(dim=1).to(torch.int64)
        ).squeeze()
        self.edits += batch_edits.sum()

    def _get_length(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = tensor == special.END_IDX
        found = mask.any(dim=-1)  # First END_IDX.
        length = mask.to(torch.int32).argmax(dim=-1)
        # If END_IDX isn't found, uses the full length.
        return torch.where(
            found, length, torch.tensor(tensor.size(-1), device=self.device)
        )

    def compute(self) -> torch.Tensor:
        if self.length == 0:
            return torch.tensor(0.0, device=self.device)
        return self.edits.float() / self.length.float()
