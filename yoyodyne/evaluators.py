"""Evaluators."""

import torch
from torch.nn import functional


class Error(Exception):
    pass


class Evaluator:
    """Evaluates predictions."""

    device: torch.device

    def __init__(self, device):
        """Initizalizes the evaluator.

        Args:
            device (torch.device).
        """
        self.device = device

    def val_accuracy(
        self,
        preds: torch.Tensor,
        golds: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> float:
        """Computes the exact word match accuracy.

        Args:
            preds (torch.Tensor): B x vocab_size x seq_len.
            golds (torch.Tensor): B x seq_len x 1.
            end_idx (int): end of sequence index.
            pad_idx (int): padding index.

        Returns:
            float: exact accuracy.
        """
        if preds.size(0) != golds.size(0):
            raise Error(
                "Preds batch size ({preds.size(0)}) and golds batch size "
                "({golds.size(0)} do not match"
            )
        # -> B x seq_len x vocab_size.
        preds = preds.transpose(1, 2)
        # Gets the max val at each dim2 in preds.
        vals, preds = torch.max(preds, dim=2)
        # Finalizes the preds.
        preds = self.finalize_preds(preds, end_idx, pad_idx, self.device)
        return self.get_accuracy(preds, golds, pad_idx)

    @staticmethod
    def get_accuracy(
        preds: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> float:
        if preds.size(1) > golds.size(1):
            preds = preds[:, : golds.size(1)]
        elif preds.size(1) < golds.size(1):
            num_pads = (0, golds.size(1) - preds.size(1))
            preds = functional.pad(preds, num_pads, "constant", pad_idx)
        # Gets the count of exactly matching tensors in the batch.
        # -> B x max_seq_len
        corr_count = torch.where((preds == golds).all(dim=1))[0].size()[0]
        # Gets the batch size (total_count).
        total_count = preds.size(0)
        return corr_count / total_count

    @staticmethod
    def finalize_preds(
        preds: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Finalizes predictions.

        Cuts off tensors at the first end_idx, and replaces the rest of the
        predictions with pad_idx, as these are erroneously decoded while the
        rest of the batch is finishing decoding.

        Args:
            preds (torch.Tensor): prediction tensor.
            end_idx (int).
            pad_idx (int).
            device (torch.device).

        Returns:
            torch.Tensor: finalized predictions.
        """
        # Not necessary if batch size is 1.
        if preds.size(0) == 1:
            return preds
        for i, x in enumerate(preds):
            assert len(x.size()) == 1
            # Gets first instance of EOS.
            EOS = (x == end_idx).nonzero(as_tuple=False)
            if len(EOS) > 0 and EOS[0].item() < len(x):
                # If an EOS was decoded and it is not the last one in the
                # sequence.
                EOS = EOS[0]
            else:
                # Leaves preds[i] alone.
                continue
            # Hack in case the first prediction is EOS. In this case,
            # torch.split will result in an error, so we change these 0's to
            # 1's, which will make the entire sequence EOS as intended.
            EOS[EOS == 0] = 1
            chars, *extras = torch.split(x, EOS)
            # Replaces everything after with PAD, to replace erroneous decoding
            # While waiting on the entire batch to finish.
            pads = torch.ones(len(x) - len(chars), device=device) * pad_idx
            pads[0] = end_idx
            preds[i] = torch.cat((chars, pads))
        return preds
