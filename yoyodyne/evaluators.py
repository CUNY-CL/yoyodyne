"""Evaluators."""

import torch
from torch.nn import functional


class Error(Exception):
    pass


class Evaluator:
    """Evaluates predictions."""

    def val_accuracy(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> float:
        """Computes the exact word match accuracy.

        Args:
            predictions (torch.Tensor): B x vocab_size x seq_len.
            golds (torch.Tensor): B x seq_len x 1.
            end_idx (int): end of sequence index.
            pad_idx (int): padding index.

        Returns:
            float: exact accuracy.
        """
        if predictions.size(0) != golds.size(0):
            raise Error(
                f"Preds batch size ({predictions.size(0)}) and "
                f"golds batch size ({golds.size(0)} do not match"
            )
        # Gets the max value at each dim2 in predictions.
        _, predictions = torch.max(predictions, dim=2)
        # Finalizes the predictions.
        predictions = self.finalize_predictions(predictions, end_idx, pad_idx)
        return self.accuracy(predictions, golds, pad_idx)

    @staticmethod
    def accuracy(
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> float:
        if predictions.size(1) > golds.size(1):
            predictions = predictions[:, : golds.size(1)]
        elif predictions.size(1) < golds.size(1):
            num_pads = (0, golds.size(1) - predictions.size(1))
            predictions = functional.pad(
                predictions, num_pads, "constant", pad_idx
            )
        # Gets the count of exactly matching tensors in the batch.
        # -> B x max_seq_len.
        corr_count = torch.where(
            (predictions.to(golds.device) == golds).all(dim=1)
        )[0].size()[0]
        # Gets the batch size (total_count).
        total_count = predictions.size(0)
        return corr_count / total_count

    @staticmethod
    def finalize_predictions(
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
        # Not necessary if batch size is 1.
        if predictions.size(0) == 1:
            return predictions
        for i, prediction in enumerate(predictions):
            # Gets first instance of EOS.
            eos = (prediction == end_idx).nonzero(as_tuple=False)
            if len(eos) > 0 and eos[0].item() < len(prediction):
                # If an EOS was decoded and it is not the last one in the
                # sequence.
                eos = eos[0]
            else:
                # Leaves predictions[i] alone.
                continue
            # Hack in case the first prediction is EOS. In this case
            # torch.split will result in an error, so we change these 0's to
            # 1's, which will make the entire sequence EOS as intended.
            eos[eos == 0] = 1
            symbols, *_ = torch.split(prediction, eos)
            # Replaces everything after with PAD, to replace erroneous decoding
            # While waiting on the entire batch to finish.
            pads = (
                torch.ones(
                    len(prediction) - len(symbols), device=symbols.device
                )
                * pad_idx
            )
            pads[0] = end_idx
            # Makes an in-place update to an inference tensor.
            with torch.inference_mode():
                predictions[i] = torch.cat((symbols, pads))
        return predictions
