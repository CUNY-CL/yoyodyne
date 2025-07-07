"""Utilities."""

import argparse
import os
import sys

from typing import Any, Optional

import torch

from . import special


class UniqueAddAction(argparse.Action):
    """Custom action that enforces uniqueness using a set."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        getattr(namespace, self.dest).add(values)


def log_info(msg: str) -> None:
    """Logs msg to sys.stderr.

    We can additionally consider logging to a file, or getting a handle to the
    PL logger.

    Args:
        msg (str): the message to log.
    """
    print(msg, file=sys.stderr)


def log_arguments(args: argparse.Namespace) -> None:
    """Logs non-null arguments via log_info.

    Args:
        args (argparse.Namespace).
    """
    log_info("Arguments:")
    for arg, val in vars(args).items():
        if val is None:
            continue
        log_info(f"\t{arg}: {val!r}")


def mkpath(path: str) -> None:
    """Creates subdirectories for a path if they do not already exist.

    Args:
        path (str).
    """
    dirname = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirname, exist_ok=True)


def pad_tensor_after_end(
    predictions: torch.Tensor,
) -> torch.Tensor:
    """Replaces everything after an END token with PADs.

    Cuts off tensors at the first END, and replaces the rest of the
    predictions with PAD_IDX, as these can be erroneously decoded while the
    rest of the batch is finishing decoding.

    Args:
        predictions (torch.Tensor): prediction tensor.

    Returns:
        torch.Tensor: finalized predictions.
    """
    # Not necessary if batch size is 1.
    if predictions.size(0) == 1:
        return predictions
    for i, prediction in enumerate(predictions):
        # Gets first instance of END.
        end = (prediction == special.END_IDX).nonzero(as_tuple=False)
        if len(end) > 0 and end[0].item() < len(prediction):
            # If an END was decoded and it is not the last one in the
            # sequence.
            end = end[0]
        else:
            # Leaves predictions[i] alone.
            continue
        # Hack in case the first prediction is END. In this case
        # torch.split will result in an error, so we change these 0's to
        # 1's, which will make the entire sequence END as intended.
        end[end == 0] = 1
        symbols, *_ = torch.split(prediction, end)
        # Replaces everything after with PAD, to replace erroneous decoding
        # While waiting on the entire batch to finish.
        pads = (
            torch.ones(len(prediction) - len(symbols), device=symbols.device)
            * special.PAD_IDX
        )
        pads[0] = special.END_IDX
        # Makes an in-place update to an inference tensor.
        with torch.inference_mode():
            predictions[i] = torch.cat((symbols, pads))
    return predictions
