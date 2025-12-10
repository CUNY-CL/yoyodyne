"""Utilities."""

import logging
import os
from typing import Any, Dict, Tuple

import torch
import yaml

from . import special


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


def load_config_and_links(path: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Loads a YAML config file, parsing the LINKS field separately.

    Args:
        path:

    Returns:
        A tuple of two dictionaries: the core config file and the links.
    """
    with open(path, "r") as source:
        config = yaml.safe_load(source)
    links = config.pop("LINKS", {})
    return config, links


def recursive_insert(config: Dict[str, Any], key: str, value) -> None:
    """Recursively inserts values into a nested dictionary.

    Args:
        config: the config dictionary.
        key: a string with the arguments separated by ".".
        value: the value to insert.
    """
    *most, last = key.split(".")
    ptr = config
    for piece in most:
        try:
            ptr = ptr[piece]
        except KeyError:
            ptr[piece] = {}
            ptr = ptr[piece]
    if last in ptr:
        logging.debug(
            "Overriding configuration argument %s with W&B sweep value: %r",
            key,
            value,
        )
    ptr[last] = value
