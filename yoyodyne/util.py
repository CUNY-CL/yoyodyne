"""Utilities."""

import sys

import torch


def get_device(use_gpu: bool) -> torch.device:
    """Uses GPU if requested and available, defaulting to CPU.

    Args:
        use_gpu (bool).

    Returns:
        torch.device.
    """
    return torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    )


def log_info(msg: str) -> None:
    """Logs msg to sys.stderr.

    We can additionally consider logging to a file, or getting a handle to
    the PL logger.

    Args:
        msg (str): the message to log.
    """
    print(msg, file=sys.stderr)


def seed(a: int) -> None:
    """Seeds global random number generator(s) for reproducibility.

    This currently just seeds the `torch` RNG; it may also prove necessary
    to seed the `numpy` or `random` RNGs as well, and this may still prove
    insufficient on certain platforms.

    Args:
        a (int).
    """
    torch.manual_seed(a)
