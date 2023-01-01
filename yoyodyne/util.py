"""Utilities."""

import sys


def log_info(msg: str) -> None:
    """Logs msg to sys.stderr.

    We can additionally consider logging to a file, or getting a handle to
    the PL logger.

    Args:
        msg (str): the message to log.
    """
    print(msg, file=sys.stderr)
