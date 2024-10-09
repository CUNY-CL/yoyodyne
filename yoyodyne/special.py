"""Symbol constants and helpers."""

# These are all reserved for internal use.

UNK = "<UNK>"
UNK_IDX = 0

PAD = "<P>"
PAD_IDX = 1

START = "<S>"
START_IDX = 2

END = "<E>"
END_IDX = 3

# Keep in above order.
SPECIAL = [UNK, PAD, START, END]


def isspecial(idx: int) -> bool:
    """Determines if a symbol is in the special range.

    Args:
        idx (int):

    Returns:
        bool.
    """
    return idx < len(SPECIAL)
