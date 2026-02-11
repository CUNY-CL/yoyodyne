"""Symbol constants and helpers."""

# These are all reserved for internal use.

PAD = "<P>"
PAD_IDX = 0

UNK = "<UNK>"
UNK_IDX = 1

START = "<S>"
START_IDX = 2
END = "<E>"
END_IDX = 3

# For the feature-invariant transformer encoder.
SOURCE = "<SOURCE>"
SOURCE_IDX = 4
FEATURES = "<FEATURES>"
FEATURES_IDX = 5

# Keep in above order.
SPECIAL = [PAD, UNK, START, END, SOURCE, FEATURES]


def isspecial(idx: int) -> bool:
    """Determines if a symbol is in the special range.

    Args:
        idx (int):

    Returns:
        bool.
    """
    return idx < len(SPECIAL)
