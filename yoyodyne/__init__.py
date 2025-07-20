"""Yoyodyne: small-vocabulary sequence-to-sequence generation.

This module just silences some uninformative warnings.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    ".*does not have many workers which may be a bottleneck.*",
)
warnings.filterwarnings(
    "ignore",
    ".*option adds dropout after all but last recurrent layer.*",
)
warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")
