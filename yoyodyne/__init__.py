"""Yoyodyne: small-vocabulary sequence-to-sequence generation."""

import warnings

import numpy
from torch import serialization

# Registers numpy objects as safe.
serialization.add_safe_globals([numpy._core.multiarray.scalar])

# Silences some stupid warnings.
warnings.filterwarnings(
    "ignore",
    ".*adds dropout after all but last recurrent layer.*",
)
warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")
