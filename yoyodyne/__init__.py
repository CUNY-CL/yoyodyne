"""Yoyodyne: small-vocabulary sequence-to-sequence generation."""

import warnings

import numpy
from torch import serialization

# Registers numpy objects as safe.
serialization.add_safe_globals([numpy._core.multiarray.scalar])

# Silences some stupid warnings.
warnings.filterwarnings("ignore", ".*both args and command line arguments.*")
warnings.filterwarnings("ignore", ".*need to be provided during `Trainer`.*")
warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")
warnings.filterwarnings("ignore", ".*`tensorboardX` has been removed.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Couldn't infer the batch indices.*")
warnings.filterwarnings("ignore", ".*dropout after all but last recurrent.*")
warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")
warnings.filterwarnings("ignore", ".*does not have a deterministic.*")
