"""Yoyodyne: small-vocabulary sequence-to-sequence generation."""

import warnings

import numpy
from torch import serialization

# Registers numpy objects as safe.
serialization.add_safe_globals([numpy._core.multiarray.scalar])

# Silences some stupid warnings.
warnings.filterwarnings("ignore", r"(?s).*both args and command line.*")
warnings.filterwarnings("ignore", r"(?s).*need to be provided during.*")
warnings.filterwarnings("ignore", r"(?s).*a wandb run already in progress.*")
warnings.filterwarnings("ignore", r"(?s).*`tensorboardX` has been removed.*")
warnings.filterwarnings("ignore", r"(?s).*does not have many workers.*")
warnings.filterwarnings("ignore", r"(?s).*Couldn't infer the batch indices.*")
warnings.filterwarnings("ignore", r"(?s).*dropout after all but last.*")
warnings.filterwarnings("ignore", r"(?s).*smaller than the logging interval.*")
warnings.filterwarnings("ignore", r"(?s).*does not have a deterministic.*")
warnings.filterwarnings("ignore", r"(?s).*Unable to serialize instance.*")
