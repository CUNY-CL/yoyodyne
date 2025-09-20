"""Default values for flags and modules."""

import numpy

from torch import optim

from . import schedulers

# All elements should be styled as CONSTANTS.

# Scalar constants.
EPSILON = 1e-7
LOG_EPSILON = numpy.log(EPSILON)
NEG_LOG_EPSILON = -numpy.log(EPSILON)
INF = numpy.inf
NEG_INF = -numpy.inf

# Default text encoding.
ENCODING = "utf-8"

# Data configuration arguments.
SOURCE_COL = 1
TARGET_COL = 2
FEATURES_COL = 0
SOURCE_SEP = ""
TARGET_SEP = ""
FEATURES_SEP = ";"
TIED_VOCABULARY = True

# Architecture arguments.
ARCH = "attentive_lstm"
ATTENTION_CONTEXT = 0
ATTENTION_HEADS = 4
BIDIRECTIONAL = True
EMBEDDING_SIZE = 128
ENFORCE_MONOTONIC = False
HIDDEN_SIZE = 512
LAYERS = 1
MAX_LENGTH = 128
TIE_EMBEDDINGS = True

# Training arguments.
BATCH_SIZE = 32
BETA1 = 0.9
BETA2 = 0.999
DROPOUT = 0.2
LABEL_SMOOTHING = 0.0
ORACLE_FACTOR = 1
OPTIMIZER = optim.Adam
SCHEDULER = schedulers.Dummy

# Decoding arguments.
BEAM_WIDTH = 1
TEACHER_FORCING = True

# Extra evaluation metrics.
EVAL_METRICS = set()
