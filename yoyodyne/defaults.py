"""Default values for flags and modules."""

# All elements should be styled as CONSTANTS.

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
ATTENTION_HEADS = 4
BIDIRECTIONAL = True
DECODER_LAYERS = 1
EMBEDDING_SIZE = 128
ENCODER_LAYERS = 1
HIDDEN_SIZE = 512
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128

# Training arguments.
BATCH_SIZE = 32
BETA1 = 0.9
BETA2 = 0.999
DROPOUT = 0.2
LEARNING_RATE = 0.001
ORACLE_EM_EPOCHS = 5
ORACLE_FACTOR = 1
OPTIMIZER = "adam"
SAVE_TOP_K = 1
WANDB = False

# Decoding arguments.
BEAM_WIDTH = 1
