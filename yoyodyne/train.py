"""Trains a sequence-to-sequence neural network."""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks, loggers
from torch.utils import data

from . import collators, datasets, evaluators, models, predict, util


def train(seed: Optional[int]) -> str:
    """Trains a sequence-to-sequence neural network.

    Args:
        seed (int, optional).

    Returns:
        (str) path to the best model checkpoint.
    """
    pl.seed_everything(seed)



def main(args: argparse.Namespace) -> None:
    util.log_info("Arguments:")
    for argument, value in vars(args).items():
        util.log_info(f"\t{argument}: {value!r}")
    best_model_path = train(args.seed)
    util.log_info(f"Best model: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", required=True, help="Name of experiment"
    )
    # Paths.
    parser.add_argument(
        "--train_path", required=True, help="Path to input training data TSV"
    )
    parser.add_argument(
        "--dev_path", required=True, help="Path to input development data TSV"
    )
    parser.add_argument(
        "--model_dir", required=True, help="Path to output model directory"
    )
    # Data arguments.
    parser.add_argument(
        "--source_col",
        type=int,
        default=1,
        help="1-based index for source column (default: %(default)s)",
    )
    parser.add_argument(
        "--target_col",
        type=int,
        default=2,
        help="1-based index for target column (default: %(default)s)",
    )
    parser.add_argument(
        "--features_col",
        type=int,
        default=0,
        help="1-based index for features column; "
        "0 indicates the model will not use features (default: %(default)s)",
    )
    parser.add_argument(
        "--source_sep",
        type=str,
        default="",
        help="String used to split source string into symbols; "
        "an empty string indicates that each Unicode codepoint "
        "is its own symbol (default: %(default)r)",
    )
    parser.add_argument(
        "--target_sep",
        type=str,
        default="",
        help="String used to split target string into symbols; "
        "an empty string indicates that each Unicode codepoint "
        "is its own symbol (default: %(default)r)",
    )
    parser.add_argument(
        "--features_sep",
        type=str,
        default=";",
        help="String used to split features string into symbols; "
        "an empty string indicates that each Unicode codepoint "
        "is its own symbol (default: %(default)r)",
    )
    parser.add_argument(
        "--tied_vocabulary",
        action="store_true",
        default=True,
        help="Share source and target embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--no_tied_vocabulary",
        action="store_false",
        dest="tied_vocabulary",
        default=True,
    )
    # Architecture arguments.
    parser.add_argument(
        "--arch",
        choices=[
            "feature_invariant_transformer",
            "lstm",
            "pointer_generator_lstm",
            "transducer",
            "transformer",
        ],
        default="lstm",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        default=True,
        help="Uses attention (LSTM architecture only; default: %(default)s)",
    )
    parser.add_argument(
        "--no_attention", action="store_false", dest="attention"
    )
    parser.add_argument(
        "--attention_heads",
        type=int,
        default=4,
        help="Number of attention heads "
        "(transformer-backed architectures only; default: %(default)s)",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="Uses a bidirectional encoder "
        "(LSTM-backed architectures only; default: %(default)s)",
    )
    parser.add_argument(
        "--no_bidirectional", action="store_false", dest="bidirectional"
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=1,
        help="Number of decoder layers (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="Dimensionality of embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=1,
        help="Number of encoder layers (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Dimensionality of the hidden layer(s) (default: %(default)s)",
    )
    parser.add_argument(
        "--max_decode_len",
        type=int,
        default=128,
        help="Maximum decoder string length (default: %(default)s)",
    )
    parser.add_argument(
        "--max_sequence_len",
        type=int,
        default=128,
        help="Maximum sequence length (default: %(default)s)",
    )
    parser.add_argument(
        "--oracle_em_epochs",
        type=int,
        default=5,
        help="Number of EM epochs "
        "(transducer architecture only; default: %(default)s",
    )
    parser.add_argument(
        "--oracle_factor",
        type=int,
        default=1,
        help="Roll-in schedule parameter "
        "(transducer architecture only; default: %(default)s)",
    )
    parser.add_argument(
        "--sed_path",
        type=str,
        help="Path to input SED parameters (transducer architecture only)",
    )
    # Training arguments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta_1 (Adam optimizer only; default: %(default)s)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="beta_2 (Adam optimizer only; default: %(default)s)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: %(default)s)",
    )
    parser.add_argument(
        "--label_smoothing", type=float, help="Coefficient for label smoothing"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adadelta", "adam", "sgd"],
        default="adam",
        help="Optimizer (default: %(default)s)",
    )
    parser.add_argument(
        "--patience", type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        help="Number of checkpoints to save (default: %(default)s)",
    )
    parser.add_argument(
        "--scheduler", choices=["warmupinvsqrt"], help="Learning rate scheduler"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps (warmupinvsqrt scheduler only; default: %(default)s)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Use weights &^ Biases logging (log-in required)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_false",
        dest="wandb",
        help="Disable weights &^ Biases logging (log-in required)",
    )
    # Among the things this adds, the following are likely to be useful to users:
    # --accelerator
    # --check-val-every-n-epoch
    # --devices
    # --gradient-clip-val
    # --max-epochs
    # --min-epochs
    # --max-steps
    # --min-steps
    # --max-time
    pl.Trainer.add_argparse_args(parser)
    main(parser.parse_args())
