"""Model classes and lookup function."""

import argparse

from .base import BaseEncoderDecoder
from .lstm import AttentiveLSTMEncoderDecoder, LSTMEncoderDecoder
from .pointer_generator import (
    PointerGeneratorLSTMEncoderDecoder,
    PointerGeneratorTransformerEncoderDecoder,
)
from .transducer import TransducerEncoderDecoder
from .transformer import TransformerEncoderDecoder


def get_model_cls(arch: str) -> BaseEncoderDecoder:
    """Model factory.

    Args:
        arch (str).
        has_features (bool).

    Raises:
        NotImplementedError.

    Returns:
        BaseEncoderDecoder.
    """
    model_fac = {
        "attentive_lstm": AttentiveLSTMEncoderDecoder,
        "lstm": LSTMEncoderDecoder,
        "pointer_generator_lstm": PointerGeneratorLSTMEncoderDecoder,
        "pointer_generator_transformer": PointerGeneratorTransformerEncoderDecoder,  # noqa: 501
        "transducer": TransducerEncoderDecoder,
        "transformer": TransformerEncoderDecoder,
    }
    try:
        return model_fac[arch]
    except KeyError:
        raise NotImplementedError(f"Architecture {arch} not found")


def get_model_cls_from_argparse_args(
    args: argparse.Namespace,
) -> BaseEncoderDecoder:
    """Creates a model from CLI arguments.

    Args:
        args (argparse.Namespace).

    Raises:
        NotImplementedError.

    Returns:
        BaseEncoderDecoder.
    """
    return get_model_cls(args.arch)


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds model options to an argument parser.

    We only add the ones needed to look up the model class itself, with
    more specific arguments specified in train.py.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--arch",
        choices=[
            "attentive_lstm",
            "lstm",
            "pointer_generator_lstm",
            "pointer_generator_transformer",
            "transducer",
            "transformer",
        ],
        default="attentive_lstm",
        help="Model architecture. Default: %(default)s.",
    )
