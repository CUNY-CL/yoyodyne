"""Model classes and lookup function."""

import argparse

from .. import util
from .base import BaseEncoderDecoder
from .lstm import LSTMEncoderDecoder, LSTMEncoderDecoderAttention
from .pointer_generator import (
    PointerGeneratorLSTMEncoderDecoderFeatures,
    PointerGeneratorLSTMEncoderDecoderNoFeatures,
)
from .transducer import TransducerFeatures, TransducerNoFeatures
from .transformer import (
    FeatureInvariantTransformerEncoderDecoder,
    TransformerEncoderDecoder,
)


def get_model_cls(
    arch: str, attention: bool, has_features: bool
) -> BaseEncoderDecoder:
    """Model factory.

    Args:
        arch (str).
        attention (bool).
        has_features (bool).

    Raises:
        NotImplementedError.

    Returns:
        BaseEncoderDecoder.
    """
    model_fac = {
        # fmt: off
        "feature_invariant_transformer":
            FeatureInvariantTransformerEncoderDecoder,
        "pointer_generator_lstm":
            PointerGeneratorLSTMEncoderDecoderFeatures
            if has_features
            else PointerGeneratorLSTMEncoderDecoderNoFeatures,
        # fmt: on
        "transducer": TransducerFeatures
        if has_features
        else TransducerNoFeatures,
        "transformer": TransformerEncoderDecoder,
        "lstm": LSTMEncoderDecoderAttention
        if attention
        else LSTMEncoderDecoder,
    }
    try:
        model_cls = model_fac[arch]
        util.log_info(f"Model: {model_cls.__name__}")
        return model_cls
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
    return get_model_cls(args.arch, args.attention, args.features_col != 0)


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds model options to an argument parser.

    We only add the ones needed to look up the model class itself, with
    more specific arguments specified in ../train.py.

    Args:
        parser (argparse.ArgumentParser).
    """
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
        help="Uses attention (LSTM architecture only). Default: %(default)s.",
    )
    parser.add_argument(
        "--no_attention", action="store_false", dest="attention"
    )
