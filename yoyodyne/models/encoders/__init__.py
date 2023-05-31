import argparse

from .base_encoder import BaseEncoder
from .transformer import TransformerEncoder, TransformerDecoder, FeatureInvariantTransformerEncoder
from .lstm import LSTMEncoder, LSTMDecoder, LSTMAttentiveDecoder
from .pointer_generator import PointerGeneratorDecoder
from .linear import LinearEncoder

from ... import util

def get_encoder_cls(arch: str) -> BaseEncoder:
    """Model factory.

    Args:
        arch (str).
        has_features (bool).

    Raises:
        NotImplementedError.

    Returns:
        BaseEncoder.
    """
    model_fac = {
        "transformer": TransformerEncoder,
        "feature_invariant_transformer": FeatureInvariantTransformerEncoder,
        'lstm': LSTMEncoder,
        'linear': LinearEncoder,
        'attentive_lstm': LSTMAttentiveDecoder
    }
    try:
        model_cls = model_fac[arch]
        util.log_info(f"Model: {model_cls.__name__}")
        return model_cls
    except KeyError:
        raise NotImplementedError(f"Architecture {arch} not found")


def get_encoder_cls_from_argparse_args(
    args: argparse.Namespace,
) -> BaseEncoder:
    """Creates a model from CLI arguments.

    Args:
        args (argparse.Namespace).

    Raises:
        NotImplementedError.

    Returns:
        BaseEncoderDecoder.
    """
    return get_encoder_cls(args.arch, args.features_col != 0)


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds model options to an argument parser.

    We only add the ones needed to look up the model class itself, with
    more specific arguments specified in ../train.py.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--source_encoder_arch",
        choices=[
            "transformer",
            "feature_invariant_transformer",
            "lstm",
        ],
        default="attentive_lstm",
        help="Model architecture to use",
    )
    parser.add_argument(
    "--feature_encoder_arch",
    choices=[
        "transformer",
    ],
    default=None,
    help="Model architecture to use for feature encoding",
)