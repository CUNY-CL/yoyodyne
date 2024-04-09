import argparse

from .base import BaseModule
from .linear import LinearEncoder
from .lstm import LSTMAttentiveDecoder, LSTMDecoder, LSTMEncoder  # noqa: F401
from .transformer import TransformerDecoder  # noqa F401
from .transformer import FeatureInvariantTransformerEncoder, TransformerEncoder


class EncoderMismatchError(Exception):
    pass


def get_encoder_cls(
    encoder_arch: str = None, model_arch: str = None
) -> BaseModule:
    """Encoder factory.

    Looks up module class for given encoder_arch string. If not found, backs
        off to find compatible encoder for given model architecture.

    Args:
        encoder_arch (str, optional).
        model_arch (str, optional).

    Raises:
        NotImplementedError.

    Returns:
        BaseEncoder.
    """
    if not (encoder_arch or model_arch):
        raise ValueError(
            "Please pass either a valid encoder or model arch string"
        )
    encoder_fac = {
        "feature_invariant_transformer": FeatureInvariantTransformerEncoder,
        "linear": LinearEncoder,
        "lstm": LSTMEncoder,
        "transformer": TransformerEncoder,
    }
    model_to_encoder_fac = {
        "attentive_lstm": LSTMEncoder,
        "lstm": LSTMEncoder,
        "pointer_generator_lstm": LSTMEncoder,
        "pointer_generator_transformer": TransformerEncoder,
        "transducer": LSTMEncoder,
        "transformer": TransformerEncoder,
    }
    if encoder_arch is None:
        try:
            return model_to_encoder_fac[model_arch]
        except KeyError:
            raise NotImplementedError(
                f"Encoder compatible with {model_arch} not found"
            )
    else:
        try:
            return encoder_fac[encoder_arch]
        except KeyError:
            raise NotImplementedError(
                f"Encoder architecture {encoder_arch} not found"
            )


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds model options to an argument parser.

    We only add the ones needed to look up the module class itself, with
    more specific arguments specified in train.py.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--source_encoder_arch",
        choices=[
            "feature_invariant_transformer",
            "linear",
            "lstm",
            "transformer",
        ],
        default=None,
        help="Model architecture to use for the source encoder.",
    )
    parser.add_argument(
        "--features_encoder_arch",
        choices=[
            "linear",
            "lstm",
            "transformer",
        ],
        default=None,
        help="Model architecture to use for the features encoder.",
    )


def check_encoder_compatibility(source_encoder_cls, features_encoder_cls):
    if features_encoder_cls is not None and isinstance(
        source_encoder_cls, FeatureInvariantTransformerEncoder
    ):
        raise EncoderMismatchError(
            "The specified encoder type is not compatible with a separate "
            "feature encoder"
        )
