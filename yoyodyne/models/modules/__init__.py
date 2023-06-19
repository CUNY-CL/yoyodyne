import argparse

from ... import util
from .base import BaseEncoder
from .linear import LinearEncoder
from .lstm import LSTMAttentiveDecoder, LSTMDecoder, LSTMEncoder  # noqa: F401
from .transformer import TransformerDecoder  # noqa F401
from .transformer import FeatureInvariantTransformerEncoder, TransformerEncoder


class EncoderMismatchError(Exception):
    pass


def get_encoder_cls(encoder_arch: str, model_arch: str = None) -> BaseEncoder:
    """Encoder factory.

    Looks up module class for given encoder_arch string. If not found, backs
        off to find compatible encoder for given model architecture.

    Args:
        encoder_arch (str).
        model_arch (bool, optional).

    Raises:
        NotImplementedError.

    Returns:
        BaseEncoder.
    """
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
        "transducer": TransformerEncoder,
        "transformer": TransformerEncoder,
    }
    try:
        model_cls = encoder_fac[encoder_arch]
        util.log_info(f"Model: {model_cls.__name__}")
        return model_cls
    except KeyError:
        util.log_info(
            f"""Architecture {encoder_arch} not found. Looking for encoder
                compatible with model: {model_arch}"""
        )
        try:
            model_cls = model_to_encoder_fac[model_arch]
            util.log_info(f"Model: {model_cls.__name__}")
            return model_cls
        except KeyError:
            raise NotImplementedError(
                f"Encoder compatible with {model_arch} not found"
            )


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
            "feature_invariant_transformer",
            "linear",
            "lstm",
            "transformer",
        ],
        default="attentive_lstm",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--feature_encoder_arch",
        choices=[
            "linear",
            "lstm",
            "transformer",
        ],
        default=None,
        help="Model architecture to use for feature encoding",
    )


def check_encoder_compatibility(source_encoder_cls, feature_encoder_cls):
    if feature_encoder_cls is not None and isinstance(
        source_encoder_cls, FeatureInvariantTransformerEncoder
    ):
        raise EncoderMismatchError(
            """The spcified encoder type is not compatible with a
                separate feature encoder."""
        )
