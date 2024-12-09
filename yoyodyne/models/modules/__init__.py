"""Yoyodyne modules."""

import argparse

from .base import BaseModule
from .hard_attention import ContextHardAttentionGRUDecoder  # noqa: F401
from .hard_attention import ContextHardAttentionLSTMDecoder  # noqa: F401
from .hard_attention import HardAttentionGRUDecoder  # noqa: F401
from .hard_attention import HardAttentionLSTMDecoder  # noqa: F401
from .linear import LinearEncoder
from .rnn import AttentiveGRUDecoder  # noqa: F401
from .rnn import AttentiveLSTMDecoder  # noqa: F401
from .rnn import GRUDecoder  # noqa: F401
from .rnn import GRUEncoder
from .rnn import LSTMDecoder  # noqa: F401
from .rnn import LSTMEncoder
from .transformer import TransformerDecoder  # noqa: F401
from .transformer import FeatureInvariantTransformerEncoder
from .transformer import TransformerEncoder


class Error(Exception):
    pass


class EncoderMismatchError(Error):
    pass


_encoder_fac = {
    "feature_invariant_transformer": FeatureInvariantTransformerEncoder,
    "gru": GRUEncoder,
    "linear": LinearEncoder,
    "lstm": LSTMEncoder,
    "transformer": TransformerEncoder,
}
_model_to_encoder_fac = {
    "attentive_gru": GRUEncoder,
    "attentive_lstm": LSTMEncoder,
    "gru": GRUEncoder,
    "hard_attention_gru": GRUEncoder,
    "hard_attention_lstm": LSTMEncoder,
    "lstm": LSTMEncoder,
    "pointer_generator_gru": GRUEncoder,
    "pointer_generator_lstm": LSTMEncoder,
    "pointer_generator_transformer": TransformerEncoder,
    "transducer_gru": GRUEncoder,
    "transducer_lstm": LSTMEncoder,
    "transformer": TransformerEncoder,
}


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
        raise Error("Pass either a valid encoder or model arch string")
    if encoder_arch is None:
        try:
            return _model_to_encoder_fac[model_arch]
        except KeyError:
            raise NotImplementedError(
                f"Encoder compatible with {model_arch} not found"
            )
    else:
        try:
            return _encoder_fac[encoder_arch]
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
        choices=_encoder_fac.keys(),
        help="Model architecture to use for the source encoder.",
    )
    parser.add_argument(
        "--features_encoder_arch",
        choices=["gru", "linear", "lstm", "transformer"],
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
