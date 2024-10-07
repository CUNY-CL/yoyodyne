"""Yoyodyne modules."""

import argparse

from .base import BaseModule
from .linear import LinearEncoder
from .rnn import RNNAttentiveDecoder, RNNDecoder, RNNEncoder  # noqa: F401
from .transformer import TransformerDecoder  # noqa F401
from .transformer import FeatureInvariantTransformerEncoder, TransformerEncoder


class Error(Exception):
    pass


class EncoderMismatchError(Error):
    pass


_encoder_fac = {
    "feature_invariant_transformer": FeatureInvariantTransformerEncoder,
    "linear": LinearEncoder,
    "rnn": RNNEncoder,
    "transformer": TransformerEncoder,
}
_model_to_encoder_fac = {
    "attentive_rnn": RNNEncoder,
    "rnn": RNNEncoder,
    "pointer_generator_rnn": RNNEncoder,
    "pointer_generator_transformer": TransformerEncoder,
    "transducer": RNNEncoder,
    "transformer": TransformerEncoder,
    "hard_attention_rnn": RNNEncoder,
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
        choices=["linear", "rnn", "transformer"],
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
