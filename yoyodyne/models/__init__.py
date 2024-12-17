"""Yoyodyne models."""

import argparse

from .. import defaults

# These are used to call the `add_argparse_args` functions.
from . import base as _base
from . import expert as _expert
from . import hard_attention as _hard_attention
from . import modules as _modules
from . import rnn as _rnn
from . import transformer as _transformer

# These are used to export the models themselves.
from .base import BaseModel
from .hard_attention import HardAttentionGRUModel
from .hard_attention import HardAttentionLSTMModel
from .pointer_generator import PointerGeneratorGRUModel
from .pointer_generator import PointerGeneratorLSTMModel
from .pointer_generator import PointerGeneratorTransformerModel
from .rnn import AttentiveGRUModel
from .rnn import AttentiveLSTMModel
from .rnn import GRUModel
from .rnn import LSTMModel
from .transducer import TransducerGRUModel
from .transducer import TransducerLSTMModel
from .transformer import TransformerModel


_model_fac = {
    "attentive_gru": AttentiveGRUModel,
    "attentive_lstm": AttentiveLSTMModel,
    "gru": GRUModel,
    "hard_attention_gru": HardAttentionGRUModel,
    "hard_attention_lstm": HardAttentionLSTMModel,
    "lstm": LSTMModel,
    "pointer_generator_gru": PointerGeneratorGRUModel,
    "pointer_generator_lstm": PointerGeneratorLSTMModel,
    "pointer_generator_transformer": PointerGeneratorTransformerModel,
    "transducer_gru": TransducerGRUModel,
    "transducer_lstm": TransducerLSTMModel,
    "transformer": TransformerModel,
}


def get_model_cls(arch: str) -> BaseModel:
    """Model factory.

    Args:
        arch (str).
        has_features (bool).

    Raises:
        NotImplementedError: Architecture not found.

    Returns:
        BaseModel.
    """
    try:
        return _model_fac[arch]
    except KeyError:
        raise NotImplementedError(f"Architecture {arch} not found")


def get_model_cls_from_argparse_args(
    args: argparse.Namespace,
) -> BaseModel:
    """Creates a model from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        BaseModel.
    """
    return get_model_cls(args.arch)


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds model options to an argument parser.

    We only add the ones needed to look up the model class itself, with
    more specific arguments specified in train.py.

    Args:
        parser (argparse.ArgumentParser).
    """
    _base.add_argparse_args(parser)
    _expert.add_argparse_args(parser)
    _hard_attention.add_argparse_args(parser)
    _modules.add_argparse_args(parser)
    _rnn.add_argparse_args(parser)
    _transformer.add_argparse_args(parser)
    parser.add_argument(
        "--arch",
        choices=_model_fac.keys(),
        default=defaults.ARCH,
        help="Model architecture. Default: %(default)s.",
    )
    parser.add_argument(
        "--tie_embeddings",
        action="store_true",
        default=defaults.TIE_EMBEDDINGS,
        help="Shares embeddings for the source and target vocabularies. "
        "Always enable this with pointer-generator and transducer "
        "architectures. Default: enabled.",
    )
    parser.add_argument(
        "--no_tie_embeddings",
        action="store_false",
        dest="tie_embeddings",
    )
