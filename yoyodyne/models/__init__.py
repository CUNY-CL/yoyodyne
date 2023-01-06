"""Model classes and lookup function."""

import pytorch_lightning as pl

from .. import util
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
) -> pl.LightningModule:
    """Model factory.

    Args:
        arch (str).
        attention (bool).
        has_features (bool).

    Raises:
        NotImplementedError.

    Returns:
        pl.LightningModule.
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
