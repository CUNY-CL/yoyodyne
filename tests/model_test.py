import pytest

from yoyodyne import models


@pytest.mark.parametrize(
    "arch, attn, include_features, expected_cls",
    [
        (
            "feature_invariant_transformer",
            True,
            True,
            models.FeatureInvariantTransformerEncoderDecoder,
        ),
        ("lstm", True, False, models.LSTMEncoderDecoderAttention),
        ("lstm", False, False, models.LSTMEncoderDecoder),
        (
            "pointer_generator_lstm",
            True,
            True,
            models.PointerGeneratorLSTMEncoderDecoderFeatures,
        ),
        (
            "pointer_generator_lstm",
            True,
            False,
            models.PointerGeneratorLSTMEncoderDecoderNoFeatures,
        ),
        ("transducer", True, True, models.TransducerFeatures),
        ("transducer", True, False, models.TransducerNoFeatures),
        ("transformer", True, True, models.TransformerEncoderDecoder),
        ("transformer", True, False, models.TransformerEncoderDecoder),
    ],
)
def test_get_model_cls(arch, attn, include_features, expected_cls):
    model_cls = models.get_model_cls(arch, attn, include_features)
    assert model_cls is expected_cls
