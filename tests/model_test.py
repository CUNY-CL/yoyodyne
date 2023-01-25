import pytest

from yoyodyne import models


@pytest.mark.parametrize(
    ["arch", "include_features", "expected_cls"],
    [
        ("attentive_lstm", True, models.AttentiveLSTMEncoderDecoder),
        ("attentive_lstm", False, models.AttentiveLSTMEncoderDecoder),
        (
            "feature_invariant_transformer",
            True,
            models.FeatureInvariantTransformerEncoderDecoder,
        ),
        ("lstm", True, models.LSTMEncoderDecoder),
        ("lstm", False, models.LSTMEncoderDecoder),
        (
            "pointer_generator_lstm",
            True,
            models.PointerGeneratorLSTMEncoderDecoderFeatures,
        ),
        (
            "pointer_generator_lstm",
            False,
            models.PointerGeneratorLSTMEncoderDecoderNoFeatures,
        ),
        ("transducer", True, models.TransducerFeatures),
        ("transducer", False, models.TransducerNoFeatures),
        ("transformer", True, models.TransformerEncoderDecoder),
        ("transformer", False, models.TransformerEncoderDecoder),
    ],
)
def test_get_model_cls(arch, include_features, expected_cls):
    model_cls = models.get_model_cls(arch, include_features)
    assert model_cls is expected_cls
