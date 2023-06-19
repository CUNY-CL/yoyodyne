import pytest

from yoyodyne import models


@pytest.mark.parametrize(
    ["arch", "include_features", "expected_cls"],
    [
        ("attentive_lstm", True, models.AttentiveLSTMEncoderDecoder),
        ("attentive_lstm", False, models.AttentiveLSTMEncoderDecoder),
        ("lstm", True, models.LSTMEncoderDecoder),
        ("lstm", False, models.LSTMEncoderDecoder),
        (
            "pointer_generator_lstm",
            True,
            models.PointerGeneratorLSTMEncoderDecoder,
        ),
        (
            "pointer_generator_lstm",
            False,
            models.PointerGeneratorLSTMEncoderDecoder,
        ),
        ("transducer", True, models.TransducerEncoderDecoder),
        ("transducer", False, models.TransducerEncoderDecoder),
        ("transformer", True, models.TransformerEncoderDecoder),
        ("transformer", False, models.TransformerEncoderDecoder),
    ],
)
def test_get_model_cls(arch, include_features, expected_cls):
    model_cls = models.get_model_cls(arch, include_features)
    assert model_cls is expected_cls
