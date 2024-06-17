import pytest

from yoyodyne import models


@pytest.mark.parametrize(
    ["arch", "expected_cls"],
    [
        ("attentive_lstm", models.AttentiveLSTMEncoderDecoder),
        ("hard_attention_lstm", models.HardAttentionLSTM),
        ("lstm", models.LSTMEncoderDecoder),
        (
            "pointer_generator_lstm",
            models.PointerGeneratorLSTMEncoderDecoder,
        ),
        ("transducer", models.TransducerEncoderDecoder),
        ("transformer", models.TransformerEncoderDecoder),
    ],
)
def test_get_model_cls(arch, expected_cls):
    model_cls = models.get_model_cls(arch)
    assert model_cls is expected_cls
