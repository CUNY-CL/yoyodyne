import pytest

from yoyodyne import models


@pytest.mark.parametrize(
    ["arch", "expected_cls"],
    [
        ("attentive_gru", models.AttentiveGRUModel),
        ("attentive_lstm", models.AttentiveLSTMModel),
        ("gru", models.GRUModel),
        ("hard_attention_lstm", models.HardAttentionLSTMModel),
        ("lstm", models.LSTMModel),
        (
            "pointer_generator_gru",
            models.PointerGeneratorGRUModel,
        ),
        (
            "pointer_generator_lstm",
            models.PointerGeneratorLSTMModel,
        ),
        ("transducer_gru", models.TransducerGRUModel),
        ("transducer_lstm", models.TransducerLSTMModel),
        ("transformer", models.TransformerModel),
    ],
)
def test_get_model_cls(arch, expected_cls):
    model_cls = models.get_model_cls(arch)
    assert model_cls is expected_cls
