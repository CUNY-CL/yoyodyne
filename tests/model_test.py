"""Tests model instantiation.

The theory here is that a lot of breakages break model __init__.
"""

import os
import string

import pytest
from yoyodyne import data, models
from yoyodyne.models import modules

# These are ordinarily set by CLI linking.
VOCAB_SIZE = 32
TARGET_VOCAB_SIZE = 32

SED_PATH = os.path.join(os.path.dirname(__file__), "testdata", "toy.sed")


# Minimal index matching the toy SED's vocabulary.
@pytest.fixture
def toy_index():
    return data.Index(
        source_vocabulary=list(string.ascii_lowercase),
        target_vocabulary=list(string.ascii_lowercase),
    )


class TestModel:

    def test_causal_transformer(self):
        model = models.CausalTransformerModel(
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.CausalTransformerModel)

    def test_hard_attention_gru(self):
        model = models.HardAttentionGRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.HardAttentionGRUModel)

    def test_hard_attention_lstm(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_linear_features(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LinearEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_shared_features(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=True,
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_transformer_features(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_transformer_source(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.HardAttentionLSTMModel)

    def test_pointer_generator_gru(self):
        model = models.PointerGeneratorGRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.PointerGeneratorGRUModel)

    def test_pointer_generator_lstm(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_linear_features(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LinearEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_separate_features(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LSTMEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_transformer_features(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_transformer_source(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_transformer(self):
        model = models.PointerGeneratorTransformerModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.PointerGeneratorTransformerModel)

    def test_gru(self):
        model = models.GRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.GRUModel)

    def test_lstm(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.LSTMModel)

    def test_lstm_linear_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LinearEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.LSTMModel)

    def test_lstm_separate_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LSTMEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.LSTMModel)

    def test_lstm_shared_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=True,
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.LSTMModel)

    def test_lstm_transformer_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.LSTMModel)

    def test_lstm_transformer_source(self):
        model = models.LSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.LSTMModel)

    def test_soft_attention_gru(self):
        model = models.SoftAttentionGRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.SoftAttentionGRUModel)

    def test_soft_attention_lstm(self):
        model = models.SoftAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.SoftAttentionLSTMModel)

    def test_transducer_gru(self, toy_index):
        model = models.TransducerGRUModel(
            SED_PATH,
            index=toy_index,
            source_encoder=modules.GRUEncoder(),
        )
        assert isinstance(model, models.TransducerGRUModel)

    def test_transducer_lstm(self, toy_index):
        model = models.TransducerLSTMModel(
            SED_PATH,
            index=toy_index,
            source_encoder=modules.LSTMEncoder(),
        )
        assert isinstance(model, models.TransducerLSTMModel)

    def test_transformer(self):
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.TransformerModel)

    def test_transformer_lstm_features(self):
        hidden_size = 128
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(model_size=hidden_size),
            features_encoder=modules.LSTMEncoder(
                bidirectional=False, hidden_size=hidden_size
            ),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.TransformerModel)

    def test_transformer_shared_features(self):
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(),
            features_encoder=True,
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.TransformerModel)

    def test_transformer_separate_features(self):
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(),
            features_encoder=modules.TransformerEncoder(),
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.TransformerModel)

    def test_feature_invariant_transformer(self):
        model = models.TransformerModel(
            source_encoder=modules.FeatureInvariantTransformerEncoder(),
            features_encoder=True,
            target_vocab_size=TARGET_VOCAB_SIZE,
            vocab_size=VOCAB_SIZE,
        )
        assert isinstance(model, models.TransformerModel)
