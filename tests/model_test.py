"""Tests model instantiation.

The theory here is that a lot of breakages break model __init__.
"""

import unittest

from yoyodyne import models
from yoyodyne.models import modules


class ModelTest(unittest.TestCase):

    # These are ordinarily set by CLI linking.
    VOCAB_SIZE = 32
    TARGET_VOCAB_SIZE = 32

    def test_hard_attention_gru(self):
        model = models.HardAttentionGRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.HardAttentionGRUModel)

    def test_hard_attention_lstm(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_linear_features(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LinearEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_shared_features(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=True,
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_transformer_features(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.HardAttentionLSTMModel)

    def test_hard_attention_lstm_transformer_source(self):
        model = models.HardAttentionLSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.HardAttentionLSTMModel)

    def test_pointer_generator_gru(self):
        model = models.PointerGeneratorGRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.PointerGeneratorGRUModel)

    def test_pointer_generator_lstm(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_linear_features(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LinearEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_separate_features(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LSTMEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_transformer_features(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_lstm_transformer_source(self):
        model = models.PointerGeneratorLSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.PointerGeneratorLSTMModel)

    def test_pointer_generator_transformer(self):
        model = models.PointerGeneratorTransformerModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.PointerGeneratorTransformerModel)

    def test_gru(self):
        model = models.GRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.GRUModel)

    def test_lstm(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.LSTMModel)

    def test_lstm_linear_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LinearEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.LSTMModel)

    def test_lstm_separate_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.LSTMEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.LSTMModel)

    def test_lstm_shared_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=True,
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.LSTMModel)

    def test_lstm_transformer_features(self):
        model = models.LSTMModel(
            source_encoder=modules.LSTMEncoder(),
            features_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.LSTMModel)

    def test_lstm_transformer_source(self):
        model = models.LSTMModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.LSTMModel)

    def test_soft_attention_gru(self):
        model = models.SoftAttentionGRUModel(
            source_encoder=modules.GRUEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.SoftAttentionGRUModel)

    def test_soft_attention_lstm(self):
        model = models.SoftAttentionLSTMModel(
            source_encoder=modules.LSTMEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.SoftAttentionLSTMModel)

    # TODO(#329): add transducer testing.

    def transformer(self):
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.TransformerModel)

    def transformer_lstm_features(self):
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(),
            features_encoder=modules.LSTMEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.TransformerModel)

    def transformer_shared_features(self):
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(),
            features_encoder=True,
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.TransformerModel)

    def transformer_separate_features(self):
        model = models.TransformerModel(
            source_encoder=modules.TransformerEncoder(),
            features_encoder=modules.TransformerEncoder(),
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.TransformerModel)

    def feature_invariant_transformer(self):
        model = models.TransformerModel(
            source_encoder=modules.FeatureInvariantTransformerEncoder(),
            features_encoder=True,
            target_vocab_size=self.TARGET_VOCAB_SIZE,
            vocab_size=self.VOCAB_SIZE,
        )
        self.assertIsInstance(model, models.TransformerModel)


if __name__ == "__main__":
    unittest.main()
