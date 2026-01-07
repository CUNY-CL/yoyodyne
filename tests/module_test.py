"""Tests module (i.e., layer) instantiation.

The theory here is that a lot of breakages break module __init__.
"""

import unittest

from parameterized import parameterized

from yoyodyne import defaults
from yoyodyne.models import modules


class ModuleTest(unittest.TestCase):

    def test_attention(self):
        module = modules.Attention()
        self.assertIsInstance(module, modules.Attention)

    @parameterized.expand(
        [
            modules.ContextHardAttentionGRUDecoder,
            modules.ContextHardAttentionLSTMDecoder,
        ],
    )
    def test_context_hard_attention_rnn_decoders(self, mod):
        module = mod()
        self.assertIsInstance(module, mod)

    def test_generation_probability(self):
        module = modules.GenerationProbability()
        self.assertIsInstance(module, modules.GenerationProbability)

    @parameterized.expand(
        [
            modules.HardAttentionGRUDecoder,
            modules.HardAttentionLSTMDecoder,
        ],
    )
    def test_hard_attention_rnn_decoders(self, mod):
        module = mod()
        self.assertIsInstance(module, mod)

    def test_linear_encoder(self):
        module = modules.LinearEncoder()
        self.assertIsInstance(module, modules.LinearEncoder)

    def test_positional_encoding(self):
        module = modules.PositionalEncoding()
        self.assertIsInstance(module, modules.PositionalEncoding)

    @parameterized.expand([modules.GRUDecoder, modules.LSTMDecoder])
    def test_rnn_decoder(self, mod):
        module = mod()
        self.assertIsInstance(module, mod)

    @parameterized.expand([modules.GRUEncoder, modules.LSTMEncoder])
    def test_rnn_encoder(self, mod):
        module = mod()
        self.assertIsInstance(module, mod)

    @parameterized.expand(
        [modules.SoftAttentionGRUDecoder, modules.SoftAttentionLSTMDecoder]
    )
    def test_soft_attention_rnn_decoder(self, mod):
        module = mod()
        self.assertIsInstance(module, mod)

    @parameterized.expand(
        [
         modules.PointerGeneratorTransformerDecoder,
         modules.TransformerDecoder,
        ]
    )
    def test_transformer_decoder(self, mod):
        module = mod()
        self.assertIsInstance(module, mod)
        
    @parameterized.expand(
        [
         modules.FeatureInvariantTransformerEncoder,
         modules.TransformerEncoder,
        ]
    )
    def test_transformer_encoder(self, mod):
        module = mod()
        self.assertIsInstance(module, mod)
    

if __name__ == "__main__":
    unittest.main()
