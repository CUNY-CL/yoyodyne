"""Tests module (i.e., layer) instantiation.

The theory here is that a lot of breakages break module __init__.
"""

import pytest

from yoyodyne.models import modules


class TestModule:

    def test_attention(self):
        module = modules.Attention()
        assert isinstance(module, modules.Attention)

    @pytest.mark.parametrize(
        "mod",
        [
            modules.CausalTransformerDecoder,
            modules.ContextHardAttentionGRUDecoder,
            modules.ContextHardAttentionLSTMDecoder,
            modules.ContextHardAttentionTransformerDecoder,
            modules.GRUDecoder,
            modules.HardAttentionGRUDecoder,
            modules.HardAttentionLSTMDecoder,
            modules.HardAttentionTransformerDecoder,
            modules.LSTMDecoder,
            modules.PointerGeneratorTransformerDecoder,
            modules.RotaryCausalTransformerDecoder,
            modules.RotaryPointerGeneratorTransformerDecoder,
            modules.RotaryTransformerDecoder,
            modules.SoftAttentionGRUDecoder,
            modules.SoftAttentionLSTMDecoder,
            modules.TransformerDecoder,
        ],
    )
    def test_decoder(self, mod):
        module = mod()
        assert isinstance(module, mod)

    @pytest.mark.parametrize(
        "mod",
        [
            modules.FeatureInvariantTransformerEncoder,
            modules.GRUEncoder,
            modules.LinearEncoder,
            modules.LSTMEncoder,
            modules.RotaryFeatureInvariantTransformerEncoder,
            modules.RotaryTransformerEncoder,
            modules.TransformerEncoder,
        ],
    )
    def test_encoder(self, mod):
        module = mod()
        assert isinstance(module, mod)

    def test_generation_probability(self):
        module = modules.GenerationProbability()
        assert isinstance(module, modules.GenerationProbability)

    @pytest.mark.parametrize(
        "mod",
        [
            modules.AbsolutePositionalEncoding,
            modules.NullPositionalEncoding,
            modules.RotaryPositionalEncoding,
            modules.SinusoidalPositionalEncoding,
        ],
    )
    def test_positional_encodings(self, mod):
        module = mod()
        assert isinstance(module, mod)
