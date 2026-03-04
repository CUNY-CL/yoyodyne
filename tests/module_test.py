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
            modules.ContextHardAttentionGRUDecoder,
            modules.ContextHardAttentionLSTMDecoder,
        ],
    )
    def test_context_hard_attention_rnn_decoders(self, mod):
        module = mod()
        assert isinstance(module, mod)

    def test_generation_probability(self):
        module = modules.GenerationProbability()
        assert isinstance(module, modules.GenerationProbability)

    @pytest.mark.parametrize(
        "mod",
        [
            modules.HardAttentionGRUDecoder,
            modules.HardAttentionLSTMDecoder,
        ],
    )
    def test_hard_attention_rnn_decoders(self, mod):
        module = mod()
        assert isinstance(module, mod)

    def test_linear_encoder(self):
        module = modules.LinearEncoder()
        assert isinstance(module, modules.LinearEncoder)

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

    @pytest.mark.parametrize(
        "mod",
        [modules.GRUDecoder, modules.LSTMDecoder],
    )
    def test_rnn_decoders(self, mod):
        module = mod()
        assert isinstance(module, mod)

    @pytest.mark.parametrize(
        "mod",
        [modules.GRUEncoder, modules.LSTMEncoder],
    )
    def test_rnn_encoders(self, mod):
        module = mod()
        assert isinstance(module, mod)

    @pytest.mark.parametrize(
        "mod",
        [modules.SoftAttentionGRUDecoder, modules.SoftAttentionLSTMDecoder],
    )
    def test_soft_attention_rnn_decoders(self, mod):
        module = mod()
        assert isinstance(module, mod)

    @pytest.mark.parametrize(
        "mod",
        [
            modules.TransformerDecoder,
            modules.CausalTransformerDecoder,
            modules.PointerGeneratorTransformerDecoder,
            modules.PointerGeneratorTransformerDecoder,
            modules.RotaryCausalTransformerDecoder,
            modules.RotaryPointerGeneratorTransformerDecoder,
            modules.RotaryTransformerDecoder,
        ],
    )
    def test_transformer_decoders(self, mod):
        module = mod()
        assert isinstance(module, mod)

    @pytest.mark.parametrize(
        "mod",
        [
            modules.TransformerEncoder,
            modules.FeatureInvariantTransformerEncoder,
            modules.RotaryFeatureInvariantTransformerEncoder,
            modules.RotaryTransformerEncoder,
        ],
    )
    def test_transformer_encoders(self, mod):
        module = mod()
        assert isinstance(module, mod)
