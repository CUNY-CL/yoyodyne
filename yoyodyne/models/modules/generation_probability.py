"""Generation probability modules for pointer-generators."""

import numpy
import torch
from torch import nn


class GenerationProbability(nn.Module):
    """Generation probability for a pointer-generator.

    Args:
        embedding_size (int): embedding dimensions.
        hidden_size (int): decoder hidden state dimensions.
        attention_size (int): dimensions of combined encoder attentions.
    """

    stdev = 1 / numpy.sqrt(100)

    W_attention: nn.Linear
    W_hs: nn.Linear
    W_emb: nn.Linear
    bias: nn.Parameter

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        attention_size: int,
    ):
        super().__init__()
        self.W_emb = nn.Linear(embedding_size, 1, bias=False)
        self.W_hs = nn.Linear(hidden_size, 1, bias=False)
        self.W_attention = nn.Linear(attention_size, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-self.stdev, self.stdev)

    def forward(
        self,
        attention_context: torch.Tensor,
        decoder_hiddens: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the generation probability

        This is a function of the context vectors, decoder hidden states, and
        target embeddings, where each is first mapped to a scalar value by a
        learnable weight matrix.

        Args:
            attention_context (torch.Tensor): combined context vector over
                source and features of shape
                B x sequence_length x attention_size.
            decoder_hiddens (torch.Tensor): decoder hidden state of shape
                B x sequence_length x hidden_size.
            target_embeddings (torch.Tensor): decoder input of shape
                B x sequence_length x embedding_size.

        Returns:
            torch.Tensor: generation probability of shape B.
        """
        # -> B x sequence_length x 1.
        p_gen = self.W_attention(attention_context) + self.W_hs(
            decoder_hiddens
        )
        p_gen += self.W_emb(target_embeddings) + self.bias.expand(
            attention_context.size(0), 1, -1
        )
        # -> B x 1 x sequence_length.
        return torch.sigmoid(p_gen)
