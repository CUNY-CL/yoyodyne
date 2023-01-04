"""Generation probability."""

import math

import torch
from torch import nn


class GenerationProbability(nn.Module):
    """Calculates the generation probability for a pointer generator."""

    stdev = 1 / math.sqrt(100)

    W_attention: nn.Linear
    W_hs: nn.Linear
    W_inp: nn.Linear
    bias: nn.Parameter

    def __init__(
        self, embedding_size: int, hidden_size: int, attention_size: int
    ):
        """Initializes the generation probability operator.

        Args:
            embedding_size (int): embedding dimensions.
            hidden_size (int): decoder hidden state dimensions.
            attention_size (int): dimensions of combined encoder attentions.
        """
        super().__init__()
        self.W_attention = nn.Linear(attention_size, 1, bias=False)
        self.W_hs = nn.Linear(hidden_size, 1, bias=False)
        self.W_inp = nn.Linear(embedding_size, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-self.stdev, self.stdev)

    def forward(
        self,
        h_attention: torch.Tensor,
        decoder_hs: torch.Tensor,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        """Computes Wh * ATTN_t + Ws * HIDDEN_t + Wy * Y_{t-1} + b.

        Args:
            h_attention (torch.Tensor): combined context vector over source and
                features of shape B x 1 x attention_size.
            decoder_hs (torch.Tensor): decoder hidden state of shape
                B x 1 x hidden_size.
            inp (torch.Tensor): decoder input of shape B x 1 x embedding_size.

        Returns:
            (torch.Tensor): generation probability of shape B.
        """
        # -> B x 1 x 1.
        p_gen = self.W_attention(h_attention) + self.W_hs(decoder_hs)
        p_gen += self.W_inp(inp) + self.bias.expand(h_attention.size(0), 1, -1)
        # -> B.
        p_gen = torch.sigmoid(p_gen.squeeze(1))
        return p_gen
