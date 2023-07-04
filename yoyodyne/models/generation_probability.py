"""Generation probability."""

import math

import torch
from torch import nn


class GenerationProbability(nn.Module):
    """Calculates the generation probability for a pointer generator."""

    stdev = 1 / math.sqrt(100)

    w_attention: nn.Linear
    w_hs: nn.Linear
    w_inp: nn.Linear
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
        self.w_attention = nn.Linear(attention_size, 1, bias=False)
        self.w_hs = nn.Linear(hidden_size, 1, bias=False)
        self.w_inp = nn.Linear(embedding_size, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-self.stdev, self.stdev)

    def forward(
        self,
        h_attention: torch.Tensor,
        decoder_hs: torch.Tensor,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        """Computes generation probability.

        The formula is:

            w_h * ATTN_t + w_s * HIDDEN_t + w_y * Y_{t-1} + b

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
        p_gen = self.w_attention(h_attention) + self.w_hs(decoder_hs)
        p_gen.add_(
            self.w_inp(inp) + self.bias.expand(h_attention.size(0), 1, -1)
        )
        # -> B.
        p_gen = torch.sigmoid(p_gen.squeeze(1))
        return p_gen
