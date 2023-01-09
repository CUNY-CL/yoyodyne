"""Attention module class."""

import math
from typing import Tuple

import torch
from torch import nn


class Attention(nn.Module):
    """Attention module.

    After:
        Luong, M.-T., Pham, H., and Manning, C. D. 2015. Effective
        approaches to attention-based neural machine translation. In
        Proceedings of the 2015 Conference on Empirical Methods in
        Natural Language Processing, pages 1412-1421.
    """

    hidden_size: int
    M: nn.Linear
    V: nn.Linear

    def __init__(self, encoder_outputs_size, hidden_size):
        """Initializes the attention module.

        Args:
            encoder_outputs_size (int).
            hidden_size (int).
        """
        super().__init__()
        # MLP to run over encoder_outputs.
        self.M = nn.Linear(encoder_outputs_size + hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the attention distribution for the encoder outputs
            w.r.t. the previous decoder hidden state.

        Args:
            hidden (torch.Tensor): hidden states from decode of shape
                B x decoder_dim.
            encoder_outputs (torch.Tensor): outputs from the encoder
                of shape B x seq_len x encoder_dim.
            mask (torch.Tensor): encoder mask of shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: weights for the encoded states
                and the weighted sum of encoder representations.
        """
        # Gets last hidden layer.
        hidden = hidden[:, -1, :].unsqueeze(1)
        # Repeats hidden to be copied for each encoder output of shape
        # B  x seq_len x decoder_dim.
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)
        # Gets the scores of each time step in the output.
        attention_scores = self.score(hidden, encoder_outputs)
        # Masks the scores with -inf at each padded character so that softmax
        # computes a 0 towards the distribution for that cell.
        attention_scores.data.masked_fill_(mask, -math.inf)
        # -> B x 1 x seq_len
        weights = nn.functional.softmax(attention_scores, dim=1).unsqueeze(1)
        # -> B x 1 x decoder_dim
        weighted = torch.bmm(weights, encoder_outputs)
        return weighted, weights

    def score(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Computes the scores with concat attention.

        Args:
            hidden (torch.Tensor): decoder hidden state repeated to match
                encoder dim.
            encoder_outputs (torch.Tensor): encoded timesteps from the encoder.

        Returns:
            scores torch.Tensor: weight for each encoded representation of
                shape B x seq_len.
        """
        # -> B x seq_len x encoder_dim + hidden_dim.
        concat = torch.cat([encoder_outputs, hidden], 2)
        # V * feed forward with tanh.
        # -> B x seq_len x hidden_size
        m = self.M(concat)
        # -> B x seq_len x 1.
        scores = self.V(torch.tanh((m)))
        # -> B x seq_len.
        return scores.squeeze(2)
