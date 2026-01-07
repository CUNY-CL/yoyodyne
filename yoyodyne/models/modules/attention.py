"""Attention module class."""

from typing import Tuple

import torch
from torch import nn

from ... import defaults


class Attention(nn.Module):
    """Attention module.

    After:
        Luong, M.-T., Pham, H., and Manning, C. D. 2015. Effective
        approaches to attention-based neural machine translation. In
        Proceedings of the 2015 Conference on Empirical Methods in
        Natural Language Processing, pages 1412-1421.

    Args:
        encoder_outputs_size (int, optional).
        hidden_size (int, optional).
    """

    hidden_size: int
    M: nn.Linear
    V: nn.Linear

    def __init__(
        self,
        encoder_outputs_size=defaults.HIDDEN_SIZE * 2,
        hidden_size=defaults.HIDDEN_SIZE,
    ):
        super().__init__()
        # MLP to run over encoder_outputs.
        self.M = nn.Linear(encoder_outputs_size + hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoded: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the attention distribution.

        This computes attention for the encoder outputs w.r.t. the previous
        decoder hidden state.

        Args:
            encoded (torch.Tensor): outputs from the encoder
                of shape B x seq_len x encoder_dim.
            hidden (torch.Tensor): hidden states from decode of shape
                B x decoder_dim.
            mask (torch.Tensor): encoder mask of shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: weights for the encoded states
                and the weighted sum of encoder representations.
        """
        # Gets last hidden layer.
        hidden = hidden[:, -1, :].unsqueeze(1)
        # Repeats hidden to be copied for each encoder output of shape
        # B  x seq_len x decoder_dim.
        hidden = hidden.repeat(1, encoded.size(1), 1)
        # Gets the scores of each time step in the output.
        attention_scores = self._score(encoded, hidden)
        # Masks the scores with -inf at each padded character so that softmax
        # computes a 0 towards the distribution for that cell.
        attention_scores.data.masked_fill_(mask, defaults.NEG_INF)
        # -> B x 1 x seq_len
        weights = nn.functional.softmax(attention_scores, dim=1).unsqueeze(1)
        # -> B x 1 x decoder_dim
        weighted = torch.bmm(weights, encoded)
        return weighted, weights

    def _score(
        self, encoded: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Computes the scores with concat attention.

        Args:
            encoded (torch.Tensor): encoded timesteps from the encoder.
            hidden (torch.Tensor): decoder hidden state repeated to match
                encoder dim.

        Returns:
            scores torch.Tensor: weight for each encoded representation of
                shape B x seq_len.
        """
        # -> B x seq_len x encoder_dim + hidden_dim.
        concat = torch.cat((encoded, hidden), dim=2)
        # V * feed-forward with tanh.
        # -> B x seq_len x hidden_size.
        m = self.M(concat)
        # -> B x seq_len.
        return self.V(torch.tanh(m)).squeeze(2)
