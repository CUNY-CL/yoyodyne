"""Linear module class."""

from ... import data
from . import base


class LinearEncoder(base.BaseModule):
    def forward(self, source: data.PaddedTensor) -> base.ModuleOutput:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            base.ModuleOutput.
        """
        return base.ModuleOutput(self.embed(source.padded))

    @property
    def name(self) -> str:
        return "linear"

    @property
    def output_size(self) -> int:
        return self.embedding_size
