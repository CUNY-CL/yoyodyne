"""Beam search classes.

A Cell is a (possibly partial) hypothesis containing the decoder output,
the symbol sequence, and the hypothesis's log-likelihood. Cells can
generate their candidate extensions (in the form of new Cells) when
provided with additional decoder output; they also know when they have reached
a final state (i.e., when END has been generated).

A Beam holds a collection of Cells and an in-progress heap.

Current limitations:

* Beam search uses Python's heap implementation; this is reasonably performant
  in cPython (it uses a C extension module where available) but there may be a
  better pure PyTorch solution.
* Beam search assumes a batch size of 1; it is not clear how to extend it to
  larger batches.
* We hard-code the use of log-likelihoods; the addition of two log
  probabilities is equivalent to multiplying real numbers.
* Not much attention has been paid to keeping data on device.

See rnn.py for sample usage.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import Iterator, List, Optional

import torch
from torch import nn

from .. import special
from . import modules


@dataclasses.dataclass(order=True)
class Cell:
    """Represents a (potentially partial) hypotheses in the beam search.

    Only the log-likelihood field is used for comparison.

    A cell is "final" once it has decoded the END symbol.

    Args:
        symbols (List[int], optional).
        score (float, optional).
        state (modules.RNNState, optional): RNN state, or None for stateless
            decoders like transformers.
    """

    symbols: List[int] = dataclasses.field(
        compare=False, default_factory=lambda: [special.START_IDX]
    )
    score: float = dataclasses.field(compare=True, default=0.0)
    state: Optional[modules.RNNState] = dataclasses.field(
        compare=False, default=None
    )

    def extensions(
        self,
        scores: torch.Tensor,
        state: Optional[modules.RNNState] = None,
    ) -> Iterator[Cell]:
        """Generates extension cells.

        Args:
            scores (torch.Tensor):
            state (modules.RNNState, optional).

        Yields:
            Cell: all single-symbol extensions of the current cell.
        """
        for symbol, score in enumerate(scores):
            yield Cell(
                self.symbols + [symbol], self.score + score.item(), state
            )

    @property
    def symbol(self) -> int:
        return self.symbols[-1]

    @property
    def final(self) -> bool:
        return self.symbols[-1] == special.END_IDX


class Heap:
    """Wrapper round Python's heap implementation.

    Args:
        heap (List[Cell], optional).
    """

    heap: List[Cell]

    def __init__(self):
        self.heap = []

    def __len__(self) -> int:
        return len(self.heap)

    def clear(self) -> None:
        self.heap.clear()

    def push(self, cell: Cell) -> None:
        heapq.heappush(self.heap, cell)

    def pop(self) -> Cell:
        return heapq.heappop(self.heap)

    def items(self) -> Iterator[Cell]:
        yield from self.heap


class Beam:
    """The beam.

    This stores stores the current set of beam cells and an in-progress heap of
    the next set separately.

    A beam is "final" once every cell has decoded the END symbol.

    Args:
        beam_width (int).
        state (modules.RNNState, optional): RNN state, or None for stateless
            decoders like transformers.
    """

    beam_width: int
    # Current cells.
    cells: List[Cell]
    # Heap of the next set of cells.
    heap: Heap

    def __init__(self, beam_width, state: Optional[modules.RNNState] = None):
        self.beam_width = beam_width
        self.cells = [Cell(state=state)]
        self.heap = Heap()

    def __len__(self) -> int:
        return len(self.cells)

    def push(self, cell: Cell) -> None:
        """Inserts the cell into the heap, maintaining the specified beam size.

        Args:
            cell (Cell).
        """
        self.heap.push(cell)
        if len(self.heap) > self.beam_width:
            self.heap.pop()

    def update(self) -> None:
        """Replaces the current cells and clears the heap."""
        self.cells = sorted(self.heap.items(), reverse=True)
        self.heap.clear()

    @property
    def final(self) -> bool:
        return all(cell.final for cell in self.cells)

    def predictions(self, device: torch.device) -> torch.Tensor:
        """Converts the best sequences into a padded tensor of predictions.

        This implementation assumes batch size is 1.

        Args:
            device (torch.device): the device to move the data to.

        Returns:
            torch.Tensor: a B x beam_width x seq_len tensor of predictions.
        """
        return nn.utils.rnn.pad_sequence(
            [torch.tensor(cell.symbols, device=device) for cell in self.cells],
            batch_first=True,
            padding_value=special.PAD_IDX,
        ).unsqueeze(0)

    def scores(self, device: torch.device) -> torch.Tensor:
        """Converts the sequence scores into tensors.

        This implementation assumes batch size is 1.

        Args:
            device (torch.device): the device to move the data to.

        Returns:
            torch.Tensor: a B x beam_width tensor of log-likelihoods.
        """
        return torch.tensor(
            [cell.score for cell in self.cells], device=device
        ).unsqueeze(0)
