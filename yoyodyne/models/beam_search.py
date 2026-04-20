"""Beam search classes.

A Cell is a (possibly partial) hypothesis containing the decoder output,
the symbol sequence, and the hypothesis's log-likelihood. Cells can
generate their candidate extensions (in the form of new Cells) when
provided with additional decoder output; they also know when they have
reached a final state (i.e., when END has been generated).

A SingleBeam holds the Cells for one batch item and an in-progress heap.

A BatchedBeam holds one SingleBeam per batch item and exposes batched
decode helpers so that all active cells across all items can be stepped
in a single forward pass.

BatchedBeam supports two decoding styles:

* Stateful (RNN-style): collect_active gathers one symbol per active cell;
  the model steps with that symbol and a per-cell state, then fan_out_stateful
  distributes updated states and scores back.
* Stateless (transformer-style): collect_active_sequences gathers the full
  padded symbol history for every active cell; the model runs its full
  attention stack over those sequences, then fan_out_stateless distributes
  scores back without tracking any state.

Remaining limitations:

* Beam search uses Python's heap implementation; this is reasonably performant
  in cPython (it uses a C extension module where available) but there may be a
  better pure PyTorch solution.
* We hard-code the use of log-likelihoods; the addition of two log
  probabilities is equivalent to multiplying real numbers.
* Not much attention has been paid to keeping data on device.
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import heapq

import torch
from torch import nn

from .. import special
from . import modules


@dataclasses.dataclass(order=True)
class Cell:
    """A (potentially partial) hypothesis in the beam search.

    Only the score field is used for comparison. A cell is "final" once it
    has decoded the END symbol.

    Args:
        symbols (list[int], optional).
        score (float, optional).
        state (modules.RNNState, optional): RNN state, or None for stateless
            decoders like transformers.
    """

    symbols: list[int] = dataclasses.field(
        compare=False, default_factory=lambda: [special.START_IDX]
    )
    score: float = dataclasses.field(compare=True, default=0.0)
    state: modules.RNNState | None = dataclasses.field(
        compare=False, default=None
    )

    def extensions(
        self,
        scores: torch.Tensor,
        state: modules.RNNState | None = None,
    ) -> Iterator[Cell]:
        """Yields all single-symbol extensions of this cell.

        Args:
            scores (torch.Tensor): tensor of per-symbol log-probs.
            state (modules.RNNState, optional).

        Yields:
            Cell.
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
    """Min-heap wrapper that retains only the top-k cells by score.

    Args:
        max_size (int): maximum number of cells to retain.
    """

    def __init__(self, max_size: int):
        self.heap: list[Cell] = []
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.heap)

    def clear(self) -> None:
        self.heap.clear()

    def push(self, cell: Cell) -> None:
        heapq.heappush(self.heap, cell)
        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)

    def pop(self) -> Cell:
        return heapq.heappop(self.heap)

    def items(self) -> Iterator[Cell]:
        yield from self.heap


class SingleBeam:
    """Beam for a single batch item.

    Args:
        beam_width (int).
        state (modules.RNNState, optional): initial RNN state for this item.
    """

    beam_width: int
    cells: list[Cell]
    heap: Heap

    def __init__(self, beam_width: int, state: modules.RNNState | None = None):
        self.beam_width = beam_width
        self.cells = [Cell(state=state)]
        self.heap = Heap(beam_width)

    def __len__(self) -> int:
        return len(self.cells)

    def push(self, cell: Cell) -> None:
        self.heap.push(cell)

    def update(self) -> None:
        self.cells = sorted(self.heap.items(), reverse=True)
        self.heap.clear()

    @property
    def final(self) -> bool:
        return all(cell.final for cell in self.cells)

    def active_cells(self) -> list[tuple[int, Cell]]:
        """Returns (cell_index, cell) pairs for non-final cells."""
        return [(i, c) for i, c in enumerate(self.cells) if not c.final]


class BatchedBeam:
    """Collection of per-item beams for a full batch.

    Args:
        beam_width (int).
        batch_size (int).
        states (list[modules.RNNState]): one initial state per batch item.
            Pass a list of Nones for stateless decoders.
    """

    def __init__(
        self,
        beam_width: int,
        batch_size: int,
        states: list[modules.RNNState | None],
    ):
        self.beam_width = beam_width
        self.beams: list[SingleBeam] = [
            SingleBeam(beam_width, state) for state in states
        ]

    def __len__(self) -> int:
        return len(self.beams)

    @property
    def final(self) -> bool:
        return all(beam.final for beam in self.beams)

    def update(self) -> None:
        for beam in self.beams:
            beam.update()

    def collect_active(
        self,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[modules.RNNState],
        list[tuple[int, int]],
    ]:
        """Collects the last symbol and state for every active cell.

        Args:
            device (torch.device).

        Returns:
            symbols (torch.Tensor): last decoded symbol per cell of
                shape N x 1.
            item_indices (torch.Tensor): batch item index per cell of
                shape N.
            states (list[modules.RNNState]): per-cell RNN states of length N.
            index_map (list[tuple[int, int]]): (beam_idx, cell_idx) per cell
                of length N; used to route results back after the forward pass.
        """
        symbols_list: list[int] = []
        item_indices_list: list[int] = []
        states: list[modules.RNNState] = []
        index_map: list[tuple[int, int]] = []
        for beam_idx, beam in enumerate(self.beams):
            if beam.final:
                continue
            for cell_idx, cell in beam.active_cells():
                symbols_list.append(cell.symbol)
                item_indices_list.append(beam_idx)
                states.append(cell.state)
                index_map.append((beam_idx, cell_idx))
        symbols = torch.tensor(
            symbols_list, dtype=torch.long, device=device
        ).unsqueeze(dim=1)
        item_indices = torch.tensor(
            item_indices_list, dtype=torch.long, device=device
        )
        return symbols, item_indices, states, index_map

    def fan_out_stateful(
        self,
        scores_batch: torch.Tensor,
        new_states: list[modules.RNNState],
        index_map: list[tuple[int, int]],
    ) -> None:
        """Distributes scored extensions back to the appropriate beams.

        For stateful (RNN-style) decoding; passes the updated state into
        each extension cell.

        Args:
            scores_batch (torch.Tensor): log-probs per active cell of shape
                N x vocab_size.
            new_states (list[modules.RNNState]): updated states of length N.
            index_map (list[tuple[int, int]]): as returned by collect_active.
        """
        for n, (beam_idx, cell_idx) in enumerate(index_map):
            beam = self.beams[beam_idx]
            cell = beam.cells[cell_idx]
            for new_cell in cell.extensions(scores_batch[n], new_states[n]):
                beam.push(new_cell)

    def collect_active_sequences(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """Collects the full symbol history for every active cell.

        Used for stateless (transformer-style) decoding, where the model
        needs the complete symbol sequence at each step.

        Args:
            device (torch.device).

        Returns:
            sequences (torch.Tensor): padded symbol histories of shape
                N x max_seq_len.
            item_indices (torch.Tensor): batch item index per cell of shape N.
            index_map (list[tuple[int, int]]): (beam_idx, cell_idx) per cell
                of length N; used to route results back after the forward pass.
        """
        seq_list: list[torch.Tensor] = []
        item_indices_list: list[int] = []
        index_map: list[tuple[int, int]] = []
        for beam_idx, beam in enumerate(self.beams):
            if beam.final:
                continue
            for cell_idx, cell in beam.active_cells():
                seq_list.append(
                    torch.tensor(cell.symbols, dtype=torch.long, device=device)
                )
                item_indices_list.append(beam_idx)
                index_map.append((beam_idx, cell_idx))
        sequences = nn.utils.rnn.pad_sequence(
            seq_list, batch_first=True, padding_value=special.PAD_IDX
        )
        item_indices = torch.tensor(
            item_indices_list, dtype=torch.long, device=device
        )
        return sequences, item_indices, index_map

    def fan_out_stateless(
        self,
        scores_batch: torch.Tensor,
        index_map: list[tuple[int, int]],
    ) -> None:
        """Distributes scored extensions back to the appropriate beams.

        For stateless (transformer-style) decoding; no state is tracked.

        Args:
            scores_batch (torch.Tensor): log-probs per active cell of
                shape N x vocab_size.
            index_map (list[tuple[int, int]]): as returned by
                collect_active_sequences.
        """
        for n, (beam_idx, cell_idx) in enumerate(index_map):
            beam = self.beams[beam_idx]
            cell = beam.cells[cell_idx]
            for new_cell in cell.extensions(scores_batch[n]):
                beam.push(new_cell)

    def push_final_cells(self) -> None:
        """Re-enqueues already-final cells so they survive the update step."""
        for beam in self.beams:
            for cell in beam.cells:
                if cell.final:
                    beam.push(cell)

    def predictions(self, device: torch.device) -> torch.Tensor:
        """Padded tensor of the best decoded sequences.

        All cell symbol lists are padded in one pass to a uniform seq_len,
        then reshaped.

        Args:
            device (torch.device).

        Returns:
            torch.Tensor.
        """
        flat = nn.utils.rnn.pad_sequence(
            [
                torch.tensor(cell.symbols, device=device)
                for beam in self.beams
                for cell in beam.cells
            ],
            batch_first=True,
            padding_value=special.PAD_IDX,
        )
        return flat.view(len(self.beams), self.beam_width, flat.size(1))

    def scores(self, device: torch.device) -> torch.Tensor:
        """Tensor of sequence log-likelihood scores.

        Args:
            device (torch.device).

        Returns:
            torch.Tensor.
        """
        return torch.stack(
            [
                torch.tensor(
                    [cell.score for cell in beam.cells], device=device
                )
                for beam in self.beams
            ],
            dim=0,
        )
