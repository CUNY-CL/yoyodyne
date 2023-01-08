"""Symbol constants and utilities."""

from typing import Dict, Iterable, List, Optional


# These are all reserved for internal use.

PAD = "<P>"
START = "<S>"
END = "<E>"
UNK = "<UNK>"

SPECIAL = [UNK, PAD, START, END]


class SymbolMap:
    """Tracks mapping from index to symbol and symbol to index."""

    index2symbol: List[str]
    symbol2index: Dict[str, int]

    def __init__(self, symbols: Iterable[str]):
        self._index2symbol = list(symbols)
        self._symbol2index = {c: i for i, c in enumerate(self._index2symbol)}

    def __len__(self) -> int:
        return len(self._index2symbol)

    def index(self, symbol: str, unk_idx: Optional[int] = None) -> int:
        """Looks up index by symbol."""
        return self._symbol2index.get(symbol, unk_idx)

    def symbol(self, index: int) -> str:
        """Looks up symbol by index."""
        return self._index2symbol[index]

    def pprint(self) -> str:
        """Pretty-prints the vocabulary."""
        return ", ".join(f"{c!r}" for c in self._index2symbol)
