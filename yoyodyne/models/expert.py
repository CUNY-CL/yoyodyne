"""Expert for character-level string transduction.

Given an input string x, a target string t, alignment index i, and a partial
prediction y, it returns optimal cost-to-go for all valid edit actions.

Also includes ActionVocabulary class for compatibility with the `maxwell`
dictionary. This class stores valid edit actions for given dataset."""

import argparse
import dataclasses
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy
from maxwell import actions, sed

from .. import data, defaults, special


class Error(Exception):
    pass


class ActionError(Error):
    pass


class AlignerError(Error):
    pass


class ActionVocabulary:
    """Manages encoding of action vocabulary for transducer training."""

    # TODO: Port more of the logic to the dataset class.
    i2w: Dict[int, actions.Edit]
    w2i: Dict[actions.Edit, int]
    start_vocab_idx: int
    target_characters: Set[Any]

    def __init__(self, index: data.indexes.Index):
        self.target_characters = set()
        self.i2w = [
            actions.Start(),
            actions.End(),
            actions.ConditionalDel(),
            actions.ConditionalCopy(),
        ]
        self.start_vocab_idx = len(self.i2w)
        self.w2i = {w: i for i, w in enumerate(self.i2w)}
        # Uses index from dataset to create action vocabulary.
        self.encode_actions([index(t) for t in index.target_vocabulary])
        # Sets unknown character decoding.
        self.encode_actions([special.UNK_IDX])
        # Adds source characters if index has tied embeddings.
        if index.tie_embeddings:
            self.encode_actions([index(s) for s in index.source_vocabulary])

    def encode(self, symb: actions.Edit) -> int:
        """Returns index referencing symbol in encoding table.

        If the symbol is not encoded, it is added to the encoding table.

        Args:
            symb (actions.Edit): edit action to add to encoding table.

        Returns:
            int: index of symbol in encoding table.
        """
        if symb in self.w2i:
            idx = self.w2i[symb]
        else:
            idx = len(self.i2w)
            self.i2w.append(symb)
            self.w2i[symb] = idx
        return idx

    def decode(self, idx: int) -> actions.Edit:
        """Returns symbol referenced by given index in encoding table.

        Args:
            idx (int): encoding index that references symbol in encoding table.

        Returns:
            actions.Edit: action in encoding table referenced by given index.
        """
        return self.i2w[idx]

    def encode_actions(self, target: Any) -> None:
        """Encodes all possible edit actions for given target.

        Args:
            target (Any): target symbol produced by encoded edits.
        """
        for c in target:
            if c in self.target_characters:
                continue
            self.encode(actions.ConditionalSub(c))
            self.encode(actions.ConditionalIns(c))
            self.target_characters.add(c)

    def encode_unseen_action(self, action: actions.Edit) -> int:
        """Encodes action unseen in training.

        Operates same as encode_action but does not expand lookup table.

        Args:
            actions (actions.Edit).

        Returns:
            int.
        """
        if action in self.w2i:
            return self.lookup(action)
        else:
            raise ActionError(f"Action {action} is out of vocabulary")

    def __len__(self) -> int:
        return len(self.i2w)

    def __repr__(self) -> str:
        return f"Vocabulary({str(self.w2i)})"

    def lookup(self, word: str) -> int:
        return self.w2i[word]

    def to_i2w(self) -> List[str]:
        return self.i2w[len(self.start_vocab_idx) :]  # noqa: E203

    @property
    def substitutions(
        self,
    ) -> List[Tuple[int, actions.ConditionalEdit]]:
        return [
            i
            for i, a in enumerate(self.i2w)
            if isinstance(a, actions.ConditionalSub)
        ]

    @property
    def insertions(self) -> List[Tuple[int, actions.ConditionalEdit]]:
        return [
            i
            for i, a in enumerate(self.i2w)
            if isinstance(a, actions.ConditionalIns)
        ]

    @property
    def beg_idx(self) -> int:
        return self.i2w.index(actions.Start())

    @property
    def end_idx(self) -> int:
        return self.i2w.index(actions.End())

    @property
    def del_idx(self) -> int:
        return self.i2w.index(actions.ConditionalDel())

    @property
    def copy_idx(self) -> int:
        return self.i2w.index(actions.ConditionalCopy())


@dataclasses.dataclass
class Prefix:
    """Represents overlapping target and prediction strings."""

    prediction: Sequence[Any]
    target: Sequence[Any]
    alignment: int

    @property
    def suffix(self) -> Sequence[Any]:
        return self.target[self.alignment :]  # noqa: E203

    @property
    def leftmost_of_suffix(self) -> Any:
        try:
            return self.target[self.alignment]
        except IndexError:
            return None


@dataclasses.dataclass
class ActionPrefix:
    """Class for wrapping possible actions associated with given prefix."""

    action: Set[actions.Edit]
    prefix: Prefix


def edit_distance(
    x: Sequence[Any],
    y: Sequence[Any],
    del_cost: float = 1.0,
    ins_cost: float = 1.0,
    sub_cost: float = 1.0,
    x_offset: int = 0,
    y_offset: int = 0,
) -> numpy.ndarray:
    """Generates edit distance matrix from source (x) to target (y).

    Args:
        x (Sequence[Any]): source sequence.
        y (Sequence[Any]): target sequence.
        del_cost (float): weight to deletion actions.
        ins_cost (float): weight to insertion actions.
        sub_cost (float): weight to substitution actions.
        x_offset (int): starting index of source sequence.
        y_offset (int): starting index of target sequence.

    Returns:
        numpy.ndarray: edit sequence matrix.
    """
    x_size = len(x) - x_offset + 1
    y_size = len(y) - y_offset + 1
    prefix_matrix = numpy.full(
        (x_size, y_size), numpy.inf, dtype=numpy.float32
    )
    for i in range(x_size):
        prefix_matrix[i, 0] = i * del_cost
    for j in range(y_size):
        prefix_matrix[0, j] = j * ins_cost
    for i in range(1, x_size):
        for j in range(1, y_size):
            if x[i - 1 + x_offset] == y[j - 1 + y_offset]:
                substitution = 0.0
            else:
                substitution = sub_cost
            prefix_matrix[i, j] = min(
                prefix_matrix[i - 1, j] + del_cost,
                prefix_matrix[i, j - 1] + ins_cost,
                prefix_matrix[i - 1, j - 1] + substitution,
            )
    return prefix_matrix


class Expert:
    """Oracle scores possible edit actions between prediction and target.

    Args:
        actions (ActionVocabulary): vocabulary of possible edit actions.
        aligner (StochasticEditDistance): an alignment object to score edits
            between source and target strings.
        oracle_factor (int): a scaling factor for scheduling predictions used
            in transducer training.
    """

    actions: ActionVocabulary
    aligner: sed.StochasticEditDistance
    oracle_factor: int
    roll_in: int

    def __init__(self, actions, aligner, oracle_factor=defaults.ORACLE_FACTOR):
        self.actions = actions
        self.oracle_factor = oracle_factor
        self.roll_in = 1
        self.aligner = aligner

    def find_valid_actions(
        self,
        source: Sequence[Any],
        alignment: int,
        prefixes: Iterable[Prefix],
    ) -> List[ActionPrefix]:
        """Provides edit actions for source symbol and prefix.

        Args:
            source (Sequence[Any]): source string to perform edit actions over.
            alignment (int): index for current symbol to edit in source string.
            prefixes (Iterable[Prefix]): prefix objects aligning current prefix
                overlap between target and source strings. (Can be multiple.)

        Returns:
            List[ActionPrefix]: edit actions and current prefix
                overlap between target and prediction associated with action.
        """
        input_not_empty = alignment < len(source)
        attention = source[alignment] if input_not_empty else None
        action_prefixes = []
        for prefix in prefixes:
            prefix_insert = (
                prefix.leftmost_of_suffix
            )  # First symbol in target string not overlapping with prediction.
            if prefix_insert is None:  # No remaining symbols in target.
                valid_action = {actions.End()}
            else:  # More symbols to go. Insertion is always valid.
                valid_action = {actions.Ins(prefix_insert)}
            if input_not_empty:
                if prefix_insert is not None:
                    # The target symbol and source symbol are same. Copy.
                    if prefix_insert == attention:
                        # TODO: These actions are in maxwell. Remove?
                        valid_action.add(
                            actions.Copy(attention, prefix_insert)
                        )
                    # Target and source symbol are different. Sub.
                    else:
                        valid_action.add(
                            actions.Sub(old=attention, new=prefix_insert)
                        )
                # Source symbol deleted. Further actions can occur.
                valid_action.add(actions.Del(attention))
            action_prefix = ActionPrefix(valid_action, prefix)
            action_prefixes.append(action_prefix)
        return action_prefixes

    def roll_in_schedule(self, epoch: int) -> float:
        """Gets probability of sampling from oracle given current epoch."""
        if self.aligner is None:
            raise AlignerError(
                """Expert called `roll_in_schedule` but there
                               is no aligner present to allow oracle
                               predictions"""
            )
        self.roll_in = 1 - self.oracle_factor / (
            self.oracle_factor + numpy.exp(epoch / self.oracle_factor)
        )

    def explore(self) -> bool:
        """Randomly determines whether expert should advise model."""
        return numpy.random.rand() <= self.roll_in

    def roll_out(
        self,
        source: Sequence[Any],
        target: Sequence[Any],
        alignment: int,
        action_prefixes: Iterable[ActionPrefix],
    ) -> Dict[actions.Edit, float]:
        """Scores potential actions by a predicted 'cost to go' for target.

        Score sums potential edit sequence with cost of action.

        Args:
            source (Sequence[Any]): source string to perform edit actions over.
            target (Sequence[Any]): target string for edit actions.
            alignment (int): index for current symbol to edit in source string.
            action_prefixes (Iterable[ActionsPrefix]): set of edit
                actions-prefix pairs to evaluate.

        Returns:
            Dict[Edit, float]: edit actions and their respective scores.
        """
        costs_to_go = {}
        for action_prefix in action_prefixes:
            suffix_begin = action_prefix.prefix.alignment
            for action in action_prefix.action:
                if isinstance(action, actions.Del):
                    s_offset = alignment + 1
                    t_offset = suffix_begin
                elif isinstance(action, actions.Ins):
                    s_offset = alignment
                    t_offset = suffix_begin + 1
                elif isinstance(action, actions.Sub):
                    s_offset = alignment + 1
                    t_offset = suffix_begin + 1
                elif isinstance(action, actions.End):
                    s_offset = alignment
                    t_offset = suffix_begin
                else:
                    raise ActionError(f"Unknown action: {action}")
                sequence_cost = self.aligner.action_sequence_cost(
                    source, target, s_offset, t_offset
                )
                action_cost = self.aligner.action_cost(action)
                cost = action_cost + sequence_cost
                if action not in costs_to_go or costs_to_go[action] > cost:
                    costs_to_go[action] = cost
        return costs_to_go

    def score(
        self,
        source: Sequence[Any],
        target: Sequence[Any],
        alignment: int,
        prediction: Sequence[Any],
        max_action_seq_len: int = 150,
    ) -> Dict[actions.Edit, float]:
        """Provides potential actions given source, target, and prediction.

        Args:
            source (Sequence[Any]): source string to perform edit actions over.
            target (Sequence[Any]): target string for edit actions.
            alignment (int): index for current symbol to edit in source string.
            prediction (Sequence[Any]): current prediction from previous edits.

        Returns:
            Dict[Edit, float]: edit actions and their respective scores.
        """
        prefixes = self.find_prefixes(prediction, target)
        valid_actions = (
            self.find_valid_actions(source, alignment, prefixes)
            if len(prediction) <= max_action_seq_len
            else {actions.End()}
        )
        valid_action_scores = self.roll_out(
            source, target, alignment, valid_actions
        )
        return valid_action_scores

    @staticmethod
    def find_prefixes(
        prediction: Sequence[Any], target: Sequence[Any]
    ) -> List[Prefix]:
        """Creates prefix objects for prediction and target.

        Args:
            prediction (Sequence[Any]): current prediction from prior edits.
            target (Sequence[Any]): target string for edit actions.

        Returns:
            List[Prefix]: prefix objects that index overlap between
               prediction and target with minimal edit distance.
        """
        prefix_matrix = edit_distance(prediction, target)
        prediction_row = prefix_matrix[-1]
        return [
            Prefix(prediction, target, i)
            for i in numpy.where(prediction_row == prediction_row.min())[0]
        ]


def get_expert(
    dataset: data.Dataset,
    index: data.Index,
    epochs: int = defaults.ORACLE_EM_EPOCHS,
    oracle_factor: int = defaults.ORACLE_FACTOR,
    sed_params_path: str = None,
    read_from_file: bool = False,
) -> Expert:
    """Generates expert object for training transducer.

    Args:
        dataset (data.Dataset): dataset for generating expert vocabulary.
        index (data.Index): index for mapping symbols to indices.
        epochs (int): number of EM epochs.
        oracle_factor (float): scaling factor to determine rate of
            expert rollout sampling.
        sed_params_path (str): path to read/write location of sed parameters.
        read_from_file (bool): bool for whether to use existing parameters.
            If True, reads from sed_params_path. If False, writes to
            sed_params_path.

    Returns:
        expert.Expert.
    """

    def _generate_data(
        dataset: data.Dataset,
        index: data.Index,
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """Helper function to manage data encoding for SED.

        We want encodings without padding. This encodes only raw source-target
        text for the Maxwell library.

        Args:
            dataset (data.Dataset): dataset for generating expert vocabulary.
            index (data.Index): index for mapping symbols to indices.

        Yields:
            Tuple[List[int, List[int]]]: lists of source and target entries.
        """
        if not dataset.has_target:
            raise Error("Dataset has no target")
        for sample in dataset.samples:
            source, *_, target = sample
            yield (
                [index(symbol) for symbol in source],
                [index(symbol) for symbol in target],
            )

    actions = ActionVocabulary(index)
    if read_from_file:
        sed_params = sed.ParamDict.read_params(sed_params_path)
        sed_aligner = sed.StochasticEditDistance(sed_params)
    else:
        sed_aligner = sed.StochasticEditDistance.fit_from_data(
            _generate_data(dataset, index),
            epochs=epochs,
        )
        sed_aligner.params.write_params(sed_params_path)
    return Expert(actions, aligner=sed_aligner, oracle_factor=oracle_factor)


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds expert configuration options to the argument parser.

    These are only needed at training time.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--oracle_em_epochs",
        type=int,
        default=defaults.ORACLE_EM_EPOCHS,
        help="Number of EM epochs "
        "(transducer architecture only). Default: %(default)s.",
    )
    parser.add_argument(
        "--oracle_factor",
        type=int,
        default=defaults.ORACLE_FACTOR,
        help="Roll-in schedule parameter "
        "(transducer architecture only). Default: %(default)s.",
    )
    parser.add_argument(
        "--sed_params",
        type=str,
        help="Path to input SED parameters (transducer architecture only).",
    )
