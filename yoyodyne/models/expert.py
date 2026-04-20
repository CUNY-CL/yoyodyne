"""Expert for character-level string transduction.

Given an input string x, a target string t, alignment index i, and a partial
prediction y, it returns optimal cost-to-go for all valid edit actions.

Also includes ActionVocabulary class for compatibility with the `maxwell`
dictionary. This class stores valid edit actions for given dataset."""

import dataclasses
import math
from typing import Any, Iterable, Sequence

import numpy
from maxwell import actions, sed
import torch

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
    i2w: list[actions.Edit]
    w2i: dict[actions.Edit, int]
    beg_idx: int
    end_idx: int
    del_idx: int
    copy_idx: int
    start_vocab_idx: int
    target_characters: set[Any]
    insertions: list[int]
    substitutions: list[int]

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
        self.beg_idx = self.w2i[actions.Start()]
        self.end_idx = self.w2i[actions.End()]
        self.del_idx = self.w2i[actions.ConditionalDel()]
        self.copy_idx = self.w2i[actions.ConditionalCopy()]
        self.insertions = [
            i
            for i, a in enumerate(self.i2w)
            if isinstance(a, actions.ConditionalIns)
        ]
        self.substitutions = [
            i
            for i, a in enumerate(self.i2w)
            if isinstance(a, actions.ConditionalSub)
        ]

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

    def lookup(self, word: actions.Edit) -> int:
        return self.w2i[word]


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

    action: set[actions.Edit]
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
    model_roll_in_prob: float

    def __init__(
        self,
        actions: ActionVocabulary,
        aligner: sed.StochasticEditDistance,
        oracle_factor: int = defaults.ORACLE_FACTOR,
    ):
        self.actions = actions
        self.oracle_factor = oracle_factor
        # Probability of sampling from the model; initially zero.
        self.model_roll_in_prob = 0.0
        self.aligner = aligner

    def find_valid_actions(
        self,
        source: Sequence[Any],
        alignment: int,
        prefixes: Iterable[Prefix],
    ) -> list[ActionPrefix]:
        """Provides edit actions for source symbol and prefix.

        Args:
            source (Sequence[Any]): source string to perform edit actions over.
            alignment (int): index for current symbol to edit in source string.
            prefixes (Iterable[Prefix]): prefix objects aligning current prefix
                overlap between target and source strings. (Can be multiple.)

        Returns:
            list[ActionPrefix]: edit actions and current prefix
                overlap between target and prediction associated with action.
        """
        input_not_empty = alignment < len(source)
        attention = source[alignment] if input_not_empty else None
        action_prefixes = []
        for prefix in prefixes:
            prefix_insert = prefix.leftmost_of_suffix
            if prefix_insert is None:
                valid_action = {actions.End()}
            else:
                valid_action = {actions.Ins(prefix_insert)}
            if input_not_empty:
                if prefix_insert is not None:
                    if prefix_insert == attention:
                        valid_action.add(
                            actions.Copy(attention, prefix_insert)
                        )
                    else:
                        valid_action.add(
                            actions.Sub(old=attention, new=prefix_insert)
                        )
                valid_action.add(actions.Del(attention))
            action_prefix = ActionPrefix(valid_action, prefix)
            action_prefixes.append(action_prefix)
        return action_prefixes

    def roll_in_schedule(self, epoch: int) -> None:
        """Updates the model roll-in probability for the given epoch.

        Uses an inverse-sigmoid (logistic) decay schedule so that the model
        starts by following the expert and gradually explores its own policy.

        Args:
            epoch (int): current training epoch (0-indexed).
        """
        if self.aligner is None:
            raise AlignerError("No aligner for oracle predictions")
        self.model_roll_in_prob = 1 - self.oracle_factor / (
            self.oracle_factor + math.exp(epoch / self.oracle_factor)
        )

    def explore(self) -> bool:
        """Randomly determines whether the model should sample its own policy.

        Returns True with probability `model_roll_in_prob`, meaning the model
        samples from its own distribution rather than following the expert.

        Returns:
            bool: True if model should sample its own action.
        """
        return torch.rand(1).item() <= self.model_roll_in_prob

    def roll_out(
        self,
        source: Sequence[Any],
        target: Sequence[Any],
        alignment: int,
        action_prefixes: Iterable[ActionPrefix],
    ) -> dict[actions.Edit, float]:
        """Scores potential actions by a predicted 'cost to go' for target.

        Score sums potential edit sequence with cost of action.

        Args:
            source (Sequence[Any]): source string to perform edit actions over.
            target (Sequence[Any]): target string for edit actions.
            alignment (int): index for current symbol to edit in source string.
            action_prefixes (Iterable[ActionsPrefix]): set of edit
                actions-prefix pairs to evaluate.

        Returns:
            dict[Edit, float]: edit actions and their respective scores.
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
    ) -> dict[actions.Edit, float]:
        """Provides potential actions given source, target, and prediction.

        Args:
            source (Sequence[Any]): source string to perform edit actions over.
            target (Sequence[Any]): target string for edit actions.
            alignment (int): index for current symbol to edit in source string.
            prediction (Sequence[Any]): current prediction from previous edits.
            max_action_seq_len (int): maximum action sequence length before
                forcing End action.

        Returns:
            dict[Edit, float]: edit actions and their respective scores.
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
    ) -> list[Prefix]:
        """Creates prefix objects for prediction and target.

        Args:
            prediction (Sequence[Any]): current prediction from prior edits.
            target (Sequence[Any]): target string for edit actions.

        Returns:
            list[Prefix]: prefix objects that index overlap between
               prediction and target with minimal edit distance.
        """
        prefix_matrix = edit_distance(prediction, target)
        prediction_row = prefix_matrix[-1]
        return [
            Prefix(prediction, target, i)
            for i in numpy.where(prediction_row == prediction_row.min())[0]
        ]


def get_expert(
    index: data.Index, path: str, oracle_factor: int = defaults.ORACLE_FACTOR
) -> Expert:
    """Generates expert object for training transducer.

    Args:
        index (data.Index): index for mapping symbols to indices.
        path (str): path to SED parameter .pkl file.
        oracle_factor (int): scaling factor to determine rate of expert
            rollout sampling.

    Returns:
        expert.Expert.
    """
    vocab = ActionVocabulary(index)
    aligner = sed.StochasticEditDistance(sed.ParamDict.read_params(path))
    return Expert(vocab, aligner, oracle_factor)
