import torch
import pytest

from yoyodyne import metrics, special


@pytest.mark.parametrize(
    "hypo, gold, num_classes, expected",
    [
        # perfect_match
        (
            [[102, 103, special.END_IDX, special.PAD_IDX]],
            [[102, 103, special.END_IDX, special.PAD_IDX]],
            200,
            1.0,
        ),
        # mixed_length
        (
            [
                [102, 103, special.END_IDX, special.PAD_IDX],
                [104, 105, special.END_IDX, special.PAD_IDX],
            ],
            [
                [102, 103, special.END_IDX, special.PAD_IDX],
                [104, 105, 106, special.END_IDX],
            ],
            200,
            0.5,
        ),
    ],
    ids=["perfect_match", "mixed_length"],
)
def test_accuracy(hypo, gold, num_classes, expected):
    metric = metrics.Accuracy(num_classes)
    hypo_tensor = torch.tensor(hypo)
    gold_tensor = torch.tensor(gold)
    result = metric(hypo_tensor, gold_tensor)
    assert result.item() == expected


@pytest.mark.parametrize(
    "hypo, gold, expected_edits, expected_len",
    [
        ([[102, 103, special.END_IDX]], [[102, 103, special.END_IDX]], 0, 2),
        ([[102, 104, special.END_IDX]], [[102, 103, special.END_IDX]], 1, 2),
        (
            [[102, 103, 104, special.END_IDX]],
            [[102, 103, special.END_IDX]],
            1,
            2,
        ),
        (
            [[102, special.END_IDX, special.PAD_IDX]],
            [[102, 103, special.END_IDX]],
            1,
            2,
        ),
    ],
    ids=["match", "substitution", "insertion", "deletion"],
)
def test_ser_basic(hypo, gold, expected_edits, expected_len):
    metric = metrics.SER()
    metric.update(torch.tensor(hypo), torch.tensor(gold))
    assert metric.edits.item() == expected_edits
    assert metric.length.item() == expected_len


def test_ser_3d_input():
    metric = metrics.SER()
    hypo_logits = torch.full((1, 200, 3), -10.0)
    hypo_logits[0, 102, 0] = 10.0
    hypo_logits[0, 104, 1] = 10.0
    hypo_logits[0, special.END_IDX, 2] = 10.0
    gold = torch.tensor([[102, 103, special.END_IDX]])
    metric.update(hypo_logits, gold)
    assert metric.edits.item() == 1
    assert metric.length.item() == 2


def test_ser_multi_batch_complex():
    metric = metrics.SER()
    hypo = torch.tensor(
        [
            [102, 103, special.END_IDX, special.PAD_IDX],
            [104, special.END_IDX, special.PAD_IDX, special.PAD_IDX],
            [106, 107, 108, special.END_IDX],
        ]
    )
    gold = torch.tensor(
        [
            [102, 103, special.END_IDX, special.PAD_IDX],
            [105, special.END_IDX, special.PAD_IDX, special.PAD_IDX],
            [106, 108, special.END_IDX, special.PAD_IDX],
        ]
    )
    metric.update(hypo, gold)
    assert metric.edits.item() == 2
    assert metric.length.item() == 5
    assert metric.compute().item() == pytest.approx(2 / 5)


def test_ser_no_end_idx():
    metric = metrics.SER()
    # If END_IDX is missing, it should use the full tensor length.
    hypo = torch.tensor([[102, 102, 102]])  # len 3.
    gold = torch.tensor([[102, 102, special.END_IDX]])  # len 2.
    metric.update(hypo, gold)
    assert metric.edits.item() == 1
    assert metric.length.item() == 2
