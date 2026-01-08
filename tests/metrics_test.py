import unittest

import torch
from parameterized import parameterized

from yoyodyne import metrics, special


class TestMetrics(unittest.TestCase):

    @parameterized.expand(
        [
            (
                "perfect_match",
                [[102, 103, special.END_IDX, special.PAD_IDX]],
                [[102, 103, special.END_IDX, special.PAD_IDX]],
                200,
                1.0,
            ),
            (
                "mixed_length",
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
        ]
    )
    def test_accuracy(self, name, hypo, gold, num_classes, expected):
        metric = metrics.Accuracy(num_classes)
        hypo_tensor = torch.tensor(hypo)
        gold_tensor = torch.tensor(gold)
        result = metric(hypo_tensor, gold_tensor)
        self.assertEqual(result.item(), expected)

    @parameterized.expand(
        [
            (
                "match",
                [[102, 103, special.END_IDX]],
                [[102, 103, special.END_IDX]],
                0,
                2,
            ),
            (
                "substitution",
                [[102, 104, special.END_IDX]],
                [[102, 103, special.END_IDX]],
                1,
                2,
            ),
            (
                "insertion",
                [[102, 103, 104, special.END_IDX]],
                [[102, 103, special.END_IDX]],
                1,
                2,
            ),
            (
                "deletion",
                [[102, special.END_IDX, special.PAD_IDX]],
                [[102, 103, special.END_IDX]],
                1,
                2,
            ),
        ]
    )
    def test_ser_basic(self, name, hypo, gold, expected_edits, expected_len):
        metric = metrics.SER()
        metric.update(torch.tensor(hypo), torch.tensor(gold))
        self.assertEqual(metric.edits.item(), expected_edits)
        self.assertEqual(metric.length.item(), expected_len)

    def test_ser_3d_input(self):
        metric = metrics.SER()
        # Batch=1, Vocab=200, SeqLen=3
        # Hypo argmax: [102, 104, END_IDX]
        # Gold: [102, 103, END_IDX]
        hypo_logits = torch.full((1, 200, 3), -10.0)
        hypo_logits[0, 102, 0] = 10.0
        hypo_logits[0, 104, 1] = 10.0
        hypo_logits[0, special.END_IDX, 2] = 10.0
        gold = torch.tensor([[102, 103, special.END_IDX]])
        metric.update(hypo_logits, gold)
        self.assertEqual(metric.edits.item(), 1)
        self.assertEqual(metric.length.item(), 2)

    def test_ser_multi_batch_complex(self):
        metric = metrics.SER()
        # Row 1: "102 103" -> "102 103" (0 edits, len 2)
        # Row 2: "105" -> "104" (1 sub, len 1)
        # Row 3: "106 107 108" -> "106 108" (1 ins, len 2)
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
        self.assertEqual(metric.edits.item(), 2)
        self.assertEqual(metric.length.item(), 5)
        self.assertAlmostEqual(metric.compute().item(), 2 / 5)

    def test_ser_no_end_idx(self):
        metric = metrics.SER()
        # If END_IDX is missing, it should use the full tensor length.
        # Symbols 102, 102, 102.
        hypo = torch.tensor([[102, 102, 102]])  # len 3.
        gold = torch.tensor([[102, 102, special.END_IDX]])  # len 2.
        metric.update(hypo, gold)
        # "102 102 102" vs "102 102" is 1 insertion.
        self.assertEqual(metric.edits.item(), 1)
        self.assertEqual(metric.length.item(), 2)
