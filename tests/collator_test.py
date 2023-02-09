import pytest

from yoyodyne import collators, dataconfig


@pytest.mark.parametrize(
    [
        "arch",
        "has_features",
        "has_target",
        "expected_separate_features",
        "max_src_length",
        "max_tgt_length",
    ],
    [
        ("feature_invariant_transformer", True, True, False, 128, 128),
        ("feature_invariant_transformer", True, False, False, 128, 128),
        ("lstm", True, True, False, 128, 128),
        ("lstm", False, True, False, 128, 128),
        ("lstm", True, False, False, 128, 128),
        ("lstm", False, False, False, 128, 128),
        ("pointer_generator_lstm", True, True, True, 128, 128),
        ("pointer_generator_lstm", False, True, False, 128, 128),
        ("pointer_generator_lstm", True, False, True, 128, 128),
        ("pointer_generator_lstm", False, False, False, 128, 128),
        ("transducer", True, True, True, 128, 128),
        ("transducer", False, True, False, 128, 128),
        ("transducer", True, False, True, 128, 128),
        ("transducer", False, False, False, 128, 128),
        ("transformer", True, True, False, 128, 128),
        ("transformer", False, True, False, 128, 128),
        ("transformer", True, False, False, 128, 128),
        ("transformer", False, False, False, 128, 128),
    ],
)
def test_get_collator(
    arch,
    has_features,
    has_target,
    expected_separate_features,
    max_src_length,
    max_tgt_length,
):
    config = dataconfig.DataConfig(
        features_col=3 if has_features else 0,
        target_col=2 if has_target else 0,
    )
    collator = collators.Collator(
        1,  # pad_idx, but it doesn't matter here.
        config,
        arch,
        max_src_length,
        max_tgt_length,
    )
    assert collator.has_target == has_target
    assert collator.separate_features == expected_separate_features
