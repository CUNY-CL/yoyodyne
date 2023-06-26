import pytest

from yoyodyne import collators, dataconfig, datasets, indexes


@pytest.mark.parametrize(
    ["arch", "has_features", "has_target", "expected_separate_features"],
    [
        ("feature_invariant_transformer", True, True, False),
        ("feature_invariant_transformer", True, False, False),
        ("lstm", True, True, False),
        ("lstm", False, True, False),
        ("lstm", True, False, False),
        ("lstm", False, False, False),
        ("pointer_generator_lstm", True, True, True),
        ("pointer_generator_lstm", False, True, False),
        ("pointer_generator_lstm", True, False, True),
        ("pointer_generator_lstm", False, False, False),
        ("transducer", True, True, True),
        ("transducer", False, True, False),
        ("transducer", True, False, True),
        ("transducer", False, False, False),
        ("transformer", True, True, False),
        ("transformer", False, True, False),
        ("transformer", True, False, False),
        ("transformer", False, False, False),
    ],
)
def test_get_collator(
    arch,
    has_features,
    has_target,
    expected_separate_features,
    make_trivial_tsv_file,
):
    config = dataconfig.DataConfig(
        features_col=3 if has_features else 0,
        target_col=2 if has_target else 0,
    )
    index = indexes.IndexFeatures if has_features else indexes.IndexNoFeatures
    dataset = datasets.BaseDataset(make_trivial_tsv_file, config, index=index)
    collator = collators.Collator(
        dataset,
        arch,
    )
    assert collator.has_target == has_target
    assert collator.separate_features == expected_separate_features
