import pytest

from yoyodyne import collators, dataconfig


@pytest.mark.parametrize(
    "arch, include_features, include_target, expected_cls",
    [
        (
            "feature_invariant_transformer",
            True,
            True,
            collators.SourceTargetCollator,
        ),
        (
            "feature_invariant_transformer",
            True,
            False,
            collators.SourceCollator,
        ),
        ("lstm", True, True, collators.SourceTargetCollator),
        ("lstm", False, True, collators.SourceTargetCollator),
        ("lstm", True, False, collators.SourceCollator),
        ("lstm", False, False, collators.SourceCollator),
        (
            "pointer_generator_lstm",
            True,
            True,
            collators.SourceFeaturesTargetCollator,
        ),
        (
            "pointer_generator_lstm",
            False,
            True,
            collators.SourceTargetCollator,
        ),
        (
            "pointer_generator_lstm",
            True,
            False,
            collators.SourceFeaturesCollator,
        ),
        ("pointer_generator_lstm", False, False, collators.SourceCollator),
        ("transducer", True, True, collators.SourceFeaturesTargetCollator),
        ("transducer", False, True, collators.SourceTargetCollator),
        ("transducer", True, False, collators.SourceFeaturesCollator),
        ("transducer", False, False, collators.SourceCollator),
        ("transformer", True, True, collators.SourceTargetCollator),
        ("transformer", False, True, collators.SourceTargetCollator),
        ("transformer", True, False, collators.SourceCollator),
        ("transformer", False, False, collators.SourceCollator),
    ],
)
def test_get_collator(arch, include_features, include_target, expected_cls):
    config = dataconfig.DataConfig(
        features_col=3 if include_features else 0,
        target_col=2 if include_target else 0,
    )
    collator = collators.get_collator(
        1, config, arch  # pad_idx, but it doesn't matter here.
    )
    assert type(collator) is expected_cls
