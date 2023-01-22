import pytest

from yoyodyne import dataconfig, datasets


@pytest.mark.parametrize(
    "features_col, expected_cls",
    [(0, datasets.DatasetNoFeatures), (3, datasets.DatasetFeatures)],
)
def test_get_dataset(make_trivial_tsv_file, features_col, expected_cls):
    filename = make_trivial_tsv_file
    config = dataconfig.DataConfig(features_col=features_col)
    dataset = datasets.get_dataset(filename, config)
    assert type(dataset) is expected_cls
