import csv
import pytest

from yoyodyne import datasets


@pytest.mark.parametrize(
    "features_col, expected_cls",
    [(0, datasets.DatasetNoFeatures), (3, datasets.DatasetFeatures)],
)
def test_get_dataset(make_tsv_file, features_col, expected_cls):
    filename = make_tsv_file
    dataset = datasets.get_dataset(filename, features_col=features_col)
    assert type(dataset) is expected_cls
