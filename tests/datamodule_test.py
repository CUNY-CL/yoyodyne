"""Tests datamodule instantiation."""

import csv
import os

import pytest
from torch.utils import data as torch_data

from yoyodyne import data


class TestDatamodule:

    @pytest.fixture
    def paths(self, tmp_path):
        # Creates a temporary directory for the model.
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # Creates a temporary TSV file with content.
        tsv_path = tmp_path / "test.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerow(
                ["ayılmak", "ayıldık", "V;IND;1;PL;PST;POS;DECL"]
            )
        return str(model_dir), str(tsv_path)

    def test_datamodule_instantiation(self, paths):
        model_dir, tsv_path = paths
        module = data.DataModule(
            model_dir=model_dir,
            train=tsv_path,
            features_col=3,
        )
        assert module.has_target
        assert module.has_features
        assert os.path.isfile(os.path.join(model_dir, "index.pkl"))
        assert isinstance(module.train_dataloader(), torch_data.DataLoader)
