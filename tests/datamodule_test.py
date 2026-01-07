"""Tests datamodule instantiation."""

import csv
import os
import tempfile
import unittest

from torch.utils import data as torch_data
from yoyodyne import data


class DatamoduleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.TemporaryDirectory()
        print(str(cls.model_dir.name))
        cls.tsv = tempfile.NamedTemporaryFile("wt", suffix=".tsv")
        tsv_writer = csv.writer(cls.tsv, delimiter="\t")
        tsv_writer.writerow(["ayılmak", "ayıldık", "V;IND;1;PL;PST;POS;DECL"])
        cls.tsv.flush()

    def test_datamodule_instantiation(self):
        module = data.DataModule(
            model_dir=self.model_dir,
            train=self.tsv.name,
            features_col=3,
        )
        self.assertTrue(module.has_target)
        self.assertTrue(module.has_features)
        self.assertTrue(os.path.isfile(f"{self.model_dir}/index.pkl"))
        self.assertIsInstance(module.train_dataloader(), torch_data.DataLoader)

    @classmethod
    def tearDownClass(cls):
        cls.model_dir.cleanup()
        cls.tsv.close()


if __name__ == "__main__":
    unittest.main()
