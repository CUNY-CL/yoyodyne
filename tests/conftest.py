"""Reusable fixtures."""


import csv

import pytest


@pytest.fixture()
def make_trivial_tsv_file(tmp_path):
    path = tmp_path / "data.tsv"
    with open(path, "w") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        tsv_writer.writerow(["abscond", "absconded", "regular"])
        tsv_writer.writerow(["muff", "muffed", "regular"])
        tsv_writer.writerow(["outshine", "outshone", "irregular"])
    return path
