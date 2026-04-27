import pytest
from yoyodyne.data import tsv


def make_tsv(tmp_path, lines):
    """Write lines (list of tuples) to a temp .tsv file."""
    p = tmp_path / "data.tsv"
    p.write_text(
        "\n".join("\t".join(row) for row in lines),
        encoding="utf-8",
    )
    return str(p)


class TestGetString:

    def test_missing_source_col_parse_line(self):
        """parse_line on a 1-column row when source_col=2."""
        parser = tsv.TsvParser(source_col=2)
        with pytest.raises(tsv.Error) as error:
            parser.parse_line("only_one_col")
        msg = str(error.value)
        assert "Column 2" in msg
        assert "1 column" in msg
        assert "only_one_col" in msg

    def test_missing_target_col_parse_line(self):
        """parse_line on a 1-column row when target_col=3."""
        parser = tsv.TsvParser(source_col=1, target_col=3)
        with pytest.raises(tsv.Error) as error:
            parser.parse_line("src\ttgt")
        msg = str(error.value)
        assert "Column 3" in msg
        assert "2 column" in msg
        assert repr(["src", "tgt"]) in msg

    def test_missing_features_col_parse_line(self):
        """parse_line with features_col pointing beyond the row."""
        parser = tsv.TsvParser(source_col=1, features_col=4, target_col=2)
        with pytest.raises(tsv.Error) as error:
            parser.parse_line("src\ttgt")
        msg = str(error.value)
        assert "Column 4" in msg
        assert "2 column" in msg


class TestSamples:
    def test_error_includes_filename(self, tmp_path):
        """An out-of-bounds column error from samples() includes path."""
        path = make_tsv(tmp_path, [("src",), ("src",)])
        parser = tsv.TsvParser(source_col=1, target_col=2)
        with pytest.raises(tsv.Error) as error:
            list(parser.samples(path))
        assert "data.tsv" in str(error.value)

    def test_error_includes_line_number_first_line(self, tmp_path):
        path = make_tsv(tmp_path, [("src",)])
        parser = tsv.TsvParser(source_col=1, target_col=2)
        with pytest.raises(tsv.Error) as error:
            list(parser.samples(path))
        assert "line 1" in str(error.value)

    def test_error_includes_line_number_later_line(self, tmp_path):
        """Good rows followed by a bad row."""
        path = make_tsv(
            tmp_path,
            [
                ("src1", "tgt1"),  # line 1 — fine
                ("src2", "tgt2"),  # line 2 — fine
                ("src3",),  # line 3 — missing target col
            ],
        )
        parser = tsv.TsvParser(source_col=1, target_col=2)
        with pytest.raises(tsv.Error) as error:
            list(parser.samples(path))
        assert "line 3" in str(error.value)

    def test_good_data_does_not_raise(self, tmp_path):
        """Well-formed data does not raise an error."""
        path = make_tsv(tmp_path, [("a b", "x y"), ("c d", "z w")])
        parser = tsv.TsvParser(source_col=1, target_col=2)
        samples = list(parser.samples(path))
        assert len(samples) == 2
