#!/usr/bin/env python
"""Concatenates source and features columns.

Note that:

* Output is to stdout to allow chaining.
* The source string in the output TSV will have space as its separator.
* Features will be kept distinct using `[...]`.
* Other columns are ignored; they can be merged back in using UNIX `cut` and
  `paste`.
"""

import argparse
import csv
import os
import sys

from yoyodyne import defaults
from yoyodyne.data import tsv


def main(args: argparse.Namespace) -> None:
    parser = tsv.TsvParser(
        source_col=args.source_col,
        features_col=args.features_col,
        target_col=0,
        source_sep=args.source_sep,
        features_sep=args.features_sep,
    )
    assert parser.has_features
    # This is really just a one-column TSV but we want to make sure
    # escaping is handled properly.
    tsv_writer = csv.writer(sys.stdout, delimiter="\t", lineterminator=os.linesep)
    for source, features in parser.samples(args.input_tsv):
        source.extend(features)
        tsv_writer.writerow([" ".join(source)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv")
    parser.add_argument(
        "--source_col",
        type=int,
        default=defaults.SOURCE_COL,
        help="1-indexed column for the input/output source column. Default: %(default)s.",
    )
    parser.add_argument(
        "--features_col",
        type=int,
        default=defaults.TARGET_COL,
        help="1-indexed column for the input features column. Default: %(default)s.",
    )
    parser.add_argument(
        "--source_sep",
        type=str,
        default=defaults.SOURCE_SEP,
        help="String used to split the input source string into symbols; "
        "an empty string indicates that each Unicode codepoint "
        "is its own symbol. Default: %(default)r.",
    )
    parser.add_argument(
        "--features_sep",
        type=str,
        default=defaults.FEATURES_SEP,
        help="String used to split features string into symbols; "
        "an empty string indicates that each Unicode codepoint "
        "is its own symbol. Default: %(default)r.",
    )
    main(parser.parse_args())
