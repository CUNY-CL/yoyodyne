#!/usr/bin/env python
"""Replaces low-frequency symbols with UNK.

Frequencies are computed over training and validation data, using separate
counts per column. Output TSVs contained only the process source, target (and
optionally, features) columns; other columns can be merged back in using UNIX
`cut` and `paste` if desired. Processed columns all have space separators.
"""

import argparse
import collections
import csv
import logging
import os
import sys

from yoyodyne import defaults, special
from yoyodyne.data import tsv


def _build_counters(
    parser: tsv.TsvParser,
    path: str,
) -> tuple[collections.Counter, collections.Counter, collections.Counter]:
    source_counter = collections.Counter()
    target_counter = collections.Counter()
    features_counter = collections.Counter()
    for source, target, features in parser.samples(path):
        source_counter.update(source)
        target_counter.update(target)
        if features is not None:
            features_counter.update(features)
    return source_counter, target_counter, features_counter


def _unk_symbols(
    symbols: list[str],
    counter: collections.Counter,
    freq: int,
) -> tuple[list[str], int, int]:
    """Returns UNK-replaced symbols, replacement count, and total count."""
    result = []
    replaced = 0
    for symbol in symbols:
        if counter[symbol] < freq:
            result.append(special.UNK)
            replaced += 1
        else:
            result.append(symbol)
    return result, replaced, len(symbols)


def _process_and_write(
    parser: tsv.TsvParser,
    input_path: str,
    output_path: str,
    source_counter: collections.Counter,
    target_counter: collections.Counter,
    features_counter: collections.Counter,
    freq: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Writes UNK-replaced TSV; returns (replaced, total) per column."""
    source_replaced = source_total = 0
    target_replaced = target_total = 0
    features_replaced = features_total = 0
    with open(output_path, "w") as sink:
        writer = csv.writer(sink, delimiter="\t", lineterminator=os.linesep)
        for source, target, features in parser.samples(input_path):
            source, sr, st = _unk_symbols(source, source_counter, freq)
            target, tr, tt = _unk_symbols(target, target_counter, freq)
            source_replaced += sr
            source_total += st
            target_replaced += tr
            target_total += tt
            row = [" ".join(source), " ".join(target)]
            if features is not None:
                features, fr, ft = _unk_symbols(
                    features, features_counter, freq
                )
                features_replaced += fr
                features_total += ft
                row.append(" ".join(features))
            writer.writerow(row)
    return (
        (source_replaced, source_total),
        (target_replaced, target_total),
        (features_replaced, features_total),
    )


def _pct(replaced: int, total: int) -> str:
    return f"{100 * replaced / total:.2f}%" if total else "N/A"


def _log_stats(
    label: str,
    stats: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    has_features: bool,
) -> None:
    (sr, st), (tr, tt), (fr, ft) = stats
    logging.info("%s source UNK rate: %s", label, _pct(sr, st))
    logging.info("%s target UNK rate: %s", label, _pct(tr, tt))
    if has_features:
        logging.info("%s features UNK rate: %s", label, _pct(fr, ft))


def main(args: argparse.Namespace) -> None:
    if (args.val_input is None) != (args.val_output is None):
        logging.error(
            "--val_input and --val_output must be specified together"
        )
        sys.exit(1)
    has_val = args.val_input is not None
    parser = tsv.TsvParser(
        source_col=args.source_col,
        features_col=args.features_col,
        target_col=args.target_col,
        source_sep=args.source_sep,
        features_sep=args.features_sep,
    )
    source_counter, target_counter, features_counter = _build_counters(
        parser, args.train_input
    )
    if has_val:
        val_source, val_target, val_features = _build_counters(
            parser, args.val_input
        )
        source_counter += val_source
        target_counter += val_target
        features_counter += val_features
    train_stats = _process_and_write(
        parser,
        args.train_input,
        args.train_output,
        source_counter,
        target_counter,
        features_counter,
        args.freq,
    )
    val_stats = None
    if has_val:
        val_stats = _process_and_write(
            parser,
            args.val_input,
            args.val_output,
            source_counter,
            target_counter,
            features_counter,
            args.freq,
        )
    _process_and_write(
        parser,
        args.test_input,
        args.test_output,
        source_counter,
        target_counter,
        features_counter,
        args.freq,
    )
    _log_stats("train", train_stats, parser.has_features)
    if val_stats is not None:
        _log_stats("val", val_stats, parser.has_features)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_input", default=None)
    parser.add_argument("--val_output", default=None)
    parser.add_argument("--test_input", required=True)
    parser.add_argument("--test_output", required=True)
    parser.add_argument(
        "--source_col",
        type=int,
        default=defaults.SOURCE_COL,
        help="1-indexed column for the source column. Default: %(default)s.",
    )
    parser.add_argument(
        "--target_col",
        type=int,
        default=defaults.TARGET_COL,
        help="1-indexed column for the target column. Default: %(default)s.",
    )
    parser.add_argument(
        "--features_col",
        type=int,
        default=0,
        help="1-indexed column for the features column; 0 disables features. "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--source_sep",
        type=str,
        default=defaults.SOURCE_SEP,
        help="String used to split the source string into symbols; "
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
    parser.add_argument(
        "--freq",
        type=int,
        default=1,
        help="Minimum frequency threshold; symbols below this are replaced "
        "with UNK. Default: %(default)s.",
    )
    main(parser.parse_args())
