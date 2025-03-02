#!/usr/bin/env python
"""Generates strings according to some simple criteria."""

import argparse
import random
import string

import numpy
from scipy.stats import poisson


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    # The docs say that Scipy uses Numpy's randomizer.
    numpy.random.seed(args.seed)
    assert 0 < args.vocabulary <= 26
    vocabulary = list(string.ascii_lowercase[: args.vocabulary])
    for length in poisson.rvs(mu=args.length, size=args.size):
        # Add-one to avoids length-zero strings.
        print("".join(random.choice(vocabulary) for _ in range(1 + length)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocabulary",
        type=int,
        required=True,
        help="value [1, 26] specifying vocabulary size",
    )
    parser.add_argument(
        "--length",
        type=float,
        required=True,
        help="length parameter for Poisson length distribution",
    )
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="number of examples to generate",
    )
    main(parser.parse_args())
