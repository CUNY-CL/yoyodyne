#!/usr/bin/env python
"""Retrieves hyperparameters from a W&B run."""

import argparse
import logging

import wandb

from yoyodyne import defaults

# Expand as needed.
FLAGS_TO_IGNORE = frozenset(
    ["eval_metrics", "local_run_dir", "num_parameters"]
)


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    run = api.run(f"{args.run_path}")
    logging.info("Run URL: %s", run.url)
    args = []
    for key, value in sorted(run.config.items()):
        # Exclusions:
        #
        # * Explicitly ignored flags
        # * Keys with "/" is for redundant parameters in the scheduler.
        # * Key/value pairs that are defaults can be omitted.
        # * Keys ending in "_cls", "_idx", and "vocab_size" are set
        #   automatically.
        # * None values are defaults definitionally.
        if key in FLAGS_TO_IGNORE:
            continue
        if "/" in key:
            continue
        key_upper = key.upper()
        if hasattr(defaults, key_upper):
            if getattr(defaults, key_upper) == value:
                continue
        if (
            key.endswith("_cls")
            or key.endswith("_idx")
            or key.endswith("vocab_size")
        ):
            continue
        if value is None:
            continue
        args.append((key, value))
    print(" ".join(f"--{key} {value}" for key, value in args))


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_path", help="Run path (in the form entity/project/run_id)"
    )
    main(parser.parse_args())
