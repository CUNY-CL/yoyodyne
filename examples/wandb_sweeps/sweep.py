#!/usr/bin/env python
"""Runs a W&B sweep."""

import argparse
import functools
import logging
import subprocess
import traceback
import warnings

from typing import List

import wandb


warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def run(argv: List[str]) -> None:
    """A single training run.

    Args:
        argv (List[str]): partial command-line arguments (the command name
            plus CLI arguments passed to this script but not declared below).
    """
    argv = populate_argv(argv)
    process = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)
    for line in process.stderr:
        logging.info(line.rstrip())
    wandb.finish(exit_code=process.wait())


def populate_argv(argv: List[str]) -> List[str]:
    """Populates argv with W&B arguments.

    This copies and mutates to prevent side effects persisting beyond the
    individual run.

    Args:
        argv (List[str]): partial command-line arguments (the command name
            plus CLI arguments passed to this script but not declared below).

    Returns:
        List[str]: complete command-line arguments populated with W&B run
            hyperparameters.
    """
    wandb.init()
    result = argv.copy()
    for key, value in wandb.config.items():
        if value is None:
            continue
        result.append(f"--{key}")
        result.append(f"{value}")
    return result


def main(args: argparse.Namespace, argv: List[str]) -> None:
    # W&B support must be enabled.
    argv = ["yoyodyne-train", "--log_wandb", *argv]
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=functools.partial(run, argv),
            count=args.count,
        )
    except Exception:
        # Exits gracefully, so wandb logs the error.
        logging.fatal(traceback.format_exc())
        wandb.finish(exit_code=1)
        exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_id", required=True, help="ID for the sweep.")
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of runs to perform.",
    )
    # We separate out the args declared above, which control the sweep itself,
    # and all others, which are passed to the subprocess as is.
    args, argv = parser.parse_known_args()
    main(args, argv)
