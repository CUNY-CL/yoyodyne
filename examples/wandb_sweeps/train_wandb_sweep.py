#!/usr/bin/env python
"""Runs the sweep itself."""

import argparse
import functools
import traceback
import warnings

import torch
import wandb

from yoyodyne import train, util


warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def train_sweep(args: argparse.Namespace) -> None:
    """Runs a single training run.

    The wandb config data used here comes from the environment.

    Args:
        args (argparse.Namespace).
    """
    wandb.init()
    # Model arguments come from the wandb sweep config and override any
    # conflicting arguments passed via the CLI.
    for key, value in dict(wandb.config).items():
        if key in args:
            util.log_info(f"Overriding CLI argument: {key}")
        setattr(args, key, value)
    try:
        train.train(args)  # Ignoring return value.
    except RuntimeError as error:
        # TODO: consider specializing this further if a solution to
        # https://github.com/pytorch/pytorch/issues/48365 is accepted.
        util.log_info(f"Runtime error: {error!s}")
    finally:
        # Clears the CUDA cache.
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    train.add_argparse_args(parser)
    parser.add_argument(
        "--sweep_id",
        required=True,
        help="ID for the sweep to run the agent in.",
    )
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument(
        "--count",
        type=int,
        help="The max number of runs for this agent",
    )
    args = parser.parse_args()
    # Forces log_wandb to True, so that the PTL trainer logs runtime metrics
    # to wandb.
    args.log_wandb = True
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=functools.partial(train_sweep, args),
            count=args.count,
        )
    except Exception:
        # Exits gracefully, so wandb logs the error.
        util.log_info(traceback.format_exc())
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
