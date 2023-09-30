#!/usr/bin/env python
"""Runs the sweep itself."""

import argparse
import functools
import traceback
import warnings

import pytorch_lightning as pl
import wandb

from yoyodyne import train, util


warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


class Error(Exception):
    pass


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
    pl.seed_everything(args.seed)
    trainer = train.get_trainer_from_argparse_args(args)
    datamodule = train.get_datamodule_from_argparse_args(args)
    model = train.get_model_from_argparse_args(args, datamodule)
    # Trains and logs the best checkpoint.
    best_checkpoint = train.train(trainer, model, datamodule, args.train_from)
    util.log_info(f"Best checkpoint: {best_checkpoint}")


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
