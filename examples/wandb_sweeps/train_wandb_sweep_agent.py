#!/usr/bin/env python

import argparse
import functools
import traceback

import pytorch_lightning as pl
import wandb

from yoyodyne import train, util


class Error(Exception):
    pass


def train_sweep(args: argparse.Namespace) -> None:
    """Runs a single training run.

    The wandb config data used here comes from the environment.

    Args:
        args (argparse.Namespace).
    """
    # Gets trainer to initialize the wandb run.
    trainer = train.get_trainer_from_argparse_args(args)
    pl.seed_everything(args.seed)
    train_set, dev_set = train.get_datasets_from_argparse_args(args)
    index = train.get_index(args.model_dir, args.experiment)
    train_set.index.write(index)
    util.log_info(f"Index: {index}")
    # Model arguments come from the wandb sweep config and override any
    # conflicting arguments passed via the CLI.
    for key, value in dict(wandb.config).items():
        if key in args:
            util.log_info(f"Overridding CLI argument: {key}")
        setattr(args, key, value)
    train_loader, dev_loader = train.get_loaders(
        train_set,
        dev_set,
        args.arch,
        args.batch_size,
        args.max_source_length,
        args.max_target_length,
    )
    model = train.get_model_from_argparse_args(train_set, args)
    # Trains and log the best checkpoint.
    best_checkpoint = train.train(
        trainer, model, train_loader, dev_loader, args.train_from
    )
    util.log_info(f"Best checkpoint: {best_checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    train.add_argparse_args(parser)
    parser.add_argument(
        "--sweep_id",
        help="ID for the sweep to run the agent in.",
    )
    parser.add_argument(
        "--max_num_runs",
        type=int,
        default=1,
        help="Max number of runs this agent should train.",
    )
    args = parser.parse_args()
    # Forces log_wandb to True, so that the PTL trainer logs runtime metrics
    # to wandb.
    args.log_wandb = True
    try:
        wandb.agent(
            args.sweep_id,
            function=functools.partial(train_sweep, args),
            project=args.experiment,
            count=args.max_num_runs,
        )
    except Exception:
        # Exits gracefully, so wandb logs the error.
        util.log_info(traceback.format_exc())
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
