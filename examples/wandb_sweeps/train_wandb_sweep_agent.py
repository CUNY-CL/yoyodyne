import functools
import traceback

import pytorch_lightning as pl
import wandb

from yoyodyne import (
    train,
    util,
)


class Error(Exception):
    pass


def run_train(args):
    # First get trainer to initialize the wandb run
    trainer = train._get_trainer_from_argparse_args(args)
    pl.seed_everything(args.seed)
    train_set, dev_set = train._get_datasets_from_argparse_args(args)
    index = train.get_index(args.model_dir, args.experiment)
    train_set.index.write(index)
    util.log_info(f"Index: {index}")

    # Model args come from the W&B sweep config.
    kwargs = dict(wandb.config)
    # Anything not specified in the config is taken from the CLI args.
    kwargs.update({k: v for k, v in vars(args).items() if k not in kwargs})
    train_loader, dev_loader = train.get_loaders(
        train_set,
        dev_set,
        args.arch,
        kwargs["batch_size"],
        args.max_source_length,
        args.max_target_length,
    )
    model = train.get_model(train_set, **kwargs)

    # Train and log the best checkpoint.
    best_checkpoint = train.train(
        trainer, model, train_loader, dev_loader, args.train_from
    )
    util.log_info(f"Best checkpoint: {best_checkpoint}")


def main():
    parser = train.get_train_argparse_parser()
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
    # Forces log_wandb to True, so that the PTL trainer logs
    # runtime metrics to W&B
    args.log_wandb
    try:
        wandb.agent(
            args.sweep_id,
            function=functools.partial(run_train, args),
            project=args.experiment,
            count=args.max_num_runs,
        )
    except Exception:
        # Exits gracefully, so wandb logs the error
        util.log_info(traceback.format_exc())
        exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
