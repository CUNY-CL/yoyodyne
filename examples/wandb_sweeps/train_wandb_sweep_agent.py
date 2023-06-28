import functools
import os
import traceback

import pytorch_lightning as pl
import wandb

from yoyodyne import (
    collators,
    dataconfig,
    defaults,
    models,
    predict,
    schedulers,
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

    if args.predict:
        # Deletes loaders to free those processes.
        del train_loader
        del dev_loader

        # Reloads the dev set.
        loader = predict.get_loader(
            dataset=dev_set,
            arch=args.arch,
            batch_size=64,
            max_source_length=defaults.MAX_SOURCE_LENGTH,
            max_target_length=defaults.MAX_TARGET_LENGTH,
        )
        # Reloads the model.
        model = predict.get_model(
            arch=args.arch,
            has_features=train_set.config.has_features,
            checkpoint=best_checkpoint,
        )
        # Hack to get the results dir for this version from the checkpoint path
        output = best_checkpoint.split("checkpoints")[0]
        output = os.path.join(output, "predictions.tsv")
        util.log_info(f"Writing predictions to {output}")
        # Writes dev predictions with the best model.
        predict.predict(trainer, model, loader, output)


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
    parser.add_argument(
        "--predict",
        action="store_true",
        default=True,
        help="Whether or not to write a predictions.tsv file with the"
        " best model on the development set.",
    )
    parser.add_argument(
        "--no_predict",
        action="store_false",
        dest="predict",
    )

    dataconfig.DataConfig.add_argparse_args(parser)
    # Collator arguments.
    collators.Collator.add_argparse_args(parser)
    # Architecture arguments.
    models.add_argparse_args(parser)
    # Scheduler-specific arguments.
    schedulers.add_argparse_args(parser)
    # Architecture-specific arguments.
    models.BaseEncoderDecoder.add_argparse_args(parser)
    models.LSTMEncoderDecoder.add_argparse_args(parser)
    models.TransformerEncoderDecoder.add_argparse_args(parser)
    models.expert.add_argparse_args(parser)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if not args.log_wandb:
        msg = "'--log_wandb' is a required arg for training a "
        "wandb sweep agent."
        raise Error(msg)

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
