"""Trains a sequence-to-sequence neural network."""

import argparse

import lightning
import wandb
from lightning.pytorch import callbacks, loggers

from . import (
    data,
    defaults,
    metrics,
    models,
    schedulers,
    sizing,
    util,
)


class Error(Exception):
    pass


def get_trainer_from_argparse_args(
    args: argparse.Namespace,
) -> lightning.Trainer:
    """Creates the trainer from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        lightning.Trainer.
    """
    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="epoch"),
        callbacks.TQDMProgressBar(),
    ]
    # Patience callback if requested.
    if args.patience is not None:
        metric = metrics.get_metric(args.patience_metric)
        trainer_callbacks.append(
            callbacks.early_stopping.EarlyStopping(
                mode=metric.mode,
                monitor=metric.monitor,
                patience=args.patience,
                min_delta=1e-4,
                verbose=True,
            )
        )
    # Checkpointing callback. Ensure that this is the last checkpoint,
    # as the API assumes that.
    metric = metrics.get_metric(args.checkpoint_metric)
    trainer_callbacks.append(
        callbacks.ModelCheckpoint(
            filename=metric.filename,
            mode=metric.mode,
            monitor=metric.monitor,
            save_top_k=args.num_checkpoints,
        )
    )
    trainer_loggers = [loggers.CSVLogger(args.model_dir)]
    # Logs the best value for the checkpointing metric.
    if args.log_wandb:
        trainer_loggers.append(
            loggers.WandbLogger(
                metric.filename, monitor=metric.monitor, mode=metric.mode
            )
        )
    return lightning.Trainer.from_argparse_args(
        args,
        default_root_dir=args.model_dir,
        enable_checkpointing=True,
        callbacks=trainer_callbacks,
        logger=trainer_loggers,
    )


def get_datamodule_from_argparse_args(
    args: argparse.Namespace,
) -> data.DataModule:
    """Creates the datamodule from CLI arguments.

    Args:
        args (Argparse.Namespace).

    Returns:
        data.DataModule.
    """
    # Please pass all arguments by keyword and keep in lexicographic order.
    datamodule = data.DataModule(
        batch_size=args.batch_size,
        features_col=args.features_col,
        features_sep=args.features_sep,
        max_features_length=args.max_features_length,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        model_dir=args.model_dir,
        source_col=args.source_col,
        source_sep=args.source_sep,
        target_col=args.target_col,
        target_sep=args.target_sep,
        tie_embeddings=args.tie_embeddings,
        train=args.train,
        val=args.val,
    )
    if not datamodule.has_target:
        raise Error("No target column specified")
    datamodule.log_vocabularies()
    return datamodule


def get_model_from_argparse_args(
    args: argparse.Namespace,
    datamodule: data.DataModule,
) -> models.BaseModel:
    """Creates the model.

    Args:
        args (argparse.Namespace).
        datamodule (data.DataModule).

    Returns:
        models.BaseModel.
    """
    model_cls = models.get_model_cls(args.arch)
    # TODO(#156): add callback interface to check this.
    if (
        args.arch
        in [
            "pointer_generator_rnn",
            "pointer_generator_transformer",
            "transducer_gru",
            "transducer_lstm",
        ]
        and not args.tie_embeddings
    ):
        raise Error(
            f"--tie_embeddings disabled, but --arch {args.arch} requires "
            "it to be enabled"
        )
    source_encoder_cls = models.modules.get_encoder_cls(
        encoder_arch=args.source_encoder_arch, model_arch=args.arch
    )
    # Instantiates expert, if needed.
    expert = None
    if args.arch in ["transducer_gru", "transducer_lstm"]:
        expert = models.expert.get_expert(
            datamodule.index, args.sed_params, args.oracle_factor
        )
    scheduler_kwargs = schedulers.get_scheduler_kwargs_from_argparse_args(args)
    features_encoder_cls = (
        models.modules.get_encoder_cls(
            encoder_arch=args.features_encoder_arch,
            model_arch=args.arch,
        )
        if datamodule.has_features
        else None
    )
    # Please pass all arguments by keyword and keep in lexicographic order.
    return model_cls(
        arch=args.arch,
        attention_context=args.attention_context,
        attention_heads=args.attention_heads,
        beta1=args.beta1,
        beta2=args.beta2,
        bidirectional=args.bidirectional,
        compute_accuracy=metrics.compute_metric(args, "accuracy"),
        compute_ser=metrics.compute_metric(args, "ser"),
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        embedding_size=args.embedding_size,
        encoder_layers=args.encoder_layers,
        enforce_monotonic=args.enforce_monotonic,
        expert=expert,
        features_encoder_cls=features_encoder_cls,
        hidden_size=args.hidden_size,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        max_features_length=args.max_features_length,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        scheduler_kwargs=scheduler_kwargs,
        source_encoder_cls=source_encoder_cls,
        target_vocab_size=(
            len(expert.actions)
            if expert is not None
            else datamodule.index.target_vocab_size
        ),
        vocab_size=(
            datamodule.index.vocab_size + len(expert.actions)
            if expert is not None
            else datamodule.index.vocab_size
        ),
    )


def train(args: argparse.Namespace) -> str:
    """Trains the model.

    Args:
        args (argparse.Namespace).

    Returns:
        str: path to best checkpoint.
    """
    if args.log_wandb:
        wandb.init()
    lightning.seed_everything(args.seed)
    trainer = get_trainer_from_argparse_args(args)
    datamodule = get_datamodule_from_argparse_args(args)
    model = get_model_from_argparse_args(args, datamodule)
    if args.log_wandb:
        wandb.config["num_parameters"] = model.num_parameters
    if args.find_batch_size:
        sizing.find_batch_size(
            args.find_batch_size,
            trainer,
            model,
            datamodule,
            steps_per_trial=args.find_batch_size_steps_per_trial,
        )
    trainer.fit(model, datamodule, ckpt_path=args.train_from)
    # The API assumes that the last callback is the checkpointing one, and
    # _get_callbacks must enforce this.
    checkpoint_callback = trainer.callbacks[-1]
    assert checkpoint_callback.best_model_path
    util.log_info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds training arguments to parser.

    Args:
        argparse.ArgumentParser.
    """
    data.add_argparse_args(parser)
    metrics.add_argparse_args(parser)
    models.add_argparse_args(parser)
    schedulers.add_argparse_args(parser)
    sizing.add_argparse_args(parser)
    lightning.Trainer.add_argparse_args(parser)
    # Path arguments.
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to output model directory.",
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to input training data TSV.",
    )
    parser.add_argument(
        "--val",
        required=True,
        help="Path to input validation data TSV.",
    )
    parser.add_argument(
        "--train_from",
        help="Path to checkpoint used to resume training.",
    )
    # Other training arguments.
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=defaults.NUM_CHECKPOINTS,
        help="Number of checkpoints to save. To save one checkpoint per "
        "epoch, use `-1`. Default: %(default)s.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Number of epochs with no progress (according to "
        "`--patience_metric`) before triggering early stopping.",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        default=defaults.LOG_WANDB,
        help="Use Weights & Biases logging (log-in required). "
        "Default: not enabled.",
    )
    parser.add_argument(
        "--no_log_wandb",
        action="store_false",
        dest="log_wandb",
    )


def main() -> None:
    """Trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_argparse_args(parser)
    args = parser.parse_args()
    util.log_arguments(args)
    train(args)


if __name__ == "__main__":
    main()
