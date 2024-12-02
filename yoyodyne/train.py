"""Trains a sequence-to-sequence neural network."""

import argparse
import os
from typing import List, Optional

import lightning
import wandb
from lightning.pytorch import callbacks, loggers

from . import (
    data,
    defaults,
    evaluators,
    metrics,
    models,
    schedulers,
    sizing,
    util,
)


class Error(Exception):
    pass


def _get_loggers(model_dir: str, log_wandb: bool) -> List:
    """Creates the logger(s).

    Args:
        model_dir (str).
        log_wandb (bool).

    Returns:
        List: logger.
    """
    trainer_loggers = [loggers.CSVLogger(model_dir)]
    if log_wandb:
        trainer_loggers.append(loggers.WandbLogger())
        # Logs the path to local artifacts made by PTL.
        wandb.config["local_run_dir"] = trainer_loggers[0].log_dir
    return trainer_loggers


def _get_callbacks(
    num_checkpoints: int = defaults.NUM_CHECKPOINTS,
    checkpoint_metric: str = defaults.CHECKPOINT_METRIC,
    patience: Optional[int] = None,
    patience_metric: str = defaults.PATIENCE_METRIC,
    log_wandb: bool = False,
) -> List[callbacks.Callback]:
    """Creates the callbacks.

    We will reach into the callback metrics list to picks ckp_callback to find
    the best checkpoint path.

    Args:
        num_checkpoints (int, optional): number of checkpoints to save. To
            save one checkpoint per epoch, use `-1`.
        checkpoint_metric (string, optional): validation metric used to
            select checkpoints.
        patience (int, optional): number of epochs with no
            progress (according to `patience_metric`) before triggering
            early stopping.
        patience_metric (string, optional): validation metric used to
            trigger early stopping.
        log_wandb (bool).

    Returns:
        List[callbacks.Callback]: callbacks.
    """
    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="epoch"),
        callbacks.TQDMProgressBar(),
    ]
    # Patience callback if requested.
    if patience is not None:
        metric = metrics.get_metric(patience_metric)
        trainer_callbacks.append(
            callbacks.early_stopping.EarlyStopping(
                mode=metric.mode,
                monitor=metric.monitor,
                patience=patience,
                min_delta=1e-4,
                verbose=True,
            )
        )
    # Checkpointing callback. Ensure that this is the last checkpoint,
    # as the API assumes that.
    metric = metrics.get_metric(checkpoint_metric)
    trainer_callbacks.append(
        callbacks.ModelCheckpoint(
            filename=metric.filename,
            mode=metric.mode,
            monitor=metric.monitor,
            save_top_k=num_checkpoints,
        )
    )
    # Logs the best value for the checkpointing metric.
    if log_wandb:
        wandb.define_metric(metric.monitor, summary=metric.mode)
    return trainer_callbacks


def get_trainer_from_argparse_args(
    args: argparse.Namespace,
) -> lightning.Trainer:
    """Creates the trainer from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        lightning.Trainer.
    """
    return lightning.Trainer.from_argparse_args(
        args,
        callbacks=_get_callbacks(
            args.num_checkpoints,
            args.checkpoint_metric,
            args.patience,
            args.patience_metric,
            args.log_wandb,
        ),
        default_root_dir=args.model_dir,
        enable_checkpointing=True,
        logger=_get_loggers(args.model_dir, args.log_wandb),
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
    separate_features = args.features_col != 0 and args.arch in [
        "hard_attention_gru",
        "hard_attention_lstm",
        "pointer_generator_gru",
        "pointer_generator_lstm",
        "pointer_generator_transformer",
        "transducer_grm",
        "transducer_lstm",
    ]
    # Please pass all arguments by keyword and keep in lexicographic order.
    datamodule = data.DataModule(
        batch_size=args.batch_size,
        features_col=args.features_col,
        features_sep=args.features_sep,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        model_dir=args.model_dir,
        separate_features=separate_features,
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
        sed_params_paths = (
            args.sed_params if args.sed_params else f"{args.model_dir}/sed.pkl"
        )
        expert = models.expert.get_expert(
            datamodule.train_dataloader().dataset,
            datamodule.index,
            epochs=args.oracle_em_epochs,
            oracle_factor=args.oracle_factor,
            sed_params_path=sed_params_paths,
            read_from_file=os.path.isfile(sed_params_paths),
        )
    scheduler_kwargs = schedulers.get_scheduler_kwargs_from_argparse_args(args)
    # We use a separate features encoder if the datamodule has features, and
    # either:
    # - a specific features encoder module is requested (in which case we use
    #   the requested module), or
    # - no specific features encoder module is requested, but the model
    #   requires that we use a separate features encoder (in which case we use
    #   the same type of module as the source encoder).
    features_encoder_cls = (
        models.modules.get_encoder_cls(
            encoder_arch=args.features_encoder_arch,
            model_arch=args.arch,
        )
        if datamodule.has_features
        and (
            args.features_encoder_arch
            or args.arch
            in [
                "hard_attention_gru",
                "hard_attention_lstm",
                "pointer_generator_gru",
                "pointer_generator_lstm",
                "pointer_generator_transformer",
                "transducer_gru",
                "transducer_lstm",
            ]
        )
        else None
    )
    features_vocab_size = (
        datamodule.index.features_vocab_size if datamodule.has_features else 0
    )
    # This makes sure we compute all metrics that'll be needed.
    eval_metrics = args.eval_metric.copy()
    if args.checkpoint_metric != "loss":
        eval_metrics.add(args.checkpoint_metric)
    if args.patience_metric != "loss":
        eval_metrics.add(args.patience_metric)
    # Please pass all arguments by keyword and keep in lexicographic order.
    return model_cls(
        arch=args.arch,
        attention_context=args.attention_context,
        beta1=args.beta1,
        beta2=args.beta2,
        bidirectional=args.bidirectional,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        embedding_size=args.embedding_size,
        encoder_layers=args.encoder_layers,
        enforce_monotonic=args.enforce_monotonic,
        eval_metrics=eval_metrics,
        expert=expert,
        features_attention_heads=args.features_attention_heads,
        features_encoder_cls=features_encoder_cls,
        features_vocab_size=features_vocab_size,
        hidden_size=args.hidden_size,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        scheduler_kwargs=scheduler_kwargs,
        source_attention_heads=args.source_attention_heads,
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
    data.add_argparse_args(parser)
    evaluators.add_argparse_args(parser)
    metrics.add_argparse_args(parser)
    models.add_argparse_args(parser)
    models.expert.add_argparse_args(parser)
    models.modules.add_argparse_args(parser)
    models.BaseModel.add_argparse_args(parser)
    models.HardAttentionRNNModel.add_argparse_args(parser)
    models.RNNModel.add_argparse_args(parser)
    models.TransformerModel.add_argparse_args(parser)
    schedulers.add_argparse_args(parser)
    sizing.add_argparse_args(parser)
    # Trainer arguments.
    # Among the things this adds, the following are likely to be useful:
    # --accelerator ("gpu" for GPU)
    # --check_val_every_n_epoch
    # --devices (for multiple device support)
    # --gradient_clip_val
    # --max_epochs
    # --min_epochs
    # --max_steps
    # --min_steps
    # --max_time
    lightning.Trainer.add_argparse_args(parser)


def main() -> None:
    """Trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_argparse_args(parser)
    args = parser.parse_args()
    util.log_arguments(args)
    train(args)


if __name__ == "__main__":
    main()
