"""Trains a sequence-to-sequence neural network."""

import argparse
from typing import List, Optional

import pytorch_lightning as pl
import wandb
from pytorch_lightning import callbacks, loggers

from . import data, defaults, evaluators, metrics, models, schedulers, util


class Error(Exception):
    pass


def _get_logger(experiment: str, model_dir: str, log_wandb: bool) -> List:
    """Creates the logger(s).

    Args:
        experiment (str).
        model_dir (str).
        log_wandb (bool).

    Returns:
        List: logger.
    """
    trainer_logger = [loggers.CSVLogger(model_dir, name=experiment)]
    if log_wandb:
        trainer_logger.append(loggers.WandbLogger(project=experiment))
        # Tells PTL to log the best validation accuracy.
        wandb.define_metric("val_accuracy", summary="max")
        # Logs the path to local artifacts made by PTL.
        wandb.config["local_run_dir"] = trainer_logger[0].log_dir
    return trainer_logger


def _get_callbacks(
    num_checkpoints: int = defaults.NUM_CHECKPOINTS,
    checkpoint_metric: str = defaults.CHECKPOINT_METRIC,
    patience: Optional[int] = None,
    patience_metric: str = defaults.PATIENCE_METRIC,
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

    Returns:
        List[callbacks.Callback]: callbacks.
    """
    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="epoch"),
        callbacks.TQDMProgressBar(),
    ]
    # Patience callback if requested.
    if patience is not None:
        metric = metrics.ValidationMetric(patience_metric)
        trainer_callbacks.append(
            callbacks.early_stopping.EarlyStopping(
                mode=metric.mode,
                monitor=metric.monitor,
                patience=patience,
            )
        )
    # Checkpointing callback. Ensure that this is the last checkpoint,
    # as the API assumes that.
    metric = metrics.ValidationMetric(checkpoint_metric)
    trainer_callbacks.append(
        callbacks.ModelCheckpoint(
            filename=metric.filename,
            mode=metric.mode,
            monitor=metric.monitor,
            save_top_k=num_checkpoints,
        )
    )
    return trainer_callbacks


def get_trainer_from_argparse_args(
    args: argparse.Namespace,
) -> pl.Trainer:
    """Creates the trainer from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        pl.Trainer.
    """
    return pl.Trainer.from_argparse_args(
        args,
        callbacks=_get_callbacks(
            args.num_checkpoints,
            args.checkpoint_metric,
            args.patience,
            args.patience_metric,
        ),
        default_root_dir=args.model_dir,
        enable_checkpointing=True,
        logger=_get_logger(args.experiment, args.model_dir, args.log_wandb),
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
        "pointer_generator_lstm",
        "pointer_generator_transformer",
        "transducer",
    ]
    datamodule = data.DataModule(
        train=args.train,
        val=args.val,
        batch_size=args.batch_size,
        source_col=args.source_col,
        features_col=args.features_col,
        target_col=args.target_col,
        source_sep=args.source_sep,
        features_sep=args.features_sep,
        target_sep=args.target_sep,
        separate_features=separate_features,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    if not datamodule.has_target:
        raise Error("No target column specified")
    datamodule.index.write(args.model_dir, args.experiment)
    datamodule.log_vocabularies()
    return datamodule


def get_model_from_argparse_args(
    args: argparse.Namespace,
    datamodule: data.DataModule,
) -> models.BaseEncoderDecoder:
    """Creates the model.

    Args:
        args (argparse.Namespace).
        datamodule (data.DataModule).

    Returns:
        models.BaseEncoderDecoder.
    """
    model_cls = models.get_model_cls(args.arch)
    source_encoder_cls = models.modules.get_encoder_cls(
        encoder_arch=args.source_encoder_arch, model_arch=args.arch
    )
    expert = (
        models.expert.get_expert(
            datamodule.train_dataloader().dataset,
            epochs=args.oracle_em_epochs,
            oracle_factor=args.oracle_factor,
            sed_params_path=args.sed_params,
        )
        if args.arch in ["transducer"]
        else None
    )
    scheduler_kwargs = schedulers.get_scheduler_kwargs_from_argparse_args(args)
    separate_features = datamodule.has_features and args.arch in [
        "pointer_generator_lstm",
        "pointer_generator_transformer",
        "transducer",
    ]
    features_encoder_cls = (
        models.modules.get_encoder_cls(
            encoder_arch=args.features_encoder_arch, model_arch=args.arch
        )
        if separate_features and args.features_encoder_arch
        else None
    )
    features_vocab_size = (
        datamodule.index.features_vocab_size if datamodule.has_features else 0
    )
    source_vocab_size = (
        datamodule.index.source_vocab_size + features_vocab_size
        if not separate_features
        else datamodule.index.source_vocab_size
    )
    # Please pass all arguments by keyword and keep in lexicographic order.
    return model_cls(
        arch=args.arch,
        source_attention_heads=args.source_attention_heads,
        features_attention_heads=args.features_attention_heads,
        beta1=args.beta1,
        beta2=args.beta2,
        bidirectional=args.bidirectional,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        embedding_size=args.embedding_size,
        encoder_layers=args.encoder_layers,
        end_idx=datamodule.index.end_idx,
        eval_metrics=args.eval_metric,
        expert=expert,
        features_encoder_cls=features_encoder_cls,
        features_vocab_size=features_vocab_size,
        hidden_size=args.hidden_size,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        optimizer=args.optimizer,
        output_size=datamodule.index.target_vocab_size,
        pad_idx=datamodule.index.pad_idx,
        scheduler=args.scheduler,
        scheduler_kwargs=scheduler_kwargs,
        source_encoder_cls=source_encoder_cls,
        source_vocab_size=source_vocab_size,
        start_idx=datamodule.index.start_idx,
        target_vocab_size=datamodule.index.target_vocab_size,
    )


def train(
    trainer: pl.Trainer,
    model: models.BaseEncoderDecoder,
    datamodule: data.DataModule,
    train_from: Optional[str] = None,
) -> str:
    """Trains the model.

    Args:
         trainer (pl.Trainer).
         model (models.BaseEncoderDecoder).
         datamodule (data.DataModule).
         train_from (str, optional): if specified, starts training from this
            checkpoint.

    Returns:
        str: path to best checkpoint.
    """
    trainer.fit(model, datamodule, ckpt_path=train_from)
    # The API assumes that the last callback is the checkpointing one, and
    # _get_callbacks must enforce this.
    ckp_callback = trainer.callbacks[-1]
    assert type(ckp_callback) is callbacks.ModelCheckpoint
    return ckp_callback.best_model_path


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
        "--experiment", required=True, help="Name of experiment."
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
        help="Path to ckpt checkpoint to resume training from. "
        "Default: not enabled.",
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
        "--checkpoint_metric",
        choices=["accuracy", "loss", "ser"],
        default=defaults.CHECKPOINT_METRIC,
        help="Selects checkpoints to maximize validation `accuracy`, "
        "or to minimize validation `loss` or `ser`. "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Number of epochs with no progress (according to "
        "`--patience_metric`) before triggering early stopping. "
        "Default: not enabled.",
    )
    parser.add_argument(
        "--patience_metric",
        choices=["accuracy", "loss", "ser"],
        default=defaults.PATIENCE_METRIC,
        help="Stops early when validation `accuracy` stops increasing or "
        "when validation `loss` or `ser` stops decreasing. "
        "Default: %(default)s.",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        default=defaults.LOG_WANDB,
        help="Use Weights & Biases logging (log-in required). Default: True.",
    )
    parser.add_argument(
        "--no_log_wandb",
        action="store_false",
        dest="log_wandb",
    )
    # Data arguments.
    data.add_argparse_args(parser)
    # Architecture arguments.
    models.add_argparse_args(parser)
    models.modules.add_argparse_args(parser)
    # Scheduler-specific arguments.
    schedulers.add_argparse_args(parser)
    # Evaluation-specific arguments.
    evaluators.add_argparse_args(parser)
    # Architecture-specific arguments.
    models.BaseEncoderDecoder.add_argparse_args(parser)
    models.LSTMEncoderDecoder.add_argparse_args(parser)
    models.TransformerEncoderDecoder.add_argparse_args(parser)
    # models.modules.BaseEncoder.add_argparse_args(parser)
    models.expert.add_argparse_args(parser)
    # Trainer arguments.
    # Among the things this adds, the following are likely to be useful:
    # --auto_lr_find
    # --accelerator ("gpu" for GPU)
    # --check_val_every_n_epoch
    # --devices (for multiple device support)
    # --gradient_clip_val
    # --max_epochs
    # --min_epochs
    # --max_steps
    # --min_steps
    # --max_time
    pl.Trainer.add_argparse_args(parser)


def main() -> None:
    """Trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_argparse_args(parser)
    args = parser.parse_args()
    util.log_arguments(args)
    pl.seed_everything(args.seed)
    trainer = get_trainer_from_argparse_args(args)
    datamodule = get_datamodule_from_argparse_args(args)
    model = get_model_from_argparse_args(args, datamodule)
    # Logs number of model parameters.
    if args.log_wandb:
        wandb.config["n_model_params"] = sum(
            p.numel() for p in model.parameters()
        )
    # Tuning options. Batch autoscaling is unsupported; LR tuning logs the
    # suggested value and then exits.
    if args.auto_scale_batch_size:
        raise Error("Batch auto-scaling is not supported")
        return
    if args.auto_lr_find:
        result = trainer.tuner.lr_find(model, datamodule=datamodule)
        util.log_info(f"Best initial LR: {result.suggestion():.8f}")
        return
    # Otherwise, train and log the best checkpoint.
    best_checkpoint = train(trainer, model, datamodule, args.train_from)
    util.log_info(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
