"""Trains a sequence-to-sequence neural network."""

import argparse
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers
from torch.utils import data
import wandb

from . import (
    collators,
    dataconfig,
    datasets,
    defaults,
    models,
    schedulers,
    util,
)


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
        trainer_logger.append(
            loggers.WandbLogger(project=experiment, log_model="all")
        )
        # Tells PTL to log best validation acc
        wandb.define_metric("val_accuracy", summary="max")
        # Logs the path to local artifacts made by PTL.
        wandb.config.update({"local_run_dir": trainer_logger[0].log_dir})

    return trainer_logger


def _get_callbacks(save_top_k: int, patience: Optional[int] = None) -> List:
    """Creates the callbacks.

    We will reach into the callback metrics list to picks ckp_callback to find
    the best checkpoint path.

    Args:
        save_top_k (int).
        patience (int, optional).

    Returns:
        List: callbacks.
    """
    trainer_callbacks = [
        callbacks.ModelCheckpoint(
            save_top_k=save_top_k,
            monitor="val_accuracy",
            mode="max",
            filename="model-{epoch:02d}-{val_accuracy:.2f}",
        ),
        callbacks.LearningRateMonitor(logging_interval="epoch"),
        callbacks.TQDMProgressBar(),
    ]
    if patience is not None:
        trainer_callbacks.append(
            callbacks.early_stopping.EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.0,
                patience=patience,
                verbose=False,
                mode="max",
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
        callbacks=_get_callbacks(args.save_top_k, args.patience),
        default_root_dir=args.model_dir,
        enable_checkpointing=True,
        logger=_get_logger(args.experiment, args.model_dir, args.log_wandb),
    )


def get_datasets_from_argparse_args(
    args: argparse.Namespace,
) -> Tuple[datasets.BaseDataset, datasets.BaseDataset]:
    """Creates the datasets from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        Tuple[datasets.BaseDataset, datasets.BaseDataset]: the training and
            development datasets.
    """
    config = dataconfig.DataConfig.from_argparse_args(args)
    if config.target_col == 0:
        raise Error("target_col must be specified for training")
    train_set = datasets.get_dataset(args.train, config)
    dev_set = datasets.get_dataset(args.dev, config, train_set.index)
    util.log_info(f"Source vocabulary: {train_set.index.source_map.pprint()}")
    if train_set.has_features:
        util.log_info(
            f"Feature vocabulary: {train_set.index.features_map.pprint()}"
        )
    util.log_info(f"Target vocabulary: {train_set.index.target_map.pprint()}")
    return train_set, dev_set


def get_loaders(
    train_set: datasets.BaseDataset,
    dev_set: datasets.BaseDataset,
    arch: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
) -> Tuple[data.DataLoader, data.DataLoader]:
    """Creates the loaders.

    Args:
        train_set (datasets.BaseDataset).
        dev_set (datasets.BaseDataset).
        arch (str).
        batch_size (int).
        max_source_length (int).
        max_target_length (int).

    Returns:
        Tuple[data.DataLoader, data.DataLoader]: the training and development
            loaders.
    """
    collator = collators.Collator(
        train_set,
        arch,
        max_source_length,
        max_target_length,
    )
    train_loader = data.DataLoader(
        train_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # Our data loading is simple.
    )
    dev_loader = data.DataLoader(
        dev_set,
        collate_fn=collator,
        batch_size=2 * batch_size,  # Because we're not collecting gradients.
        num_workers=1,
    )
    return train_loader, dev_loader


def get_model_from_argparse_args(
    train_set: datasets.BaseDataset,
    args: argparse.Namespace,
) -> models.BaseEncoderDecoder:
    """Creates the model.

    Args:
        train_set (datasets.BaseDataset).
        args (argparse.Namespace).

    Returns:
        models.BaseEncoderDecoder.
    """
    model_cls = models.get_model_cls(args.arch, train_set.has_features)
    expert = (
        models.expert.get_expert(
            train_set,
            epochs=args.oracle_em_epochs,
            oracle_factor=args.oracle_factor,
            sed_params_path=args.sed_params,
        )
        if args.arch in ["transducer"]
        else None
    )
    scheduler_kwargs = schedulers.get_scheduler_kwargs_from_argparse_args(args)
    separate_features = train_set.has_features and args.arch in [
        "pointer_generator_lstm",
        "transducer",
    ]
    features_vocab_size = (
        train_set.index.features_vocab_size if train_set.has_features else 0
    )
    source_vocab_size = (
        train_set.index.source_vocab_size + features_vocab_size
        if not separate_features
        else train_set.index.source_vocab_size
    )
    # Please pass all arguments by keyword and keep in lexicographic order.
    return model_cls(
        arch=args.arch,
        attention_heads=args.attention_heads,
        beta1=args.beta1,
        beta2=args.beta2,
        bidirectional=args.bidirectional,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        embedding_size=args.embedding_size,
        encoder_layers=args.encoder_layers,
        end_idx=train_set.index.end_idx,
        expert=expert,
        features_vocab_size=features_vocab_size,
        hidden_size=args.hidden_size,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        optimizer=args.optimizer,
        pad_idx=train_set.index.pad_idx,
        scheduler=args.scheduler,
        scheduler_kwargs=scheduler_kwargs,
        source_vocab_size=source_vocab_size,
        start_idx=train_set.index.start_idx,
        target_vocab_size=train_set.index.target_vocab_size,
    )


def train(
    trainer: pl.Trainer,
    model: models.BaseEncoderDecoder,
    train_loader: data.DataLoader,
    dev_loader: data.DataLoader,
    train_from: Optional[str] = None,
) -> str:
    """Trains the model.

    Args:
         trainer (pl.Trainer).
         model (models.BaseEncoderDecoder).
         train_loader (data.DataLoader).
         dev_loader (data.DataLoader).
         train_from (str, optional): if specified, starts training from this
            checkpoint.

    Returns:
        str: path to best checkpoint.
    """
    trainer.fit(model, train_loader, dev_loader, ckpt_path=train_from)
    ckp_callback = trainer.callbacks[-1]
    # TODO: feels flimsy.
    assert type(ckp_callback) is callbacks.ModelCheckpoint
    return ckp_callback.best_model_path


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds training arguments to parser.

    Args:
        argparse.ArgumentParser.
    """
    parser.add_argument(
        "--experiment", required=True, help="Name of experiment"
    )
    # Path arguments.
    parser.add_argument(
        "--train",
        required=True,
        help="Path to input training data TSV",
    )
    parser.add_argument(
        "--dev",
        required=True,
        help="Path to input development data TSV",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to output model directory",
    )
    parser.add_argument(
        "--train_from",
        help="Path to ckpt checkpoint to resume training from",
    )
    # Other training arguments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=defaults.BATCH_SIZE,
        help="Batch size. Default: %(default)s.",
    )
    parser.add_argument(
        "--patience", type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=defaults.SAVE_TOP_K,
        help="Number of checkpoints to save. Default: %(default)s.",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
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
    # Data configuration arguments.
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
    train_set, dev_set = get_datasets_from_argparse_args(args)
    index = train_set.index.index_path(args.model_dir, args.experiment)
    train_set.index.write(index)
    util.log_info(f"Index: {index}")
    train_loader, dev_loader = get_loaders(
        train_set,
        dev_set,
        args.arch,
        args.batch_size,
        args.max_source_length,
        args.max_target_length,
    )
    model = get_model_from_argparse_args(train_set, args)
    # Tuning options. Batch autoscaling is unsupported; LR tuning logs the
    # suggested value and then exits.
    if args.auto_scale_batch_size:
        raise Error("Batch auto-scaling is not supported")
        return
    if args.auto_lr_find:
        result = trainer.tuner.lr_find(model, train_loader, dev_loader)
        util.log_info(f"Best initial LR: {result.suggestion():.8f}")
        return
    # Otherwise, train and log the best checkpoint.
    best_checkpoint = train(
        trainer, model, train_loader, dev_loader, args.train_from
    )
    util.log_info(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
