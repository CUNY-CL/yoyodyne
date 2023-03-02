"""Trains a sequence-to-sequence neural network."""

import argparse
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers
from torch.utils import data

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


def _get_logger(experiment: str, model_dir: str, wandb: bool) -> List:
    """Creates the logger(s).

    Args:
        experiment (str).
        model_dir (str).
        wandb (bool).

    Returns:
        List: logger.
    """
    trainer_logger = [loggers.CSVLogger(model_dir, name=experiment)]
    if wandb:
        trainer_logger.append(
            loggers.WandbLogger(project=experiment, log_model="all")
        )
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


def get_trainer(
    experiment: str,
    model_dir: str,
    save_top_k: int,
    patience: Optional[int] = None,
    wandb: bool = True,
    **kwargs,
) -> pl.Trainer:
    """Creates the trainer.

    Args:
        experiment (str).
        model_dir (str).
        patience (int, optional).
        save_top_k (int).
        wandb (bool).
        **kwargs: passed to the trainer.

    Returns:
        pl.Trainer.
    """
    return pl.Trainer(
        callbacks=_get_callbacks(save_top_k, patience),
        default_root_dir=model_dir,
        enable_checkpointing=True,
        logger=_get_logger(experiment, model_dir, wandb),
        **kwargs,
    )


def _get_trainer_from_argparse_args(
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
        logger=_get_logger(args.experiment, args.model_dir, args.wandb),
    )


def get_datasets(
    train: str,
    dev: str,
    config: dataconfig.DataConfig,
) -> Tuple[datasets.BaseDataset, datasets.BaseDataset]:
    """Creates the datasets.

    Args:
        train (str).
        dev (str).
        config (dataconfig.DataConfig)

    Returns:
        Tuple[datasets.BaseDataset, datasets.BaseDataset]: the training and
            development datasets.
    """
    if config.target_col == 0:
        raise Error("target_col must be specified for training")
    train_set = datasets.get_dataset(train, config)
    dev_set = datasets.get_dataset(dev, config, train_set.index)
    util.log_info(f"Source vocabulary: {train_set.index.source_map.pprint()}")
    util.log_info(f"Target vocabulary: {train_set.index.target_map.pprint()}")
    return train_set, dev_set


def _get_datasets_from_argparse_args(
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
    return get_datasets(args.train, args.dev, config)


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
        train_set.index.pad_idx,
        train_set.config,
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


def get_model(
    # Data arguments.
    train_set: datasets.BaseDataset,
    *,
    # Architecture arguments.
    arch: str = defaults.ARCH,
    attention_heads: int = defaults.ATTENTION_HEADS,
    bidirectional: bool = defaults.BIDIRECTIONAL,
    decoder_layers: int = defaults.DECODER_LAYERS,
    embedding_size: int = defaults.EMBEDDING_SIZE,
    encoder_layers: int = defaults.ENCODER_LAYERS,
    hidden_size: int = defaults.HIDDEN_SIZE,
    max_source_length: int = defaults.MAX_SOURCE_LENGTH,
    max_target_length: int = defaults.MAX_TARGET_LENGTH,
    # Training arguments.
    batch_size: int = defaults.BATCH_SIZE,
    beta1: float = defaults.BETA1,
    beta2: float = defaults.BETA2,
    dropout: float = defaults.DROPOUT,
    learning_rate: float = defaults.LEARNING_RATE,
    oracle_em_epochs: int = defaults.ORACLE_EM_EPOCHS,
    oracle_factor: int = defaults.ORACLE_FACTOR,
    optimizer: str = defaults.OPTIMIZER,
    sed_params: Optional[str] = None,
    scheduler: Optional[str] = None,
    **kwargs,
) -> models.BaseEncoderDecoder:
    """Creates the model.

    Args:
        train_set (datasets.BaseDataset)
        arch (str).
        attention_heads (int).
        bidirectional (bool).
        decoder_layers (int).
        embedding_size (int).
        encoder_layers (int).
        hidden_size (int).
        max_target_length (int).
        max_source_length (int).
        batch_size (int).
        beta1 (float).
        beta2 (float).
        batch_size (int).
        dropout (float).
        learning_rate (float).
        oracle_em_epochs (int).
        oracle_factor (int).
        optimizer (str).
        sed_params (str, optional).
        scheduler (str, optional).
        **kwargs

    Returns:
        models.BaseEncoderDecoder.
    """
    model_cls = models.get_model_cls(arch, train_set.config.has_features)
    expert = (
        models.expert.get_expert(
            train_set,
            epochs=oracle_em_epochs,
            oracle_factor=oracle_factor,
            sed_params_path=sed_params,
        )
        if arch in ["transducer"]
        else None
    )
    scheduler_kwargs = schedulers.get_scheduler_kwargs_from_argparse_args(
        **kwargs
    )
    # Please pass all arguments by keyword and keep in lexicographic order.
    return model_cls(
        arch=arch,
        attention_heads=attention_heads,
        beta1=beta1,
        beta2=beta2,
        bidirectional=bidirectional,
        decoder_layers=decoder_layers,
        dropout=dropout,
        embedding_size=embedding_size,
        encoder_layers=encoder_layers,
        end_idx=train_set.index.end_idx,
        expert=expert,
        features_vocab_size=getattr(
            train_set.index, "features_vocab_size", -1
        ),
        features_idx=getattr(train_set.index, "features_idx", -1),
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        optimizer=optimizer,
        output_size=train_set.index.target_vocab_size,
        pad_idx=train_set.index.pad_idx,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        start_idx=train_set.index.start_idx,
        vocab_size=train_set.index.source_vocab_size,
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


def get_index(model_dir: str, experiment: str) -> str:
    """Computes the index path.

    Args:
        model_dir (str).
        experiment (str).

    Returns:
        str.
    """
    return f"{model_dir}/{experiment}/index.pkl"


def main() -> None:
    """Trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--wandb",
        action="store_true",
        default=defaults.WANDB,
        help="Use Weights & Biases logging (log-in required). Default: True.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_false",
        dest="wandb",
    )
    args = parser.parse_args()
    util.log_arguments(args)
    pl.seed_everything(args.seed)
    trainer = _get_trainer_from_argparse_args(args)
    train_set, dev_set = _get_datasets_from_argparse_args(args)
    index = get_index(args.model_dir, args.experiment)
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
    model = get_model(train_set, **vars(args))
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
