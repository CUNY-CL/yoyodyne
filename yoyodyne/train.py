"""Training."""

import os
import time

import click
import numpy
import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers
from torch.utils import data

from . import (
    collators,
    dataconfig,
    datasets,
    evaluators,
    models,
    predict,
    util,
)


@click.command()
@click.option("--experiment", required=True)
@click.option("--train", required=True)
@click.option("--dev", required=True)
@click.option("--model-dir", required=True)
@click.option("--dev-predictions")
@click.option(
    "--source-col", type=int, default=1, help="1-based index for source column"
)
@click.option(
    "--target-col", type=int, default=2, help="1-based index for target column"
)
@click.option(
    "--features-col",
    type=int,
    default=0,
    help="1-based index for features column; "
    "0 indicates the model will not use features",
)
@click.option("--source-sep", type=str, default="")
@click.option("--target-sep", type=str, default="")
@click.option("--features-sep", type=str, default=";")
@click.option("--tied-vocabulary/--no-tied-vocabulary", default=True)
@click.option("--dataloader-workers", type=int, default=1)
@click.option(
    "--seed", type=int, default=time.time_ns() % numpy.iinfo(numpy.uint32).max
)
@click.option("--max-epochs", type=int, default=50)
@click.option(
    "--arch",
    type=click.Choice(
        [
            "feature_invariant_transformer",
            "lstm",
            "pointer_generator_lstm",
            "transducer",
            "transformer",
        ]
    ),
    default="lstm",
)
@click.option(
    "--oracle-em-epochs",
    type=int,
    default=5,
    help="Number of EM epochs (`--arch transducer` only)",
)
@click.option(
    "--oracle-factor",
    type=int,
    default=1,
    help="Roll-in schedule parameter (`--arch transducer` only)",
)
@click.option(
    "--sed-params-path",
    type=str,
    default=None,
    help="Path to SED parameters (`transducer` only)",
)
@click.option("--patience", type=int)
@click.option("--learning-rate", type=float, default=0.001)
@click.option("--label-smoothing", type=float)
@click.option("--gradient-clip", type=float)
@click.option("--batch-size", type=int, default=32)
@click.option("--embedding-size", type=int, default=128)
@click.option("--hidden-size", type=int, default=512)
@click.option("--dropout", type=float, default=0.2)
@click.option("--encoder-layers", type=int, default=1)
@click.option("--decoder-layers", type=int, default=1)
@click.option("--max-seq-len", type=int, default=128)
@click.option("--attention-heads", type=int, default=4)
@click.option("--dropout", type=float, default=0.1)
@click.option("--optimizer", default="adam")
@click.option(
    "--beta1",
    default=0.9,
    type=float,
    help="beta1 (`--optimizer adam` only)",
)
@click.option(
    "--beta2",
    default="0.999",
    type=float,
    help="beta2 (`--optimizer adam` only)",
)
@click.option("--warmup-steps", type=int)
@click.option("--scheduler")
@click.option(
    "--train-from", help="Path to checkpoint to continue training from"
)
@click.option("--bidirectional/--no-bidirectional", type=bool, default=True)
@click.option(
    "--attention/--no-attention",
    type=bool,
    default=True,
    help="Use attention (`--arch lstm` only)",
)
@click.option("--max-decode-len", type=int, default=128)
@click.option("--save-top-k", type=int, default=1)
@click.option("--eval-every", type=int, default=2)
@click.option("--wandb/--no-wandb", default=False)
@click.option("--accelerator")
def main(
    experiment,
    train,
    dev,
    model_dir,
    dev_predictions,
    tied_vocabulary,
    source_col,
    target_col,
    features_col,
    source_sep,
    target_sep,
    features_sep,
    dataloader_workers,
    seed,
    max_epochs,
    arch,
    oracle_em_epochs,
    oracle_factor,
    sed_params_path,
    patience,
    learning_rate,
    label_smoothing,
    gradient_clip,
    batch_size,
    embedding_size,
    hidden_size,
    dropout,
    encoder_layers,
    decoder_layers,
    max_seq_len,
    attention_heads,
    optimizer,
    beta1,
    beta2,
    warmup_steps,
    scheduler,
    train_from,
    bidirectional,
    attention,
    max_decode_len,
    save_top_k,
    eval_every,
    wandb,
    accelerator,
):
    """Trainer."""
    util.log_info("Arguments:")
    for arg, val in click.get_current_context().params.items():
        util.log_info(f"\t{arg}: {val!r}")
    pl.seed_everything(seed)
    if config.target_col == 0:
        raise dataconfig.Error("target_col must be specified for training")
    train_set = datasets.get_dataset(train, config)
    util.log_info(f"Source vocabulary: {train_set.source_symbol2i}")
    util.log_info(f"Target vocabulary: {train_set.target_symbol2i}")
    # PL logging.
    logger = [loggers.CSVLogger(model_dir, name=experiment)]
    if wandb:
        logger.append(loggers.WandbLogger(project=experiment, log_model="all"))
    # ckp_callback is used later for logging the best checkpoint path.
    ckp_callback = callbacks.ModelCheckpoint(
        save_top_k=save_top_k,
        monitor="val_accuracy",
        mode="max",
        filename="model-{epoch:02d}-{val_accuracy:.2f}",
    )
    trainer_callbacks = [
        ckp_callback,
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
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=logger,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip,
        check_val_every_n_epoch=eval_every,
        enable_checkpointing=True,
        default_root_dir=model_dir,
        callbacks=trainer_callbacks,
        log_every_n_steps=len(train_set) // batch_size,
        num_sanity_val_steps=0,
    )
    # So we can write indices to it before PL creates it.
    os.makedirs(trainer.loggers[0].log_dir, exist_ok=True)
    train_set.write_index(trainer.loggers[0].log_dir, experiment)
    collator = collators.get_collator(train_set.pad_idx, config, arch)
    # TODO: dataloader indexing Dicts should probably be added to model state.
    train_loader = data.DataLoader(
        train_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
    )
    dev_set = datasets.get_dataset(dev, config)
    dev_set.load_index(trainer.loggers[0].log_dir, experiment)
    dev_loader = data.DataLoader(
        dev_set,
        collate_fn=collator,
        batch_size=2 * batch_size,  # Because we're not collecting gradients.
        shuffle=False,
        num_workers=dataloader_workers,
    )
    evaluator = evaluators.Evaluator()
    model_cls = models.get_model_cls(arch, attention, config.has_features)
    if train_from is not None:
        util.log_info(f"Loading model from {train_from}")
        model = model_cls.load_from_checkpoint(train_from)
        util.log_info("Training...")
        trainer.fit(model, train_loader, dev_loader, ckpt_path=train_from)
    else:
        model = model_cls(
            arch=arch,
            train_set=train_set,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            vocab_size=train_set.source_vocab_size,
            features_vocab_size=getattr(train_set, "features_vocab_size", -1),
            features_idx=getattr(train_set, "features_idx", -1),
            output_size=train_set.target_vocab_size,
            pad_idx=train_set.pad_idx,
            start_idx=train_set.start_idx,
            end_idx=train_set.end_idx,
            optimizer=optimizer,
            beta1=beta1,
            beta2=beta2,
            learning_rate=learning_rate,
            evaluator=evaluator,
            max_decode_len=max_decode_len,
            dropout=dropout,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            label_smoothing=label_smoothing,
            warmup_steps=warmup_steps,
            scheduler=scheduler,
            bidirectional=bidirectional,
            attention_heads=attention_heads,
            max_seq_len=max_seq_len,
            expert=models.expert.get_expert(
                train_set,
                epochs=oracle_em_epochs,
                oracle_factor=oracle_factor,
                sed_params_path=sed_params_path,
            )
            if arch in ["transducer"]
            else None,
        )
        util.log_info("Training...")
        util.log_info(f"Model: {model.__class__.__name__}")
        util.log_info(f"Dataset: {train_set.__class__.__name__}")
        util.log_info(f"Collator: {collator.__class__.__name__}")
        trainer.fit(model, train_loader, dev_loader)
    util.log_info("Training complete")
    util.log_info(
        f"Best model checkpoint path: {ckp_callback.best_model_path}"
    )
    # Writes development set predictions using the best checkpoint,
    # if a predictions path is specified.
    # TODO: Add beam-width option so we can make predictions with beam search.
    if dev_predictions:
        best_model = model_cls.load_from_checkpoint(
            ckp_callback.best_model_path
        )
        predict.write_predictions(
            best_model,
            dev_loader,
            dev_predictions,
            arch,
            batch_size,
            config,
        )


if __name__ == "__main__":
    main()
