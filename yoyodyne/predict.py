"""Prediction."""

import argparse
import csv
import os

import pytorch_lightning as pl
import torch
from torch.utils import data

from . import collators, dataconfig, datasets, models, util


def get_trainer(**kwargs) -> pl.Trainer:
    """Creates the trainer.

    Args:
        **kwargs: passed to the trainer.

    Returns:
        pl.Trainer.
    """
    return pl.Trainer(max_epochs=0, **kwargs)


def _get_trainer_from_argparse_args(
    args: argparse.Namespace,
) -> pl.Trainer:
    """Creates the trainer from CLI arguments."""
    return pl.Trainer.from_argparse_args(args, max_epochs=0)


def get_dataset(
    predict: str,
    config: dataconfig.DataConfig,
    experiment: str,
    model_dir: str,
) -> data.Dataset:
    """Creates the dataset.

    Args:
        predict (str).
        config (dataconfig.DataConfig).
        experiment (str).
        model_dir (str).

    Returns:
        data.Dataset.
    """
    # TODO: Since we don't care about the target column, we should be able to
    # set config.source_col = 0 and avoid the overhead for parsing it.
    # This does not work because the modules expect it to be present even if
    # they ignore it.
    dataset = datasets.get_dataset(predict, config)
    dataset.load_index(model_dir, experiment)
    return dataset


def get_loader(
    dataset: data.Dataset,
    arch: str,
    batch_size: int,
) -> data.DataLoader:
    """Creates the loader.

    Args:
        dataset (data.Dataset).
        arch (str).
        batch_size (int).

    Returns:
        data.DataLoader.
    """
    collator = collators.get_collator(dataset.pad_idx, dataset.config, arch)
    return data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        num_workers=1,  # Our data loading is simple.
    )


def _get_loader_from_argparse_args(
    args: argparse.Namespace,
) -> data.DataLoader:
    """Creates the loader from CLI arguments."""
    config = dataconfig.DataConfig.from_argparse_args(args)
    dataset = get_dataset(
        args.predict, config, args.experiment, args.model_dir
    )
    return get_loader(dataset, args.arch, args.batch_size)


def get_model(
    arch: str,
    attention: bool,
    has_features: bool,
    checkpoint: str,
) -> pl.LightningModule:
    """Creates the model from checkpoint.

    Args:
        arch (str).
        attention (bool).
        has_features (bool).
        checkpoint (str).

    Returns:
        pl.Lightningmodule.
    """
    model_cls = models.get_model_cls(arch, attention, has_features)
    return model_cls.load_from_checkpoint(checkpoint)


def _get_model_from_argparse_args(
    args: argparse.Namespace,
) -> pl.LightningModule:
    """Creates the model from CLI arguments."""
    model_cls = models.get_model_cls_from_argparse_args(args)
    return model_cls.load_from_checkpoint(args.checkpoint)


def predict(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    loader: data.DataLoader,
    output: str,
) -> None:
    """Predicts from the model.

    Args:
         trainer (pl.Trainer).
         model (pl.LightningModule).
         loader (data.DataLoader).
         output (str)
    """
    model.eval()  # TODO: is this necessary?
    predictions = trainer.predict(model, dataloaders=loader)
    dataset = loader.dataset
    config = dataset.config
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        for batch, prediction_batch in zip(loader, predictions):
            # TODO: Can we move some of this logic into the module
            # `predict_step` methods? I do not understand why it lives here.
            if not (
                isinstance(model, models.TransducerNoFeatures)
                or isinstance(model, models.TransducerFeatures)
            ):
                # -> B x seq_len x vocab_size.
                prediction_batch = prediction_batch.transpose(1, 2)
                _, prediction_batch = torch.max(prediction_batch, dim=2)
            prediction_batch = model.evaluator.finalize_predictions(
                prediction_batch, dataset.end_idx, dataset.pad_idx
            )
            prediction_strs = dataset.decode_target(
                prediction_batch,
                symbols=True,
                special=False,
            )
            source_strs = dataset.decode_source(
                batch[0], symbols=True, special=False
            )
            features_batch = batch[2] if config.has_features else batch[0]
            features_strs = (
                dataset.decode_features(
                    features_batch, symbols=True, special=False
                )
                if config.has_features
                else [None for _ in range(loader.batch_size)]
            )
            for source, prediction, features in zip(
                source_strs, prediction_strs, features_strs
            ):
                tsv_writer.writerow(
                    config.get_row(source, prediction, features)
                )


def main() -> None:
    """Predictor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", required=True, help="Name of experiment"
    )
    # Path arguments.
    parser.add_argument(
        "--predict",
        required=True,
        help="Path to prediction input data TSV",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to prediction output data TSV",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to output model directory",
    )
    parser.add_argument("--checkpoint", help="Path to ckpt checkpoint")
    # Data configuration arguments.
    dataconfig.DataConfig.add_argparse_args(parser)
    # Architecture arguments.
    models.add_argparse_args(parser)
    # Predicting arguments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size. Default: %(default)s.",
    )
    # TODO: add --beam_width.
    # Among the things this adds, the following are likely to be useful:
    # --accelerator ("gpu" for GPU)
    # --devices (for multiple device support)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    util.log_arguments(args)
    trainer = _get_trainer_from_argparse_args(args)
    loader = _get_loader_from_argparse_args(args)
    model = _get_model_from_argparse_args(args)
    # TODO: We right now assume that the input config and the output config
    # are the same. However, this may not be desirable. It is an open question
    # whether we ought to do anything about this. One possible solution is to
    # generate only the target side and put aside the rest.
    predict(trainer, model, loader, args.output)


if __name__ == "__main__":
    main()
