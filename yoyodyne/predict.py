"""Prediction."""

import argparse
import os

import pytorch_lightning as pl
from torch.utils import data as torch_data

from . import dataconfig, data, defaults, models, util


def get_trainer_from_argparse_args(
    args: argparse.Namespace,
) -> pl.Trainer:
    """Creates the trainer from CLI arguments.

    Args:
        args (argparse.Namespace).

    Return:
        pl.Trainer.
    """
    return pl.Trainer.from_argparse_args(args, max_epochs=0)


def get_dataset_from_argparse_args(
    args: argparse.Namespace,
) -> data.BaseDataset:
    """Creates the dataset from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        data.BaseDataset.
    """
    config = dataconfig.DataConfig.from_argparse_args(args)
    return data.get_dataset(args.predict, config, args.index)


def get_loader(
    dataset: data.BaseDataset,
    arch: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
) -> torch_data.DataLoader:
    """Creates the loader.

    Args:
        dataset (data.Dataset).
        arch (str).
        batch_size (int).
        max_source_length (int).
        max_target_length (int).

    Returns:
        torch_data.DataLoader.
    """
    collator = data.Collator(
        dataset,
        arch,
        max_source_length,
        max_target_length,
    )
    return torch_data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        num_workers=1,
    )


def get_model_from_argparse_args(
    args: argparse.Namespace,
) -> models.BaseEncoderDecoder:
    """Creates the model from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        models.BaseEncoderDecoder.
    """
    model_cls = models.get_model_cls_from_argparse_args(args)
    return model_cls.load_from_checkpoint(args.checkpoint)


def _mkdir(output: str) -> None:
    """Creates directory for output file if necessary.

    Args:
        output (str): output to output file.
    """
    dirname = os.path.dirname(output)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def predict(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    loader: torch_data.DataLoader,
    output: str,
) -> None:
    """Predicts from the model.

    Args:
         trainer (pl.Trainer).
         model (pl.LightningModule).
         loader (torch_data.DataLoader).
         output (str).
         target_sep (str).
    """
    dataset = loader.dataset
    target_sep = dataset.config.target_sep
    util.log_info(f"Writing to {output}")
    _mkdir(output)
    with open(output, "w") as sink:
        for batch in trainer.predict(model, dataloaders=loader):
            batch = model.evaluator.finalize_predictions(
                batch, dataset.index.end_idx, dataset.index.pad_idx
            )
            for prediction in dataset.decode_target(
                batch,
                symbols=True,
                special=False,
            ):
                print(target_sep.join(prediction), file=sink)


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds prediction arguments to parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--experiment", required=True, help="Name of experiment."
    )
    # Path arguments.
    parser.add_argument(
        "--predict",
        required=True,
        help="Path to prediction input data TSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to prediction output data TSV.",
    )
    parser.add_argument("--index", required=True, help="Path to index (.pkl).")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint (.ckpt)."
    )
    # Predicting arguments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=defaults.BATCH_SIZE,
        help="Batch size. Default: %(default)s.",
    )
    # TODO: add --beam_width.
    # Data configuration arguments.
    dataconfig.DataConfig.add_argparse_args(parser)
    # Data arguments.
    data.add_argparse_args(parser)
    # Architecture arguments; the architecture-specific ones are not needed.
    models.add_argparse_args(parser)
    # Among the things this adds, the following are likely to be useful:
    # --accelerator ("gpu" for GPU)
    # --devices (for multiple device support)
    pl.Trainer.add_argparse_args(parser)


def main() -> None:
    """Predictor."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_argparse_args(parser)
    args = parser.parse_args()
    util.log_arguments(args)
    trainer = get_trainer_from_argparse_args(args)
    loader = get_loader(
        get_dataset_from_argparse_args(args),
        args.arch,
        args.batch_size,
        args.max_source_length,
        args.max_target_length,
    )
    model = get_model_from_argparse_args(args)
    predict(trainer, model, loader, args.output)


if __name__ == "__main__":
    main()
