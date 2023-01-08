"""Prediction."""

import argparse
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
    index: str,
) -> data.Dataset:
    """Creates the dataset.

    Args:
        predict (str).
        config (dataconfig.DataConfig).
        index (str).

    Returns:
        data.Dataset.
    """
    # TODO: Since we don't care about the target column, we should be able to
    # set config.source_col = 0 and avoid the overhead for parsing it.
    # This does not work because the modules expect it to be present even if
    # they ignore it.
    dataset = datasets.get_dataset(predict, config)
    dataset.read_index(index)
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
    collator = collators.Collator(dataset.pad_idx, dataset.config, arch)
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
    dataset = get_dataset(args.predict, config, args.index)
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
    target_sep: str = "",
) -> None:
    """Predicts from the model.

    Args:
         trainer (pl.Trainer).
         model (pl.LightningModule).
         loader (data.DataLoader).
         output (str).
         target_sep (str).
    """
    model.eval()  # TODO: is this necessary?
    dataset = loader.dataset
    util.log_info(f"Writing to {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as sink:
        for batch in trainer.predict(model, dataloaders=loader):
            # TODO: can we move some of this into module `predict_step`
            # methods? I do not understand why it lives here.
            if not (
                isinstance(model, models.TransducerNoFeatures)
                or isinstance(model, models.TransducerFeatures)
            ):
                # -> B x seq_len x vocab_size
                batch = batch.transpose(1, 2)
                _, batch = torch.max(batch, dim=2)
            batch = model.evaluator.finalize_preds(
                batch, dataset.end_idx, dataset.pad_idx
            )
            for prediction in dataset.decode_target(
                batch,
                symbols=True,
                special=False,
            ):
                print(sink, target_sep.join(prediction))


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
    parser.add_argument("--index", required=True, help="Path to index (.pkl)")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint (.ckpt)"
    )
    # Data configuration arguments.
    config = dataconfig.DataConfig.add_argparse_args(parser)
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
    predict(trainer, model, loader, args.output, config.target_sep)


if __name__ == "__main__":
    main()
