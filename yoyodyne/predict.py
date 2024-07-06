"""Prediction."""

import argparse
import os

import pytorch_lightning as pl

from . import data, models, util


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


def get_datamodule_from_argparse_args(
    args: argparse.Namespace,
) -> data.DataModule:
    """Creates the dataset from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        data.DataModule.
    """
    separate_features = args.features_col != 0 and args.arch in [
        "pointer_generator_lstm",
        "pointer_generator_transformer",
        "transducer",
    ]
    index = data.Index.read(args.model_dir, args.experiment)
    return data.DataModule(
        predict=args.predict,
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
        index=index,
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
    model: models.BaseEncoderDecoder,
    datamodule: data.DataModule,
    output: str,
) -> None:
    """Predicts from the model.

    Args:
         trainer (pl.Trainer).
         model (pl.LightningModule).
         datamodule (data.DataModule).
         output (str).
    """
    util.log_info(f"Writing to {output}")
    _mkdir(output)
    loader = datamodule.predict_dataloader()
    with open(output, "w") as sink:
        for batch in trainer.predict(model, loader):
            batch = util.pad_tensor_after_eos(
                batch, datamodule.index.end_idx, datamodule.index.pad_idx
            )
            for prediction in loader.dataset.decode_target(batch):
                print(prediction, file=sink)


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds prediction arguments to parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    # Path arguments.
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint (.ckpt)."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to output model directory.",
    )
    parser.add_argument(
        "--experiment", required=True, help="Name of experiment."
    )
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
    # Prediction arguments.
    # TODO: add --beam_width.
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
    datamodule = get_datamodule_from_argparse_args(args)
    model = get_model_from_argparse_args(args)
    predict(trainer, model, datamodule, args.output)


if __name__ == "__main__":
    main()
