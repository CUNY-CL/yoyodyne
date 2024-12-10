"""Prediction."""

import argparse
import csv
import itertools

import lightning

from . import data, defaults, models, util


def get_trainer_from_argparse_args(
    args: argparse.Namespace,
) -> lightning.Trainer:
    """Creates the trainer from CLI arguments.

    Args:
        args (argparse.Namespace).

    Return:
        lightning.Trainer.
    """
    return lightning.Trainer.from_argparse_args(args, max_epochs=0)


def get_datamodule_from_argparse_args(
    args: argparse.Namespace,
) -> data.DataModule:
    """Creates the datamoodule from CLI arguments.

    Args:
        args (argparse.Namespace).

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
    return data.DataModule(
        batch_size=args.batch_size,
        features_col=args.features_col,
        features_sep=args.features_sep,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        model_dir=args.model_dir,
        predict=args.predict,
        separate_features=separate_features,
        source_col=args.source_col,
        source_sep=args.source_sep,
        target_col=args.target_col,
        target_sep=args.target_sep,
    )


def get_model_from_argparse_args(
    args: argparse.Namespace,
) -> models.BaseModel:
    """Creates the model from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        models.BaseModel.
    """
    model_cls = models.get_model_cls_from_argparse_args(args)
    kwargs = {}
    if args.beam_width:
        kwargs["beam_width"] = args.beam_width
    # Passes kwargs when loading the model.
    return model_cls.load_from_checkpoint(args.checkpoint, **kwargs)


def predict(
    trainer: lightning.Trainer,
    model: models.BaseModel,
    datamodule: data.DataModule,
    output: str,
) -> None:
    """Predicts from the model.

    Args:
         trainer (lightning.Trainer).
         model (lightning.LightningModule).
         datamodule (data.DataModule).
         output (str).
    """
    util.log_info(f"Writing to {output}")
    util.mkpath(output)
    loader = datamodule.predict_dataloader()
    parser = datamodule.parser
    mapper = data.Mapper(datamodule.index)
    with open(output, "w", encoding=defaults.ENCODING) as sink:
        if model.beam_width > 1:
            # Beam search.
            tsv_writer = csv.writer(sink, delimiter="\t")
            for batch_predictions, batch_scores in trainer.predict(
                model, loader
            ):
                # Even though beam search currently assumes batch size of 1,
                # this assumption is not baked-in here and should generalize
                # if this restriction is lifted.
                for beam, beam_scores in zip(batch_predictions, batch_scores):
                    beam_strings = [
                        parser.target_string(mapper.decode_target(prediction))
                        for prediction in beam
                    ]
                    # Collates target strings and their scores.
                    row = itertools.chain.from_iterable(
                        zip(beam_strings, beam_scores.tolist())
                    )
                    tsv_writer.writerow(row)
        else:
            # Greedy search.
            for batch in trainer.predict(model, loader):
                for prediction in batch:
                    print(
                        parser.target_string(mapper.decode_target(prediction)),
                        file=sink,
                    )


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds prediction arguments to parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    # Path arguments.
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to output model directory.",
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
    # Data arguments.
    data.add_argparse_args(parser)
    # Architecture arguments; the architecture-specific ones are not needed.
    models.add_argparse_args(parser)
    models.BaseModel.add_predict_argparse_args(parser)
    # Among the things this adds, the following are likely to be useful:
    # --accelerator ("gpu" for GPU)
    # --devices (for multiple device support)
    lightning.Trainer.add_argparse_args(parser)


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
