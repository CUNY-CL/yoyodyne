"""Prediction."""

import csv
import os

import click
import pytorch_lightning as pl
import torch
from torch.utils import data

from . import collators, dataconfig, datasets, models, util


def write_predictions(
    model: models.base.BaseEncoderDecoder,
    loader: torch.utils.data.DataLoader,
    output: str,
    arch: str,
    accelerator: str,
    batch_size: int,
    config: dataconfig.DataConfig,
    beam_width: int = None,
) -> None:
    """Writes predictions to output file.

    Args:
        model (models.BaseEncoderDecoder).
        loader (torch.utils.data.DataLoader).
        output (str).
        arch (str).
        accelerator (str).
        batch_size (int).
        config (dataconfig.DataConfig).
        beam_width (int, optional).
    """
    model.beam_width = beam_width
    model.eval()
    # Test loop.
    tester = pl.Trainer(
        accelerator=accelerator,
        max_epochs=0,  # Silences a warning.
    )
    util.log_info("Predicting...")
    predicted = tester.predict(model, dataloaders=loader)
    dataset = loader.dataset
    util.log_info(f"Writing to {output}")
    with open(output, "w") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        for batch, pred_batch in zip(loader, predicted):
            if arch != "transducer":
                # -> B x seq_len x vocab_size
                pred_batch = pred_batch.transpose(1, 2)
                if beam_width is not None:
                    pred_batch = pred_batch.squeeze(2)
                else:
                    _, pred_batch = torch.max(pred_batch, dim=2)
            pred_batch = model.evaluator.finalize_preds(
                pred_batch, dataset.end_idx, dataset.pad_idx
            )
            prediction_strs = dataset.decode_target(
                pred_batch,
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
                else [None for _ in range(batch_size)]
            )
            for source, prediction, features in zip(
                source_strs, prediction_strs, features_strs
            ):
                tsv_writer.writerow(
                    config.make_row(source, prediction, features)
                )
    util.log_info("Prediction complete")


@click.command()
@click.option("--experiment", required=True)
@click.option("--predict", required=True)
@click.option("--output", required=True)
@click.option("--model-dir", required=True)
@click.option("--checkpoint", required=True)
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
@click.option("--tied-vocabulary", default=True)
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
    required=True,
)
@click.option("--batch-size", type=int, default=1)
@click.option(
    "--beam-width", type=int, help="If specified, beam search is used"
)
@click.option(
    "--attention/--no-attention",
    type=bool,
    default=True,
    help="Use attention (`lstm` only)",
)
@click.option("--bidirectional/--no-bidirectional", type=bool, default=True)
@click.option("--accelerator")
def main(
    experiment,
    predict,
    output,
    model_dir,
    checkpoint,
    source_col,
    target_col,
    features_col,
    source_sep,
    target_sep,
    features_sep,
    tied_vocabulary,
    arch,
    batch_size,
    beam_width,
    attention,
    bidirectional,
    accelerator,
):
    """Predictor."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    config = dataconfig.DataConfig(
        source_col=source_col,
        features_col=features_col,
        target_col=target_col,
        source_sep=source_sep,
        features_sep=features_sep,
        target_sep=target_sep,
        tied_vocabulary=tied_vocabulary,
    )
    # TODO: Do not need to enforce once we have batch beam decoding.
    if beam_width is not None:
        util.log_info("Decoding with beam search; forcing batch size to 1")
        batch_size = 1
    dataset_cls = datasets.get_dataset_cls(config.has_features)
    dataset = dataset_cls(predict, config)
    dataset.load_index(model_dir, experiment)
    util.log_info(f"Source vocabulary: {dataset.source_symbol2i}")
    util.log_info(f"Target vocabulary: {dataset.target_symbol2i}")
    collator_cls = collators.get_collator_cls(
        arch, config.has_features, include_targets=False
    )
    collator = collator_cls(dataset.pad_idx)
    loader = data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=False,
    )
    # Model.
    model_cls = models.get_model_cls(arch, attention, config.has_features)
    util.log_info(f"Loading model from {checkpoint}")
    model = model_cls.load_from_checkpoint(checkpoint)
    write_predictions(
        model,
        loader,
        output,
        arch,
        accelerator,
        batch_size,
        config,
        beam_width,
    )


if __name__ == "__main__":
    main()
