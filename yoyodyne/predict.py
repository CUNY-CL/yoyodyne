"""Prediction."""

import csv
import os

import click
import pytorch_lightning as pl
import torch
from torch.utils import data

from . import collators, datasets, models, util


def write_predictions(
    model: models.base.BaseEncoderDecoder,
    data_loader: torch.utils.data.DataLoader,
    output_path: str,
    arch: str,
    batch_size: int,
    source_col: int,
    target_col: int,
    features_col: int,
    source_sep: str,
    target_sep: str,
    features_sep: str,
    include_features: bool,
    gpu: bool,
    beam_width: int = None,
) -> None:
    """Writes predictions to output file.

    Args:
        model (models.BaseEncoderDecoder).
        data_loader (torch.utils.data.DataLoader).
        output_path (str).
        arch (str).
        batch_size (int).
        source_col (int).
        target_col (int).
        features_col (int).
        source_sep (str).
        target_sep (str).
        features_sep (str).
        include_features (bool).
        gpu (bool).
        beam_width (int, optional).
    """
    model.beam_width = beam_width
    model.eval()
    # Test loop.
    accelerator = "gpu" if gpu and torch.cuda.is_available() else "cpu"
    tester = pl.Trainer(
        accelerator=accelerator,
        devices=1,
    )
    util.log_info("Predicting...")
    predicted = tester.predict(model, dataloaders=data_loader)
    util.log_info(f"Writing to {output_path}")
    util.log_info(f"Source column: {source_col}")
    util.log_info(f"Target column: {target_col}")
    if features_col:
        util.log_info(f"Features column: {features_col}")
    collator = data_loader.collate_fn
    test_set = data_loader.dataset
    with open(output_path, "w") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        row_template = [""] * max(source_col, target_col, features_col)
        for batch, pred_batch in zip(data_loader, predicted):
            if arch != "transducer":
                # -> B x seq_len x vocab_size
                pred_batch = pred_batch.transpose(1, 2)
                if beam_width is not None:
                    pred_batch = pred_batch.squeeze(2)
                else:
                    _, pred_batch = torch.max(pred_batch, dim=2)
            # Uses CPU because PL seems to always return CPU tensors.
            pred_batch = model.evaluator.finalize_preds(
                pred_batch, test_set.end_idx, test_set.pad_idx, "cpu"
            )
            prediction_strs = test_set.decode_target(
                pred_batch,
                symbols=True,
                special=False,
            )
            source_strs = test_set.decode_source(
                batch[0], symbols=True, special=False
            )
            features_batch = batch[2] if collator.has_features else batch[0]
            features_strs = (
                test_set.decode_features(
                    features_batch, symbols=True, special=False
                )
                if include_features
                else [None for _ in range(batch_size)]
            )
            for source, prediction, features in zip(
                source_strs, prediction_strs, features_strs
            ):
                row = row_template.copy()
                # -1 because we're using base-1 indexing.
                row[source_col - 1] = source_sep.join(source)
                row[target_col - 1] = target_sep.join(prediction)
                if include_features:
                    row[features_col - 1] = features_sep.join(features)
                tsv_writer.writerow(row)
    util.log_info("Prediction complete")


@click.command()
@click.option("--lang", required=True)
@click.option("--data-path", required=True)
@click.option("--source-col", type=int, default=1)
@click.option("--target-col", type=int, default=2)
@click.option(
    "--features-col",
    type=int,
    default=3,
    help="0 indicates that no feature column should be used",
)
@click.option("--source-sep", type=str, default="")
@click.option("--target-sep", type=str, default="")
@click.option("--features-sep", type=str, default=";")
@click.option("--tied-vocabulary", default=True)
@click.option("--output-path", required=True)
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
@click.option("--results-path", required=True)
@click.option("--model-path", required=True)
@click.option("--batch-size", type=int, default=1)
@click.option(
    "--beam-width",
    type=int,
    help="If specified, decode with beam search",
)
@click.option(
    "--attn/--no-attn",
    type=bool,
    default=True,
    help="Use attention (`lstm` only)",
)
@click.option("--bidirectional/--no-bidirectional", type=bool, default=True)
@click.option("--gpu/--no-gpu", default=True)
def main(
    lang,
    data_path,
    source_col,
    target_col,
    features_col,
    source_sep,
    target_sep,
    features_sep,
    tied_vocabulary,
    output_path,
    arch,
    results_path,
    model_path,
    batch_size,
    beam_width,
    attn,
    bidirectional,
    gpu,
):
    """Prediction.

    Args:
        lang (_type_): _description_
        data_path (_type_): _description_
        source_col (_type_): _description_
        target_col (_type_): _description_
        features_col (_type_): _description_
        source_sep (_type_): _description_
        target_sep (_type_): _description_
        features_sep (_type_): _description_
        tied_vocabulary (_type_): _description_
        output_path (_type_): _description_
        arch (_type_): _description_
        results_path (_type_): _description_
        model_path (_type_): _description_
        batch_size (_type_): _description_
        attn (_type_): _description_
        bidirectional (_type_): _description_
        beam_width (_type_): _description_
        gpu (_type_): _description_
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = util.get_device(gpu)
    # TODO: Do not need to enforce once we have batch beam decoding.
    if beam_width is not None:
        util.log_info("Decoding with beam search; forcing batch size to 1")
        batch_size = 1
    include_features = features_col != 0
    dataset_cls = datasets.get_dataset_cls(include_features)
    test_set = dataset_cls(
        data_path,
        tied_vocabulary,
        source_col,
        0,  # Target columns unnecessary.
        source_sep,
        target_sep,
        features_col=features_col,
        features_sep=features_sep,
    )
    test_set.load_index(results_path, lang)
    util.log_info(f"Source vocabulary: {test_set.source_symbol2i}")
    util.log_info(f"Target vocabulary: {test_set.target_symbol2i}")
    collator_cls = collators.get_collator_cls(
        arch, include_features, include_targets=False
    )
    collator = collator_cls(test_set.pad_idx)
    data_loader = data.DataLoader(
        test_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=False,
    )
    # Model.
    model_cls = models.get_model_cls(arch, attn, include_features)
    util.log_info(f"Loading model from {model_path}")
    model = model_cls.load_from_checkpoint(model_path).to(device)
    write_predictions(
        model,
        data_loader,
        output_path,
        arch,
        batch_size,
        source_col,
        target_col,
        features_col,
        source_sep,
        target_sep,
        features_sep,
        include_features,
        gpu,
        beam_width,
    )


if __name__ == "__main__":
    main()
