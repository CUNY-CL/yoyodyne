"""Custom callbacks."""

import csv
import itertools
from typing import Sequence, TextIO

import lightning
import torch
from lightning.pytorch import callbacks, trainer
from lightning.pytorch.utilities import model_summary
from rich import console, table

from . import data, defaults, models, util


class CompactModelSummary(callbacks.ModelSummary):
    """ModelSummary callback that hides zero-parameter rows."""

    _console = console.Console()

    def on_fit_start(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        # Prevents redundant logging in multi-GPU setups.
        if not trainer.is_global_zero:
            return
        param_table = table.Table(
            show_header=True, header_style="bold magenta"
        )
        param_table.add_column("Name")
        param_table.add_column("Type")
        param_table.add_column("Parameters", justify="right")
        summary = model_summary.ModelSummary(pl_module)
        for name, layer in summary._layer_summary.items():
            if layer.num_parameters == 0:
                continue
            param_table.add_row(
                name, layer.layer_type, f"{layer.num_parameters:,}"
            )
        self._console.print(param_table)
        self._console.print(
            f"Trainable parameters: {summary.trainable_parameters:,}"
        )


class PredictionWriter(callbacks.BasePredictionWriter):
    """Callback for writing out predictions.

    Args:
        path (str): string path for the predictions file.
        target_sep (str):
    """

    path: str
    sink: TextIO | None

    # path is given a default argument to silence a warning if no prediction
    # callback is configured.
    # TODO: remove default if this is addressed:
    #
    #   https://github.com/Lightning-AI/pytorch-lightning/issues/20851
    def __init__(
        self,
        path: str = "",
        target_sep: str = defaults.TARGET_SEP,
    ):
        super().__init__("batch")
        self.path = path
        self.sink = None
        self.target_sep = target_sep

    # Required API.

    def on_predict_start(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        # Placing this here prevents the creation of an empty file in the case
        # where a prediction callback was specified but this is not running
        # in predict mode.
        util.mkpath(self.path)
        self.sink = open(self.path, "w", encoding=defaults.ENCODING)

    def write_on_batch_end(
        self,
        trainer: trainer.Trainer,
        model: models.BaseModel,
        predictions: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        batch_indices: Sequence[int] | None,
        batch: data.Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        mapper = data.Mapper(trainer.datamodule.index)
        if hasattr(model, "beam_width") and model.beam_width > 1:
            # Beam search.
            tsv_writer = csv.writer(self.sink, delimiter="\t")
            # Even though beam search currently assumes batch size of 1,
            # this assumption is not baked-in here and should generalize
            # if this restriction is lifted.
            for beam, beam_scores in zip(*predictions):
                beam_strings = [
                    self.target_sep.join(mapper.decode_target(prediction))
                    for prediction in beam
                ]
                # Collates target strings and their scores.
                row = itertools.chain.from_iterable(
                    zip(beam_strings, beam_scores.tolist())
                )
                tsv_writer.writerow(row)
        else:
            # Greedy search.
            for prediction in predictions:
                print(
                    self.target_sep.join(mapper.decode_target(prediction)),
                    file=self.sink,
                )
        self.sink.flush()

    def on_predict_end(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self.sink.close()
