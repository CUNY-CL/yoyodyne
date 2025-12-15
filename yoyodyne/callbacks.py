"""Custom callbacks."""

import csv
import itertools
from typing import Optional, Sequence, TextIO, Tuple, Union

import lightning
from lightning.pytorch import callbacks, trainer
import torch

from . import data, defaults, models, util


class PredictionWriter(callbacks.BasePredictionWriter):
    """Writes predictions.

    Args:
        path: Path for the predictions file.
    """

    path: str
    sink: Optional[TextIO]

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
        self.sink = open(self.path, "w")

    def write_on_batch_end(
        self,
        trainer: trainer.Trainer,
        model: models.BaseModel,
        predictions: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
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
