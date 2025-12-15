"""Command-line interface."""

import logging

from lightning.pytorch import callbacks as pytorch_callbacks, cli

from .. import callbacks, data, models, trainers


class YoyodyneCLI(cli.LightningCLI):
    """The Yoyodyne CLI interface.

    Use with `--help` to see the full list of options.
    """

    def add_arguments_to_parser(
        self, parser: cli.LightningArgumentParser
    ) -> None:
        parser.add_lightning_class_args(
            pytorch_callbacks.ModelCheckpoint,
            "checkpoint",
            required=False,
        )
        parser.add_lightning_class_args(
            callbacks.PredictionWriter,
            "prediction",
            required=False,
        )
        parser.link_arguments(
            "data.target_vocab_size",
            "model.init_args.target_vocab_size",
            apply_on="instantiate",
        )
        # Just needed for the transducer to create the expert.
        parser.link_arguments(
            "data.index",
            "model.init_args.index",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.target_sep",
            "prediction.target_sep",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.vocab_size",
            "model.init_args.vocab_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.model_dir",
            "trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    # Select the model.
    YoyodyneCLI(
        model_class=models.BaseModel,
        datamodule_class=data.DataModule,
        save_config_callback=None,
        subclass_mode_model=True,
        # Prevents predictions from accumulating in memory; see the
        # documentation in `trainers.py` for more context.
        trainer_class=trainers.Trainer,
    )


def python_interface(args: cli.ArgsType = None) -> None:
    """Interface to use models through Python."""
    YoyodyneCLI(
        models.BaseModel,
        data.DataModule,
        save_config_callback=None,
        subclass_mode_model=True,
        # See above for explanation.
        trainer_class=trainers.Trainer,
        args=args,
    )
