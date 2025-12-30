"""Command-line interface."""

import logging

from lightning.pytorch import callbacks as pytorch_callbacks, cli
import omegaconf

from .. import callbacks, data, models, trainers


# Register OmegaConf resolvers here.
#
# This allows expressions of the form:
#
# hidden_size: ${multiply:${model.init_args.embedding_size}, 4}
#
# which means that the hidden size is 4x the embedding size.
omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)


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
        models.BaseModel,
        data.DataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
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
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=None,
        subclass_mode_model=True,
        # See above for explanation.
        trainer_class=trainers.Trainer,
        args=args,
    )
