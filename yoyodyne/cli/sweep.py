"""Runs a W&B sweep."""

import argparse
import functools
import logging
import subprocess
import sys
import tempfile
import traceback
import warnings
from typing import Any, Dict, List

import wandb
import yaml

from .. import util

warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def train_sweep(
    config: Dict[str, Any],
    argv_template: List[str],
) -> None:
    """Runs a single training run.

    Args:
        config: config dictionary (base config, will be copied).
        argv_template: command-line arguments template with None placeholder
            for the config path in 4th position.
    """
    # Create a fresh temp file for this run
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_config:
        populate_config(config.copy(), temp_config)
        temp_config_path = temp_config.name
    # Fills in config path at known position.
    argv = argv_template.copy()
    assert argv[3] is None
    argv[3] = temp_config_path
    run_sweep(argv)


def run_sweep(argv: List[str]) -> None:
    """Actually runs the sweep.

    Args:
        argv: command-line arguments.

    We encapsulate each run by using a separate subprocess, which ought to
    ensure that memory is returned (etc.).
    """
    process = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)
    for line in process.stderr:
        logging.info(line.rstrip())
    wandb.finish(exit_code=process.wait())


def populate_config(
    config: Dict[str, Any],
    temp_config_handle,
) -> None:
    """Populates temporary configuration file.

    The wandb config data used here comes from the environment.

    Args:
        config: config dictionary (will be modified in place).
        temp_config_handle: temporary configuration file handle.
    """
    wandb.init()
    for key, value in wandb.config.items():
        util.recursive_insert(config, key, value)
    yaml.safe_dump(config, temp_config_handle)
    temp_config_handle.flush()


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_id", required=True, help="ID for the sweep.")
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument(
        "--command",
        default="yoyodyne",
        help="Command to use when running sweep (default: %(default)s).",
    )
    parser.add_argument("--count", type=int, help="Number of runs to perform.")
    parser.add_argument("--config", required=True)
    # We pass the known args to main but remove them from ARGV.
    # See: https://docs.python.org/3/library/argparse.html#partial-parsing
    # This allows the user to override config arguments with CLI arguments.
    args, sys.argv[1:] = parser.parse_known_args()
    config = util.load_config(args.config)
    # Creates argv template with None placeholder for config path.
    argv_template = [
        args.command,
        "fit",
        "--config",
        None,  # Placeholder to be replaced in each run.
        *sys.argv[1:],
    ]
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=functools.partial(train_sweep, config, argv_template),
            count=args.count,
        )
    except Exception:
        # Exits gracefully, so W&B logs the error.
        logging.fatal(traceback.format_exc())
        wandb.finish(exit_code=1)
        exit(1)
