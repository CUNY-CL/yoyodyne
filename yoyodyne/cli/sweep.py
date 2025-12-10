"""Runs a W&B sweep."""

import argparse
import functools
import logging
import subprocess
import sys
import tempfile
import traceback
import warnings
from typing import Any, Dict, List, TextIO

import wandb
import yaml

from .. import util

warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def train_sweep(
    config: Dict[str, Any],
    links: Dict[str, List[str]],
    temp_config: TextIO,
    argv: List[str],
) -> None:
    """Runs a single training run.

    Args:
        config: config dictionary.
        links: links dictionary.
        temp_config: temporary configuration file handle.
        argv: command-line arguments.
    """
    populate_config(config, links, temp_config)
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
    links: Dict[str, List[str]],
    temp_config_handle: TextIO,
) -> None:
    """Populates temporary configuration file.

    The wandb config data used here comes from the environment.

    Args:
        config: config dictionary.
        links: links dictionary.
        temp_config_handle: temporary configuration file handle.

    Raises:
        KeyError: link not found.
    """
    wandb.init()
    for key, value in wandb.config.items():
        util.recursive_insert(config, key, value)
    for source, dests in links.items():
        value = wandb.config[source]
        for dest in dests:
            logging.info("Linking: %s (%r) -> %s", source, value, dest)
            util.recursive_insert(config, dest, value)
    yaml.safe_dump(config, temp_config_handle)


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
    config, links = util.load_config_and_links(args.config)
    # TODO: Consider enabling the W&B logger; we are not sure if things will
    # unless this is configured.
    temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml")
    argv = [
        args.command,
        "fit",
        "--config",
        temp_config.name,
        *sys.argv[1:],
    ]
    try:
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=functools.partial(
                train_sweep, config, links, temp_config, argv
            ),
            count=args.count,
        )
    except Exception:
        # Exits gracefully, so W&B logs the error.
        logging.fatal(traceback.format_exc())
        wandb.finish(exit_code=1)
        exit(1)
