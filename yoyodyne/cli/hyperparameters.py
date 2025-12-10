"""Retrieves hyperparameters from a W&B run."""

import argparse
import logging
from typing import Any, Dict

import wandb
import yaml

from .. import util

# Top-level categories allowed in Lightning CLI configs.
ALLOWED_TOP_LEVEL_CATEGORIES = frozenset(
    ["data", "model", "trainer", "checkpoint", "prediction"]
)


def dot_to_nested_dict(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Converts dot notation dictionary to nested dictionary.

    Args:
        flat_dict: Dictionary with dot-separated keys.

    Returns:
        Nested dictionary with proper structure.
    """
    nested = {}
    for key, value in flat_dict.items():
        if (
            any(
                key.startswith(f"{cat}.")
                for cat in ALLOWED_TOP_LEVEL_CATEGORIES
            )
            or key in ALLOWED_TOP_LEVEL_CATEGORIES
        ):
            util.recursive_insert(nested, key, value)
    return nested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_path", help="Run path (in the form entity/project/run_id)"
    )
    args = parser.parse_args()
    api = wandb.Api()
    run = api.run(f"{args.run_path}")
    logging.info("Run URL: %s", run.url)
    # Gets all config values from this run.
    flat_config = {
        key: value
        for key, value in sorted(run.config.items())
        if value is not None and "/" not in key
    }
    # Converts to nested dictionary structure.
    config = dot_to_nested_dict(flat_config)
    print(
        yaml.safe_dump(config, default_flow_style=False, sort_keys=False),
        end="",
    )
