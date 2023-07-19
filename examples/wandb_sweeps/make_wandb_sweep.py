#!/usr/bin/env python
"""Creates a hyperparameter sweep in the wandb API.

Manually modify functions below with different Dicts to change
the hyperparameters, or sweep method (defaults to random).

See here for details:

https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
"""

import argparse
from typing import Optional

import wandb

# TODO: Update the hyperparameter grid with new hyperparameters
# and value distributions.
HPARAMS = {
    "embedding_size": {
        "distribution": "q_uniform",
        "q": 16,
        "min": 16,
        "max": 1024,
    },
    "hidden_size": {
        "distribution": "q_uniform",
        "q": 16,
        "min": 16,
        "max": 1024,
    },
    "dropout": {
        "distribution": "uniform",
        "min": 0,
        "max": 0.5,
    },
    "batch_size": {
        "distribution": "q_uniform",
        "q": 16,
        "min": 16,
        "max": 128,
    },
    "learning_rate": {
        "distribution": "log_uniform_values",
        "min": 0.00001,
        "max": 0.01,
    },
    "label_smoothing": {"distribution": "uniform", "min": 0.0, "max": 0.2},
}


def make_sweep(
    method: str,
    entity: Optional[str] = None,
    project: Optional[str] = None,
) -> int:
    """Creates the sweep in the wandb API.

    This uses the hyperparameter in `HPARAMS`.

    See here for documentation on the interpretation of the arguments:

        https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
        https://docs.wandb.ai/ref/python/sweep

    Args:
        method (str): Search strategy.
        entity (str, optional).
        project (str, optional).

    Returns:
        int: The wandb sweep ID for this configuration.
    """
    # TODO: Change search method or metric to maximize/minimize
    config = {
        "method": method,
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": HPARAMS,
    }
    return wandb.sweep(config, entity, project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["grid", "random", "bayes"],
        default="random",
        help="Search strategy",
    )
    parser.add_argument("--entity", required=True, help="Entity name")
    parser.add_argument("--project", required=True, help="Project name")
    args = parser.parse_args()
    # This returns an integer but it's already been logged.
    make_sweep(args.method, args.entity, args.project)


if __name__ == "__main__":
    main()
