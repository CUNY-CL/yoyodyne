#!/usr/bin/env python

"""Creates a hyperparameter sweep in the wandb API.

Manually modify functions below with different Dicts to change
the hyperparameters, or sweep method (defaults to random).

See here for details:
https://docs.wandb.ai/guides/sweeps/define-sweep-configuration"""

import argparse

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


def make_sweep(project: str, sweep: str) -> int:
    """Creates the sweep in the wandb API, according to the hyperparameter
    ranges in `HPARAMS`.

    Args:
        project (str): Name of the wandb project.
        sweep (str): Name of the wandb sweep.

    Returns:
        int: The wandb sweep ID for this configuration.
    """
    # TODO: Change search method or metric to maximize/minimize
    sweep_configuration = {
        "method": "random",
        "name": sweep,
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": HPARAMS,
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    return sweep_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Name of the project")
    parser.add_argument("--sweep", required=True, help="Name of the sweep")
    args = parser.parse_args()

    sweep_id = make_sweep(args.project, args.sweep)
    print(f"Made sweep: {sweep_id}")


if __name__ == "__main__":
    main()
