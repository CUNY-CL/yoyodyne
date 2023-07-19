#!/usr/bin/env python

"""Gets the run name and config for all runs in a project,
and writes them to a TSV."""

import argparse
from typing import Dict, List

import pandas
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep_id",
        required=True,
        help="ID for the sweep to run the agent in.",
    )
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument(
        "--output", required=True, help="Path for results TSV."
    )
    args = parser.parse_args()
    api = wandb.Api()
    # Computes the path for the run.
    runs = api.sweep(f"{args.entity}/{args.project}/{args.sweep_id}").runs
    summaries: List[Dict] = []
    for run in runs:
        # Ignores running and failed jobs.
        if run.state == "running":
            print(f"{run.id} is still running...")
            continue
        elif run.state != "finished":
            print(f"{run.id} presumably crashed...")
            continue
        run.summary.val_accuracy
        summary = dict(run.summary)
        # Saves the run name.
        summary["name"] = run.name
        # Flattens the nested field here.
        summary["max_val_accuracy"] = summary.pop("val_accuracy")["max"]
        # Removes this non-flat field.
        summary.pop("_wandb")
        # Adds the hyperparameters set by the sweep.
        for key, value in run.config.items():
            if key.startswith("_"):
                continue
            summary[key] = value
        summaries.append(summary)
    runs_df = pandas.DataFrame(summaries)
    runs_df.to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()
