"""Gets the run name and config for all runs in a project,
and writes them to a TSV."""

import pandas as pd
import wandb
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project_name",
        required=True,
        help="Name of the wandb project. "
        "Should be of the format <entity/project-name>.",
    )
    parser.add_argument(
        "--output_filepath",
        required=True,
        help="Path to store CSV of results.",
    )
    args = parser.parse_args()
    api = wandb.Api()
    runs = api.runs(args.project_name)
    run_dicts = []
    for run in runs:
        # Ignores running and failed jobs.
        if run.state == "running":
            print(f"{run.id} is still running...")
            continue
        elif run.state != "finished":
            print(f"{run.id} presumably crashed...")
            continue

        summary = run.summary._json_dict
        val_acc = summary.pop("val_accuracy")["max"]
        summary["max_val_accuracy"] = val_acc
        summary.pop("_wandb")

        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        run_dict = config
        run_dict.update(summary)
        run_dict["name"] = run.name
        run_dicts.append(run_dict)

    runs_df = pd.DataFrame(run_dicts)
    runs_df.to_csv(args.output_filepath, sep="\t")


if __name__ == "__main__":
    main()
