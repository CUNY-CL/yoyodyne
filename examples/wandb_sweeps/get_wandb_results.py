"""Gets the run name and config for all runs in a project, and writes them to a CSV."""

import pandas as pd 
import wandb
import click


@click.command()
@click.option("--project_name", help="Name of the wandb project. "
              "Should be of the format <entity/project-name>.")
@click.option("--output_filepath", help="Path to store CSV of results.")
def main(project_name, output_filepath):
    api = wandb.Api()
    runs = api.runs(project_name)
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

        config = {
            k: v for k, v in run.config.items() if not k.startswith('_')
        }
        run_dict = config
        run_dict.update(summary)
        run_dict["name"] = run.name
        run_dicts.append(run_dict)

    runs_df = pd.DataFrame(run_dicts)
    runs_df.to_csv(output_filepath)


if __name__ == "__main__":
    main()