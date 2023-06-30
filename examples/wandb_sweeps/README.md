# W&B Sweeps

Example scripts for making a W&B sweep, training agents in the sweep space, and downloading the results. Read about W&B sweeps [here](https://docs.wandb.ai/guides/sweeps).

- The example in [make_wandb_sweep.py](make_wandb_sweep.py) is just one possible hyparparameter grid. You should edit that file directly to build a hyperparameter grid for your problem.
- Consider also that you can run a bayes search instead of the default random search. See [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#configuration-keys)
- When running [train_wandb_sweep_agent.py](train_wandb_sweep_agent.py) you need the `project_name` and `sweep_id` from an existing sweep. Then, it can be called with the same arguments as `yoyodyne-train` where any hyperparameters in the sweep config will override command-line hyperparameter args.

## Usage

```
# Creates a wandb sweep
python examples/wandb_sweeps/make_wandb_sweep.py --sweep_name foo
# Runs an agent to train on hyperparameters sampled from the sweep
# Defaults to a single run.
python train_wandb_sweep_agent.py --sweep_id bar --experiment baz ...
# Pulls results from the sweep from the wandb API
python get_wandb_results.py --project_name entity/baz --output_filepath output.tsv
```