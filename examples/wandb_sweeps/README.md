# W&B Sweeps

This directory contains example scripts for running a hyperparameter sweep with [Weights & Biases](https://wandb.ai/site).

- [`make_wandb_sweep.py`](make_wandb_sweep.py) contains just one possible hyparparameter grid; edit that file directly to build a hyperparameter grid appropriate for your problem.
- Consider also running Bayesian search instead of the default random search; see [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#configuration-keys) for more information.
- When running [`train_wandb_sweep_agent.py`](train_wandb_sweep_agent.py) you must provide the `project_name` and `sweep_id` from an existing sweep. It can be called with the same arguments as `yoyodyne-train` where any hyperparameters in the sweep config will override command-line hyperparameter arguments.

For more information about W&B sweeps, [read here](https://docs.wandb.ai/guides/sweeps).

## Usage

```
# Creates a sweep; save the sweep ID for later.
./make_wandb_sweep.py --project apollo --sweep engine
# Runs the sweep itself.
./train_wandb_sweep.py --sweep_id ... 
#python get_wandb_results.py --project_name entity/baz --output_filepath output.tsv
```
