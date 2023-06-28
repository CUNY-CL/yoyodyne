# W&B Sweeps

Example scripts for making a W&B sweep, training agents in the sweep space, and downloading the results. Read about W&B sweeps [here](https://docs.wandb.ai/guides/sweeps).

When running `train_wandb_sweep_agent` you need the `project_name` and `sweep_id` from an existing sweep. Then, it can be called with the same arguments as `yoyodyne-train`, where any hyperparameters in the sweep config will override command-line hyperparameter args.