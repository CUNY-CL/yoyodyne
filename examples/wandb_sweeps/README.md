W&B Sweeps
==========

This directory contains example scripts for running a hyperparameter sweep with
[Weights & Biases](https://wandb.ai/site).

-   [`config.yaml`](config.yaml.py) contains just one possible hyparparameter
    grid, designed for random search over an attentive LSTM; edit that file
    directly to build a hyperparameter grid appropriate for your problem.
-   Consider also running Bayesian search (`method: bayes`) instead of random
    search; see
    [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#configuration-keys)
    for more information.
-   When running [`train_wandb_sweep.py`](train_wandb_sweep.py) you must provide
    the `--entity`, `--project` and `--sweep_id`. It can otherwise be called
    with the same arguments as `yoyodyne-train` where any hyperparameters in the
    sweep config will override command-line hyperparameter arguments.
-   By default `random` and `bayes` search run indefinitely, until they are
    killed. To specify a fixed number of samples, provide the `--count` argument
    to [`train_wandb_sweep.py`](train_wandb_sweep.py).

For more information about W&B sweeps, [read
here](https://docs.wandb.ai/guides/sweeps).

Usage
-----

    # Creates a sweep; save the sweep ID for later.
    wandb sweep --entity nasa --project apollo config.yaml
    # Runs the sweep itself.
    ./train_wandb_sweep.py --entity nasa --project apollo --sweep_id ...
    # Retrieves results.
    ./get_wandb_results.py --entity nasa --project apollo --output results.tsv
