# W&B Sweeps

This directory contains example scripts for running a hyperparameter sweep with
[Weights & Biases](https://wandb.ai/site).

-   [`config.yaml`](config.yaml) contains just one possible hyparparameter
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

## Usage

Execute the following to create and run the sweep; here `${ENTITY}` and
`${PROJECT}` are assumed to be pre-specified environmental variables.

    # Creates a sweep; save the sweep ID as ${SWEEP_ID} for later.
    wandb sweep --entity "${ENTITY}" --project "${PROJECT}" config.yaml
    # Runs the sweep itself.
    ./train_wandb_sweep.py --entity "${ENTITY}" --project "${PROJECT}" \
         --sweep_id "${SWEEP_ID}" --count "${COUNT}" ...

Then, one can retrieve the results as follows:

1.  Visit the following URL:
    `https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}`

2.  Switch to "table view" by either clicking on the spreadsheet icon in the top
    left or typing Ctrl+J.

3.  Click on the downward arrow link, select "CSV Export", then click "Save as
    CSV".
